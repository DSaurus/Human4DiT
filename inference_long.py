from copy import deepcopy

import colossalai
import torch
import torch.distributed as dist
import wandb
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin, GeminiPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from tqdm import tqdm
import os

from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)
from opensora.acceleration.plugin import ZeroSeqParallelPlugin
from opensora.datasets import DatasetFromCSV, DatasetIMGFromCSV, get_transforms_image, get_transforms_video, prepare_dataloader
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import create_logger, load, model_sharding, record_model_param_shape, save, load_from_sharded_state_dict
from opensora.utils.config_utils import (
    create_experiment_workspace,
    create_tensorboard_writer,
    parse_configs,
    save_training_config,
)
from opensora.utils.misc import all_reduce_mean, format_numel_str, get_model_numel, requires_grad, to_torch_dtype
from opensora.utils.train_utils import update_ema

from opensora.datasets import save_sample
from diffusers import EulerDiscreteScheduler, StableVideoDiffusionPipeline, DDPMScheduler, DDIMScheduler


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    cfg = parse_configs(training=True)
    print(cfg)
    exp_name, exp_dir = create_experiment_workspace(cfg)
    save_training_config(cfg._cfg_dict, exp_dir)

    # ======================================================
    # 2. runtime variables & colossalai launch
    # ======================================================
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # assert cfg.dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"q

    # 2.1. colossalai init distributed training
    # colossalai.launch_from_torch({})
    # coordinator = DistCoordinator()
    device = get_current_device()
    dtype = to_torch_dtype(cfg.dtype)

    # 2.3. initialize ColossalAI booster
    # if cfg.plugin == "zero2":
    #     plugin = LowLevelZeroPlugin(
    #         stage=2,
    #         precision=cfg.dtype,
    #         initial_scale=2**16,
    #         max_norm=cfg.grad_clip,
    #     )
    #     set_data_parallel_group(dist.group.WORLD)
    # elif cfg.plugin == "zero2-seq":
    #     plugin = ZeroSeqParallelPlugin(
    #         sp_size=cfg.sp_size,
    #         stage=2,
    #         precision=cfg.dtype,
    #         initial_scale=2**16,
    #         max_norm=cfg.grad_clip,
    #     )
    #     set_sequence_parallel_group(plugin.sp_group)
    #     set_data_parallel_group(plugin.dp_group)
    # else:
    #     raise ValueError(f"Unknown plugin {cfg.plugin}")
    # booster = Booster(plugin=plugin)

    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    dataset = DatasetIMGFromCSV(
        cfg.data_path,
        # TODO: change transforms
        transform=(
            get_transforms_video(cfg.image_size[0])
            if not cfg.use_image_transform
            else get_transforms_image(cfg.image_size[0])
        ),
        num_frames=cfg.num_frames,
        frame_interval=cfg.frame_interval,
        root=cfg.root,
        resolution=cfg.image_size[0]
    )

    # TODO: use plugin's prepare dataloader
    # a batch contains:
    # {
    #      "video": torch.Tensor,  # [B, C, T, H, W],
    #      "text": List[str],
    # }
    # dataloader = prepare_dataloader(
    #     dataset,
    #     batch_size=cfg.batch_size,
    #     num_workers=cfg.num_workers,
    #     shuffle=True,
    #     drop_last=True,
    #     pin_memory=True,
    #     process_group=get_data_parallel_group(),
    # )

    # total_batch_size = cfg.batch_size * dist.get_world_size() // cfg.sp_size

    # ======================================================
    # 4. build model
    # ======================================================
    # 4.1. build model
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    image_encoder = build_module(cfg.image_encoder, MODELS)
    latent_size = vae.get_latent_size(input_size)
    
    model = build_module(
        cfg.model,
        MODELS,
        dtype=dtype,
    )
    model.fix_params()
    model_numel, model_numel_trainable = get_model_numel(model)

    # 4.3. move to device
    vae = vae.to(device, dtype)
    image_encoder = image_encoder.to(torch.float32)
    model = model.to(device, dtype)

    # 4.4. build scheduler
    # scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", subfolder="scheduler")
    # train_scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", subfolder="scheduler")
    scheduler = DDIMScheduler(**cfg.test_scheduler)
    train_scheduler = DDIMScheduler(**cfg.scheduler)
    training_timesteps = train_scheduler.timesteps.to(device)
    # print(scheduler.config)
    # scheduler = pipe.scheduler
    # print(scheduler.config)

    # 4.5. setup optimizer
    # optimizer = HybridAdam(
    #     filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=0, adamw_mode=True
    # )
    # lr_scheduler = None

    model.eval()

    # =======================================================
    # 5. boost model for distributed training with colossalai
    # =======================================================
    torch.set_default_dtype(dtype)
    # model, optimizer, _, dataloader, lr_scheduler = booster.boost(
    #     model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, dataloader=dataloader
    # )
    torch.set_default_dtype(torch.float)
    # num_steps_per_epoch = len(dataloader)

    # =======================================================
    # 6. training loop
    # =======================================================
    start_epoch = start_step = log_step = sampler_start_idx = 0
    running_loss = 0.0

    # 6.1. resume training
    if cfg.load is not None:
        print("Loading checkpoint")
        # booster.load_model(model, os.path.join(cfg.load, "model"))
        load_from_sharded_state_dict(model, cfg.load)

    with torch.no_grad():

        # batch_ref = dataset.getitem(0, start_frame_ind=0, end_frame_ind=24, ref_img_index=0)
        # dataset.num_frames = 1
        # batch_ref = dataset.getitem(0, start_frame_ind=0, end_frame_ind=1, ref_img_index=0)
        import cv2
        # img = cv2.imread("/root/humansora_sdxl/test_final.jpg")[:, :, ::-1]
        if cfg.ref_img is not None:
            img = cv2.imread(cfg.ref_img)[:, :, ::-1]
            img = torch.FloatTensor(img.copy()).permute(2, 0, 1) / 255.0
            img = img.cuda().unsqueeze(0)
            import torchvision.transforms as transforms
            def process_video(video):
                T, C, H, W = video.shape
                if H > W:
                    pad_video = torch.zeros(T, C, H, H)
                    pad_video[:, :, :, (H - W) // 2 : (H - W) // 2 + W] = video
                else:
                    pad_video = torch.zeros(T, C, W, W)
                    pad_video[:, :, (W - H) // 2 : (W - H) // 2 + H, :] = video
                resize = transforms.Resize(dataset.resolution)
                video = resize(pad_video)
                video = dataset.transform(video)  # T C H W
                video = video.permute(1, 0, 2, 3)
                return video
            img = process_video(img).unsqueeze(1).unsqueeze(0)

        T = 96*1
        dataset.num_frames = T
        batch = dataset.getitem(cfg.ref_id, start_frame_ind=0, end_frame_ind=int(T)*1.5, ref_img_index=0)
        # batch = dataset.getitem(0, start_frame_ind=0, end_frame_ind=24, ref_img_index=0)
        # for key in batch_ref:
        #     if torch.is_tensor(batch_ref[key]):
        #         batch_ref[key] = batch_ref[key].unsqueeze(0)
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].unsqueeze(0)

        # ref condition
        if cfg.ref_img is not None:
            ref = img.to(device, dtype)
        else:
            ref = batch_ref["ref_video"].to(device, dtype)


        img_embeddings = image_encoder.encode(ref[:, :, 0, 0] * 0.5 + 0.5)
        img_latent = vae.encode(ref)
        img_latent = img_latent[:, :, 0, 0]

        latents = torch.randn(1, 4, 1, T, img_latent.shape[2], img_latent.shape[3])
        scheduler.set_timesteps(25, device=latents.device)
        timesteps = scheduler.timesteps
        W_T = 24
        for i, t in tqdm(enumerate(timesteps)):
            noise_pred_result = torch.zeros_like(latents)
            noise_pred_weight = torch.zeros_like(latents)
            for w_t_i in tqdm(range(0, T - W_T + 1, W_T // 2)):
                w_t_left = w_t_i
                w_t_right = min(w_t_i + W_T, T)
                latents_slice = latents[:, :, :, w_t_left:w_t_right].clone().to(device, dtype)
                latent_model_input = latents_slice
                cond_video = batch["cond_video"][:, :, :, w_t_left:w_t_right].clone().to(device, dtype)

                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # print(latent_model_input.shape, img_latent.shape, cond_video.shape)
                noise_pred, guidance_feat = model(
                    latent_model_input,
                    t,
                    img_latent,
                    img_embeddings,
                    cond_video,
                    need_reference=True,
                    guidance_feat=None
                )
                noise_pred_temp_weight = torch.zeros_like(noise_pred_weight)
                for t_i in range(w_t_left, w_t_right):
                    distance = min(abs(t_i-w_t_left), abs(t_i-w_t_right+1))
                    distance = (distance + 1) / W_T
                    noise_pred_temp_weight[:, :, :, t_i] += distance
                noise_pred_result[:, :, :, w_t_left:w_t_right] += noise_pred.detach().cpu() * noise_pred_temp_weight[:, :, :, w_t_left:w_t_right]
                noise_pred_weight += noise_pred_temp_weight
            
            # root -> W_G_N x (W_V_N x W_T_N)
            W_V_N = 4
            W_T_N = 6
            W_G_N = 4
            W_G_VT_N = W_V_N * W_T_N
            W_ROOT_N = W_G_VT_N * W_G_N
            N_ROOT = T // W_ROOT_N
            for root_i in range(N_ROOT):
                for w_g_i in tqdm(range(W_G_N)): 
                    latents_slice_list = []
                    cond_video_list = []
                    for w_v_i in range(W_V_N):
                        w_t_left = root_i * W_ROOT_N + w_g_i * W_T_N + w_v_i * W_G_VT_N
                        # print("here")
                        # print(w_t_left)
                        latents_slice = latents[:, :, :, w_t_left:w_t_left+W_T_N].clone().to(device, dtype)
                        # print(latents_slice.shape)
                        # exit(0)
                        latents_slice_list.append(latents_slice)
                        # print(w_t_left, w_t_left+W_T_N)
                        cond_video = batch["cond_video"][:, :, :, w_t_left:w_t_left+W_T_N].clone().to(device, dtype)
                        cond_video_list.append(cond_video)
                    # exit(0)
                    latent_model_input = torch.cat(latents_slice_list, dim=2)
                    cond_video = torch.cat(cond_video_list, dim=2)
            
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                    noise_pred, guidance_feat = model(
                        latent_model_input,
                        t,
                        img_latent,
                        img_embeddings,
                        cond_video,
                        need_reference=True,
                        guidance_feat=None
                    )
                    noise_pred_temp_weight = torch.zeros_like(noise_pred).detach().cpu()
                    
                    noise_pred = noise_pred.detach().cpu()
                    for w_v_i in range(W_V_N):
                        w_t_left = root_i * W_ROOT_N + w_g_i * W_T_N + w_v_i * W_G_VT_N
                        for t_i in range(w_t_left, w_t_left+W_T_N):
                            distance = min(abs(t_i-w_t_left), abs(t_i-w_t_left-W_T_N+1))
                            distance = (distance + 1) / W_T_N
                            noise_pred_temp_weight[:, :, w_v_i:w_v_i+1, t_i-w_t_left] += distance * 0.2
                        noise_pred_result[:, :, :, w_t_left:w_t_left+W_T_N] += noise_pred[:, :, w_v_i:w_v_i+1] * noise_pred_temp_weight[:, :, w_v_i:w_v_i+1]
                        noise_pred_weight[:, :, :, w_t_left:w_t_left+W_T_N] += noise_pred_temp_weight[:, :, w_v_i:w_v_i+1]
            noise_pred = noise_pred_result / noise_pred_weight

            output = scheduler.step(noise_pred, t, latents)
            latents = output.prev_sample
            original = output.pred_original_sample

        samples_list = []
        W_B = 12
        for mini_batch in tqdm(range(0, latents.shape[3], W_B)):
            samples = vae.decode(latents[:, :, :, mini_batch:mini_batch+W_B].to(device, dtype)).detach().cpu()
            samples_list.append(samples)
        samples = torch.cat(samples_list, dim=3)

        global_step = 0
        save_path = os.path.join(exp_dir, f"sample_{global_step}_{cfg.ref_id}")
        save_sample(samples[0], fps=25, save_path=save_path)
        
        save_path = os.path.join(exp_dir, f"sample_train_cond_{global_step}")
        save_sample(cond_video[0, 3:6], fps=25, save_path=save_path)

        # x = img_latent[:1]
        # x = vae.decode(x.unsqueeze(2))
        # save_path = os.path.join(exp_dir, f"sample_train_gt_{global_step}")
        # save_sample(x[0], fps=10, save_path=save_path)



if __name__ == "__main__":
    main()
