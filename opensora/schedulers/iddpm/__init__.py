from functools import partial

import torch

from opensora.registry import SCHEDULERS

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


@SCHEDULERS.register_module("iddpm")
class IDDPM(SpacedDiffusion):
    def __init__(
        self,
        num_sampling_steps=None,
        timestep_respacing=None,
        noise_schedule="linear",
        use_kl=False,
        sigma_small=False,
        model_mean_type="x_start", # x_start, eps, v_prediction
        learn_sigma=True,
        rescale_learned_sigmas=False,
        diffusion_steps=1000,
        cfg_scale=4.0,
    ):
        betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
        if num_sampling_steps is not None:
            assert timestep_respacing is None
            timestep_respacing = str(num_sampling_steps)
        if timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [diffusion_steps]
        if model_mean_type == "x_start":
            mean_type =  gd.ModelMeanType.START_X
        elif model_mean_type == "v_prediction":
            mean_type = gd.ModelMeanType.V_PREDICTION
        elif model_mean_type == "eps":
            mean_type = gd.ModelMeanType.EPSILON
        else:
            raise ValueError(f"Unknown model_mean_type: {model_mean_type}")
        super().__init__(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=mean_type,
            model_var_type=(
                (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            # rescale_timesteps=rescale_timesteps,
        )

        self.cfg_scale = cfg_scale

    def sample(
        self,
        model,
        text_encoder,
        z_size,
        prompts,
        device,
        additional_args=None,
        vae=None,
    ):
        n = len(prompts)
        z = torch.randn(n, *z_size, device=device)
        # z = torch.cat([z, z], 0)
        model_args = text_encoder.encode(prompts)
        # y_null = text_encoder.null(n)
        # model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        forward = partial(forward_with_cfg, model, cfg_scale=self.cfg_scale)
        samples = self.ddim_sample_loop(
            forward,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_args,
            progress=True,
            device=device,
            vae=vae
        )
        # samples, _ = samples.chunk(2, dim=0)
        return samples

    def sample_svd(
        self,
        model,
        model_args,
        z_size,
        prompts,
        device,
        additional_args=None,
        vae=None,
    ):
        n = len(prompts)
        z = torch.randn(n, *z_size, device=device)
        
        if additional_args is not None:
            model_args.update(additional_args)

        forward = partial(forward_with_cfg, model, cfg_scale=self.cfg_scale)
        samples = self.ddim_sample_loop(
            forward,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_args,
            progress=True,
            device=device,
            vae=vae
        )
        # samples, _ = samples.chunk(2, dim=0)
        return samples
    
    def sample_image(
        self,
        model,
        text_encoder,
        z,
        z_size,
        prompts,
        device,
        additional_args=None,
        vae=None,
        denoised_weight=1.0
    ):
        n = len(prompts)
        model_args = text_encoder.encode(prompts)
        # y_null = text_encoder.null(n)
        # model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        forward = partial(forward_with_cfg, model, cfg_scale=self.cfg_scale)
        samples = self.ddim_sample_loop_image(
            forward,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_args,
            progress=True,
            device=device,
            vae=vae,
            denoised_weight=denoised_weight
        )
        # samples, _ = samples.chunk(2, dim=0)
        return samples


def forward_with_cfg(model, x, timestep, y, cfg_scale, **kwargs):
    # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    model_out = model.forward(x, timestep, y, **kwargs)
    model_out = model_out["x"] if isinstance(model_out, dict) else model_out
    return model_out

    half = x[: len(x) // 2]
    combined = torch.cat([half, half], dim=0)
    model_out = model.forward(combined, timestep, y, **kwargs)
    model_out = model_out["x"] if isinstance(model_out, dict) else model_out
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)
