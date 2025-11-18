num_frames = 24
frame_interval = 1
multiview = 1
image_size = (768, 768)

# Define dataset
root = None
data_path = "CSV_PATH"
use_image_transform = False
num_workers = 0
ref_img = None
ref_id = 0

# Define acceleration
dtype = "bf16"
grad_checkpoint = False
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="HumanSORA",
    base_model_path="./configs/human4dit.json",
    guidance_types=["dwpose", "normal"],
    guidance_encoder_kwargs=dict(
        guidance_embedding_channels=320,
        guidance_input_channels=3,
        block_out_channels=[16, 32, 96, 256]
    ),
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="./checkpoints/sdxl-vae",
    sub_folder=None,
)
image_encoder = dict(
    type="clip-image",
    from_pretrained="./checkpoints/stable-video-diffusion-img2vid-xt"
)
scheduler = dict(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="linear",
    steps_offset=1,
    clip_sample=False,
    rescale_betas_zero_snr=True,
    timestep_spacing="trailing",
    prediction_type="v_prediction",
)
test_scheduler = dict(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="linear",
    steps_offset=1,
    clip_sample=False,
    rescale_betas_zero_snr=True,
    timestep_spacing="trailing",
    prediction_type="v_prediction",
)
max_train_timesteps = None

# Others
seed = 42
outputs = "outputs"
wandb = False

epochs = 1000
log_every = 2
ckpt_every = 500
sample_every = 500
load = None

batch_size = 1
lr = 1e-5
grad_clip = 1.0
