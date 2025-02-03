# Define video-to-video super-resolution configuration

# Define dataset settings
dataset = dict(
    root="dataset/OpenVid-1M/video",  # Path to video data
    data_path="dataset/OpenVid-1M/data/train/OpenVid-1M.csv",  # CSV containing metadata
    num_frames=16,  # Number of frames per video
    frame_interval=3,  # Interval between consecutive frames
    image_size=(512, 512),  # Image resolution (height, width)
    use_image_transform=False,  # If True, apply image transformations
    num_workers=4,  # Number of workers for data loading
)

# Define model settings
model = dict(
    type="VideoToVideo_sr",  # Model type (Video-to-Video Super Resolution)
    space_scale=1.0,  # Scaling factor for space (spatial resolution)
    time_scale=1.0,  # Scaling factor for time (temporal resolution)
    from_pretrained="path_to_pretrained_model",  # Pretrained model path
    enable_flashattn=True,  # Whether to enable flash attention
    enable_layernorm_kernel=True,  # Enable layer normalization kernel optimization
)

# Define VAE settings
vae = dict(
    type="AutoencoderKLTemporalDecoder",  # VAE type
    from_pretrained="stabilityai/stable-video-diffusion-img2vid",  # Pretrained VAE path
    cache_dir="/group/ossdphi_algo_scratch_14/sichegao/checkpoints",  # Cache directory for VAE model
    subfolder="vae",  # Subfolder containing the VAE weights
    variant="fp16",  # Whether to use half precision (FP16)
)

# Define text encoder settings
text_encoder = dict(
    type="FrozenOpenCLIPEmbedder",  # Text encoder type (Frozen CLIP)
    from_pretrained="laion2b_s32b_b79k",  # Pretrained model for text embedding
    model_max_length=120,  # Maximum length for input text
    shardformer=True,  # Whether to use Shardformer for efficient text encoding
)

# Define noise scheduler settings
scheduler = dict(
    type="GaussianDiffusion",  # Scheduler type (Gaussian Diffusion)
    noise_schedule="logsnr_cosine_interp",  # Type of noise schedule
    total_noise_levels=1000,  # Total noise levels (for diffusion process)
    steps=50,  # Number of sampling steps
    guide_scale=7.5,  # Guidance scale for controlling strength of generation
    solver="dpmpp_2m_sde",  # Solver type (DPM++ 2M SDE)
    solver_mode="fast",  # Solver mode (e.g., 'fast' or 'accurate')
)

# Training and inference settings
training = dict(
    epochs=1000,  # Number of epochs
    batch_size=4,  # Batch size during training
    lr=2e-5,  # Learning rate
    grad_clip=1.0,  # Gradient clipping
    grad_checkpoint=True,  # Whether to use gradient checkpointing
    plugin="zero2",  # Acceleration plugin (e.g., ZeRO-2 for memory optimization)
    sp_size=1,  # Size of each pipeline stage (for model parallelism)
)

# Other settings
seed = 42  # Random seed for reproducibility
outputs = "output/"  # Directory for saving outputs
wandb = False  # Whether to use Weights & Biases for logging
log_every = 10  # How often to log metrics (in terms of steps)
ckpt_every = 2500  # How often to save model checkpoints (in terms of steps)
load = None  # Path to a pre-trained model checkpoint (if loading one)
