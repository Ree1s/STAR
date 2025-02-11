num_frames = 16
# num_frames = 1
frame_interval = 3
# image_size = (16, 16)
image_size = (256, 256)

# Define dataset
root = "dataset/OpenVid-1M/video"
data_path = "dataset/OpenVid-1M/data/train/OpenVid-1M_subset.csv"

use_image_transform = False
num_workers = 4

# Define acceleration
dtype = "fp16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="MVDiT-XL/2",
    space_scale=0.5,
    time_scale=1.0,
    from_pretrained=None,
    enable_flashattn=False,
    enable_layernorm_kernel=False,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=False,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "experiments"
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 2500
load = None

batch_size = 8
# batch_size = 1
lr = 2e-5
# lr = 1.25e-6
grad_clip = 1.0
