degradation_yaml = 'realbasicvsr.yaml'

model_path = "./pretrained_weight/heavy_deg.pt"



num_workers = 1

# Define acceleration
dtype = "fp16"
grad_checkpoint = False
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

prediction_type = 'v_prediction'
use_df_loss = False

# Others
seed = 42
outputs = "experiments"
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 100
load = None

batch_size = 1
# batch_size = 1
lr = 5e-7
grad_clip = 1.0
