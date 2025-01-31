from huggingface_hub import hf_hub_download

model_path = '/group/ossdphi_algo_scratch_14/sichegao/checkpoints/STAR'   # The local directory to save downloaded checkpoint
hf_hub_download(repo_id="SherryX/STAR/tree/main/I2VGen-XL-based", filename="light_deg.pt", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')