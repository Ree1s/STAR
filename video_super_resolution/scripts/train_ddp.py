import os
import sys
import csv
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from copy import deepcopy
from tqdm import tqdm
import wandb

# Append current directory to sys.path to locate modules.
sys.path.append('.')
from openvid.datasets import DatasetFromCSV, get_transforms_image, get_transforms_video, prepare_dataloader
from openvid.registry import MODELS, SCHEDULERS, build_module
from openvid.utils.ckpt_utils import create_logger, load, model_sharding, record_model_param_shape, save
from openvid.utils.config_utils import create_experiment_workspace, create_tensorboard_writer, parse_configs, save_training_config
from openvid.utils.misc import all_reduce_mean, format_numel_str, get_model_numel, requires_grad, to_torch_dtype
from openvid.utils.train_utils import update_ema
import torch.optim as optim
from torch.optim import AdamW
# torch.autograd.set_detect_anomaly(True)
from video_to_video.modules import ControlledV2VUNet, FrozenOpenCLIPEmbedder
from openvidsr import RealVSRCSVVideoDataset
from video_to_video.diffusion.diffusion_sdedit import GaussianDiffusion
from video_to_video.diffusion.schedules_sdedit import noise_schedule
from video_to_video.video_to_video_model_train import VideoToVideo_sr
from diffusers import AutoencoderKLTemporalDecoder
from einops import rearrange

def check_gradients(model: torch.nn.Module):
    """
    Iterate over all model parameters and print gradient statistics.
    """
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_mean = param.grad.mean().item()
        grad_std = param.grad.std().item()
        grad_max = param.grad.abs().max().item()
        grad_min = param.grad.abs().min().item()
        print(f"{name}: grad_mean = {grad_mean:.6f}, grad_std = {grad_std:.6f}, "
              f"grad_max = {grad_max:.6f}, grad_min = {grad_min:.6f}")

def main():
    # ======================================================
    # 1. Load configs and create experiment workspace
    # ======================================================
    cfg = parse_configs(training=True)
    print(cfg)
    exp_name, exp_dir = create_experiment_workspace(cfg)
    
    # ======================================================
    # 2. Initialize torch.distributed (DDP)
    # ======================================================
    # Use NCCL backend for GPUs.
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    dtype = to_torch_dtype(cfg.dtype)
    
    # ======================================================
    # 3. Setup logger, tensorboard, and wandb (only on rank 0)
    # ======================================================
    if dist.get_rank() == 0:
        save_training_config(cfg._cfg_dict, exp_dir)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory created at {exp_dir}")
        writer = create_tensorboard_writer(exp_dir)
        if cfg.wandb:
            wandb.init(project="video_super_resolution", name=exp_name, config=cfg._cfg_dict)
    else:
        logger = create_logger(None)
    
    # ======================================================
    # 4. Build dataset and dataloader with DistributedSampler
    # ======================================================
    dataset = RealVSRCSVVideoDataset(cfg.degradation_yaml)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} videos")
    total_batch_size = cfg.batch_size * dist.get_world_size()
    logger.info(f"Total batch size: {total_batch_size}")
    
    # ======================================================
    # 5. Build model, wrap with DDP, and set up optimizer/EMA
    # ======================================================
    model = VideoToVideo_sr(cfg, device=device).to(device)
    # Wrap model with torch.nn.parallel.DistributedDataParallel
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    # Record model parameters and create EMA model.
    model_numel, model_numel_trainable, trainable_list, untrainable_list = get_model_numel(model.module)
    logger.info(f"Trainable model params: {format_numel_str(model_numel_trainable)}, Total model params: {format_numel_str(model_numel)}")
    logger.info(f"Trainable list: {trainable_list}")
    logger.info(f"Untrainable list: {untrainable_list}")
    
    ema = deepcopy(model.module).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=0)
    lr_scheduler = None  # Adjust or build your scheduler if needed
    scaler = torch.cuda.amp.GradScaler()
    # ======================================================
    # 6. Prepare model for training
    # ======================================================
    # if cfg.grad_checkpoint:
    #     # If you have a gradient checkpointing function, call it here.
    #     from openvid.acceleration.checkpoint import set_grad_checkpoint
    #     set_grad_checkpoint(model.module.generator)
    model.train()
    update_ema(ema, model.module, decay=0, sharded=False)
    ema.eval()
    
    num_steps_per_epoch = len(dataloader)
    logger.info(f"Training for {cfg.epochs} epochs with {num_steps_per_epoch} steps per epoch")
    
    start_epoch = 0
    start_step = 0
    sampler_start_idx = 0
    running_loss = 0.0
    log_step = 0
    
    # Optionally resume training if checkpoint is provided.
    if cfg.load is not None:
        logger.info("Loading checkpoint")
        start_epoch, start_step, sampler_start_idx = load(model.module, ema, optimizer, lr_scheduler, cfg.load)
        logger.info(f"Loaded checkpoint {cfg.load} at epoch {start_epoch} step {start_step}")
    
    # Set the sampler start index and epoch
    if hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(start_epoch)
    # model_sharding(ema)
    
    # ======================================================
    # 7. Training Loop
    # ======================================================
    for epoch in range(start_epoch, cfg.epochs):
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info(f"Beginning epoch {epoch}...")
        
        with tqdm(range(start_step, num_steps_per_epoch),
                  desc=f"Epoch {epoch}",
                  disable=(dist.get_rank() != 0),
                  total=num_steps_per_epoch,
                  initial=start_step) as pbar:
            for step in pbar:
                batch = next(dataloader_iter)
                x = batch["lqs"].to(device, dtype)
                y = batch["gts"].to(device, dtype)
                text = batch["text"]
                # Optionally print video path on master process.
                # if dist.get_rank() == 0:
                #     print(batch['video_path'])
                with torch.cuda.amp.autocast(enabled=True):
                
                    loss = model.module.train_losses(x, y, text)
                print(loss.item())
                optimizer.zero_grad()
                # with torch.autograd.detect_anomaly():

                scaler.scale(loss).backward()
                grad_clip = 1.0
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                # if dist.get_rank() == 0:

                    # check_gradients(model)
                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()
                update_ema(ema, model.module, optimizer=optimizer, sharded=False)
                
                # Optionally log loss (all_reduce_mean if necessary)
                running_loss += loss.item()
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1
                
                if dist.get_rank() == 0 and (global_step + 1) % cfg.log_every == 0:
                    avg_loss = running_loss / log_step
                    pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                    running_loss = 0
                    log_step = 0
                    writer.add_scalar("loss", loss.item(), global_step)
                    if cfg.wandb:
                        wandb.log({
                            "iter": global_step,
                            "num_samples": global_step * total_batch_size,
                            "epoch": epoch,
                            "loss": loss.item(),
                            "avg_loss": avg_loss,
                        }, step=global_step)
                
                # Save checkpoint periodically on rank 0.
                if cfg.ckpt_every > 0 and (global_step + 1) % cfg.ckpt_every == 0 and dist.get_rank() == 0:
                    save(model.module, ema, optimizer, lr_scheduler, epoch, step + 1, global_step + 1, cfg.batch_size, exp_dir, ema_shape_dict)
                    # dist.barrier()
                    logger.info(f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {exp_dir}")
        # Reset for next epoch.
        start_step = 0

if __name__ == "__main__":
    main()
