import torch
import colossalai
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from openvid.utils.config_utils import parse_configs
from openvid.utils.train_utils import update_ema
from openvid.utils.ckpt_utils import save
from openvid.utils.misc import requires_grad, to_torch_dtype
from openvid.registry import MODELS, build_module
from openvid.datasets import DatasetFromCSV, prepare_dataloader
from openvid.utils.config_utils import create_experiment_workspace, create_tensorboard_writer
import wandb
from tqdm import tqdm
import sys
sys.path.append('.')
from video_to_video_model import VideoToVideo_sr

def main():
    # ======================================================
    # 1. args & cfg
    # ======================================================
    cfg = parse_configs(training=True)
    print(cfg)
    exp_name, exp_dir = create_experiment_workspace(cfg)

    # ======================================================
    # 2. runtime variables & colossalai launch
    # ======================================================
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert cfg.dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg.dtype}"

    # 2.1. colossalai init distributed training
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()
    device = torch.device(f'cuda:{coordinator.local_rank}')
    dtype = to_torch_dtype(cfg.dtype)

    # 2.2. init logger, tensorboard & wandb
    if not coordinator.is_master():
        logger = None
    else:
        logger = create_logger(exp_dir)
        writer = create_tensorboard_writer(exp_dir)
        if cfg.wandb:
            wandb.init(project="video_super_resolution", name=exp_name, config=cfg._cfg_dict)

    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    dataset = DatasetFromCSV(
        cfg.data_path,
        transform=get_transforms_video(cfg.image_size[0]),
        num_frames=cfg.num_frames,
        frame_interval=cfg.frame_interval,
        root=cfg.root,
    )

    dataloader = prepare_dataloader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} videos ({cfg.data_path})")

    # ======================================================
    # 4. build model
    # ======================================================
    model = VideoToVideo_sr(cfg)
    model = model.to(device, dtype)
    model.train()

    # ======================================================
    # 5. optimizer & scheduler
    # ======================================================
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=0, adamw_mode=True
    )
    lr_scheduler = None

    # ======================================================
    # 6. training loop
    # ======================================================
    for epoch in range(cfg.epochs):
        dataloader_iter = iter(dataloader)
        logger.info(f"Beginning epoch {epoch}...")

        with tqdm(
            range(len(dataloader)),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            total=len(dataloader),
        ) as pbar:
            for step in pbar:
                batch = next(dataloader_iter)
                x = batch["video"].to(device, dtype)
                y = batch["text"]

                # Forward pass
                loss = model(x, y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Log loss values
                if coordinator.is_master() and (step + 1) % cfg.log_every == 0:
                    writer.add_scalar("loss", loss.item(), epoch * len(dataloader) + step)
                    if cfg.wandb:
                        wandb.log({"loss": loss.item()})

                # Save checkpoint
                if cfg.ckpt_every > 0 and (step + 1) % cfg.ckpt_every == 0:
                    save(model, optimizer, epoch, step + 1, exp_dir)
                    logger.info(f"Saved checkpoint at epoch {epoch} step {step + 1}")

if __name__ == "__main__":
    main()
