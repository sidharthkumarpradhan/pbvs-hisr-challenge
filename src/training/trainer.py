"""Training loop for hyperspectral SR model."""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from src.data.dataset import HyperspectralPatchDataset, HyperspectralValDataset
from src.evaluation.metrics import compute_psnr, compute_sam
from src.training.losses import build_loss


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg["training"]
    name = train_cfg.get("optimizer", "adam").lower()
    lr = float(train_cfg.get("lr", 2e-4))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return Adam(model.parameters(), lr=lr, weight_decay=wd)


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    train_cfg = cfg["training"]
    name = train_cfg.get("scheduler", "cosine").lower()
    epochs = int(train_cfg.get("epochs", 300))
    lr_min = float(train_cfg.get("lr_min", 1e-6))
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
    elif name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    return None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("Using Apple MPS")
    else:
        dev = torch.device("cpu")
        print("Using CPU")
    return dev


def validate(model: nn.Module, val_dataset: HyperspectralValDataset, device: torch.device, scale: int) -> dict:
    """Run validation on a few patches per scene, return mean PSNR and SAM."""
    model.eval()
    all_psnr, all_sam = [], []
    patch_size_lr = 200

    with torch.no_grad():
        for i in range(len(val_dataset)):
            lr, hr, vmin, vmax, path = val_dataset[i]
            if hr is None:
                continue
            lr_h, lr_w, c = lr.shape

            patches_done = 0
            for attempt in range(20):
                y = np.random.randint(0, max(1, lr_h - patch_size_lr))
                x = np.random.randint(0, max(1, lr_w - patch_size_lr))
                lr_patch = lr[y:y + patch_size_lr, x:x + patch_size_lr, :]
                hr_patch = hr[y * scale:y * scale + patch_size_lr * scale, x * scale:x * scale + patch_size_lr * scale, :]
                if lr_patch.shape[0] < 10 or hr_patch.shape[0] < 10:
                    continue

                lr_t = torch.from_numpy(lr_patch.transpose(2, 0, 1)).unsqueeze(0).to(device)
                sr_t = model(lr_t)
                sr = sr_t.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
                sr = np.clip(sr, 0, 1)
                hr_norm = hr_patch

                psnr = compute_psnr(sr, hr_norm, data_range=1.0)
                sam = compute_sam(sr, hr_norm)
                all_psnr.append(psnr)
                all_sam.append(sam)
                patches_done += 1
                if patches_done >= 2:
                    break

    return {
        "psnr": float(np.mean(all_psnr)) if all_psnr else 0.0,
        "sam": float(np.mean(all_sam)) if all_sam else 999.0,
    }


def train(cfg: dict, checkpoint_dir: str = "checkpoints") -> Path:
    """Full training loop. Returns path to best checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "training_log.jsonl"

    device = get_device()

    # Build model
    from src.models.hsr_net import build_model
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg['model']['name']} | Parameters: {total_params:,}")

    # Build loss
    criterion = build_loss(cfg)

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, cfg)

    # Build datasets
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    augment_cfg = train_cfg.get("augmentation", {})

    train_dataset = HyperspectralPatchDataset(
        scene_dir=data_cfg["train_dir"],
        patch_size=train_cfg.get("patch_size", 64),
        scale=data_cfg.get("scale", 4),
        num_patches_per_scene=train_cfg.get("num_patches_per_scene", 200),
        augment=True,
        augment_cfg=augment_cfg,
    )

    num_workers = train_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = HyperspectralValDataset(scene_dir=data_cfg["val_dir"])

    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    # Resume from checkpoint if requested
    start_epoch = 1
    best_psnr = 0.0
    best_ckpt = checkpoint_dir / "best_model.pth"
    last_ckpt = checkpoint_dir / "last_model.pth"

    if train_cfg.get("resume", False) and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state.get("epoch", 0) + 1
        best_psnr = state.get("best_psnr", 0.0)
        print(f"Resumed from epoch {start_epoch - 1}, best PSNR={best_psnr:.2f}")

    epochs = int(train_cfg.get("epochs", 300))
    val_every = int(train_cfg.get("val_every", 5))
    save_every = int(train_cfg.get("save_every", 10))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    scale = int(data_cfg.get("scale", 4))

    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_losses = {"total": 0.0, "l1": 0.0, "sam": 0.0}
        t0 = time.time()

        for batch_idx, (lr, hr) in enumerate(train_loader):
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            optimizer.zero_grad()
            sr = model(lr)
            loss, loss_dict = criterion(sr, hr)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += loss_dict.get(k, 0.0)

        if scheduler is not None:
            scheduler.step()

        n_batches = len(train_loader)
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        log_entry = {
            "epoch": epoch,
            "loss": avg_losses["total"],
            "l1": avg_losses.get("l1", 0),
            "sam_loss": avg_losses.get("sam", 0),
            "lr": lr_now,
            "time_s": elapsed,
        }

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Loss={avg_losses['total']:.4f} "
            f"L1={avg_losses['l1']:.4f} "
            f"SAM={avg_losses['sam']:.4f} "
            f"LR={lr_now:.2e} "
            f"({elapsed:.1f}s)"
        )

        # Validation
        if epoch % val_every == 0 or epoch == epochs:
            val_metrics = validate(model, val_dataset, device, scale)
            log_entry.update({"val_psnr": val_metrics["psnr"], "val_sam": val_metrics["sam"]})
            print(f"  Val: PSNR={val_metrics['psnr']:.2f} dB | SAM={val_metrics['sam']:.4f}°")

            if val_metrics["psnr"] > best_psnr:
                best_psnr = val_metrics["psnr"]
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "psnr": best_psnr, "cfg": cfg},
                    best_ckpt,
                )
                print(f"  >> New best model saved (PSNR={best_psnr:.2f})")

        # Save periodic checkpoint
        if epoch % save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_psnr": best_psnr,
                    "cfg": cfg,
                },
                last_ckpt,
            )

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    # Final save
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epochs,
            "best_psnr": best_psnr,
            "cfg": cfg,
        },
        last_ckpt,
    )
    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f} dB")
    return best_ckpt if best_ckpt.exists() else last_ckpt
