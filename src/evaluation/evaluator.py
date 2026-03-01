"""Evaluation pipeline: run model on validation set and report metrics."""

import json
import numpy as np
import torch
from pathlib import Path

from src.data.dataset import HyperspectralValDataset
from src.evaluation.metrics import compute_psnr, compute_sam, aggregate_metrics


def extract_patches(arr: np.ndarray, patch_size: int, n_patches: int = 6, seed: int = 42) -> list[np.ndarray]:
    """Extract n random patches from an array (H, W, C)."""
    rng = np.random.RandomState(seed)
    h, w, c = arr.shape
    patches = []
    attempts = 0
    while len(patches) < n_patches and attempts < 1000:
        y = rng.randint(0, max(1, h - patch_size))
        x = rng.randint(0, max(1, w - patch_size))
        p = arr[y:y + patch_size, x:x + patch_size, :]
        if p.shape[0] == patch_size and p.shape[1] == patch_size:
            patches.append(p)
        attempts += 1
    return patches


def run_model_on_patches(
    model: torch.nn.Module,
    lr_patches: list[np.ndarray],
    device: torch.device,
    scale: int = 4,
) -> list[np.ndarray]:
    """Run model on a list of LR patches, return SR patches."""
    model.eval()
    sr_patches = []
    with torch.no_grad():
        for patch in lr_patches:
            t = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).to(device)
            sr_t = model(t)
            sr = sr_t.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
            sr = np.clip(sr, 0, 1)
            sr_patches.append(sr)
    return sr_patches


def evaluate_model(model: torch.nn.Module, val_dir: str, device: torch.device, scale: int = 4) -> dict:
    """Evaluate model on validation set using PSNR and SAM.

    Uses same protocol as competition: 6 patches per scene at 200x200.
    """
    val_dataset = HyperspectralValDataset(val_dir)
    patch_size_hr = 200
    patch_size_lr = patch_size_hr // scale

    all_metrics = []

    for i in range(len(val_dataset)):
        lr_norm, hr_norm, vmin, vmax, path = val_dataset[i]
        scene_name = Path(path).stem
        print(f"  Evaluating scene {i + 1}/{len(val_dataset)}: {scene_name}")

        lr_patches = extract_patches(lr_norm, patch_size_lr, n_patches=6, seed=42 + i)
        hr_patches = extract_patches(hr_norm, patch_size_hr, n_patches=6, seed=42 + i)

        if not lr_patches:
            print(f"    Skipping: not enough data")
            continue

        sr_patches = run_model_on_patches(model, lr_patches, device, scale)

        scene_psnr, scene_sam = [], []
        for sr, hr in zip(sr_patches, hr_patches):
            psnr = compute_psnr(sr, hr, data_range=1.0)
            sam = compute_sam(sr, hr)
            scene_psnr.append(psnr)
            scene_sam.append(sam)
            all_metrics.append({"psnr": psnr, "sam": sam})

        print(f"    PSNR={np.mean(scene_psnr):.2f} dB | SAM={np.mean(scene_sam):.4f}°")

    if not all_metrics:
        return {"error": "No metrics computed"}

    agg = aggregate_metrics(all_metrics)
    print(f"\nOverall: PSNR={agg['psnr_mean']:.2f}±{agg['psnr_std']:.2f} dB | SAM={agg['sam_mean']:.4f}±{agg['sam_std']:.4f}°")
    return agg


def load_model_from_checkpoint(checkpoint_path: str, cfg: dict, device: torch.device) -> torch.nn.Module:
    """Load model weights from checkpoint."""
    from src.models.hsr_net import build_model
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model
