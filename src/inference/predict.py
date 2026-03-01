"""Generate SR predictions for the test set using a trained model."""

import itertools
from pathlib import Path

import h5py
import numpy as np
import torch

from src.utils.io import save_h5_prediction


def _tile_predict(
    model: torch.nn.Module,
    lr_norm: np.ndarray,
    device: torch.device,
    tile_size: int = 200,
    overlap: int = 32,
    scale: int = 4,
    use_tta: bool = True,
) -> np.ndarray:
    """Tile-based inference with optional test-time augmentation (TTA).

    Args:
        lr_norm: (H, W, C) normalized [0,1] LR array.
        tile_size: LR tile size.
        overlap: LR overlap between tiles.
        scale: Upscaling factor.
        use_tta: If True, apply 8-fold TTA (flips + rotations).
    Returns:
        sr_norm: (H*scale, W*scale, C) predicted SR array.
    """
    lr_h, lr_w, c = lr_norm.shape
    hr_h, hr_w = lr_h * scale, lr_w * scale
    output = np.zeros((hr_h, hr_w, c), dtype=np.float32)
    weight = np.zeros((hr_h, hr_w, 1), dtype=np.float32)

    step = tile_size - overlap
    y_starts = list(range(0, max(1, lr_h - tile_size + 1), step))
    x_starts = list(range(0, max(1, lr_w - tile_size + 1), step))
    if not y_starts or y_starts[-1] + tile_size < lr_h:
        y_starts.append(max(0, lr_h - tile_size))
    if not x_starts or x_starts[-1] + tile_size < lr_w:
        x_starts.append(max(0, lr_w - tile_size))
    y_starts = sorted(set(y_starts))
    x_starts = sorted(set(x_starts))

    model.eval()
    with torch.no_grad():
        for y, x in itertools.product(y_starts, x_starts):
            y_end = min(y + tile_size, lr_h)
            x_end = min(x + tile_size, lr_w)
            patch_lr = lr_norm[y:y_end, x:x_end, :]
            p_h, p_w = patch_lr.shape[:2]

            if use_tta:
                sr_patch = _tta_predict(model, patch_lr, device, scale)
            else:
                t = torch.from_numpy(patch_lr.transpose(2, 0, 1)).unsqueeze(0).to(device)
                sr_t = model(t)
                sr_patch = sr_t.squeeze(0).permute(1, 2, 0).cpu().float().numpy()

            sr_patch = np.clip(sr_patch, 0.0, 1.0)
            hr_y = y * scale
            hr_x = x * scale
            hr_h_p = p_h * scale
            hr_w_p = p_w * scale
            output[hr_y:hr_y + hr_h_p, hr_x:hr_x + hr_w_p] += sr_patch[:hr_h_p, :hr_w_p]
            weight[hr_y:hr_y + hr_h_p, hr_x:hr_x + hr_w_p] += 1.0

    weight = np.maximum(weight, 1e-8)
    return output / weight


def _tta_predict(
    model: torch.nn.Module, patch: np.ndarray, device: torch.device, scale: int
) -> np.ndarray:
    """8-fold test-time augmentation: average over flips and 90° rotations."""
    preds = []
    for flip_h in [False, True]:
        for flip_v in [False, True]:
            aug = patch.copy()
            if flip_h:
                aug = aug[:, ::-1, :]
            if flip_v:
                aug = aug[::-1, :, :]
            t = torch.from_numpy(aug.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
            sr_t = model(t)
            sr = sr_t.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
            if flip_h:
                sr = sr[:, ::-1, :]
            if flip_v:
                sr = sr[::-1, :, :]
            preds.append(sr.copy())
    return np.mean(preds, axis=0)


def predict_scene(
    model: torch.nn.Module,
    lr_path: Path,
    output_path: Path,
    device: torch.device,
    scale: int = 4,
    tile_size: int = 200,
    overlap: int = 32,
    use_tta: bool = True,
) -> np.ndarray:
    """Predict SR for a single scene and save to HDF5.

    Returns: sr_denorm (H*scale, W*scale, C) in original data range.
    """
    lr_path = Path(lr_path)
    with h5py.File(lr_path, "r") as f:
        keys = list(f.keys())
        lr = None
        for k in keys:
            if "lr" in k.lower():
                lr = f[k][:].astype(np.float32)
                break
        if lr is None:
            lr = f[keys[0]][:].astype(np.float32)

    vmin = float(lr.min())
    vmax = float(lr.max())
    if vmax - vmin > 1e-8:
        lr_norm = (lr - vmin) / (vmax - vmin)
    else:
        lr_norm = lr.copy()

    print(f"  LR shape: {lr.shape}, range [{vmin:.4f}, {vmax:.4f}]")
    sr_norm = _tile_predict(model, lr_norm, device, tile_size, overlap, scale, use_tta)

    # Denormalize back to original range
    sr_denorm = sr_norm * (vmax - vmin) + vmin

    save_h5_prediction(output_path, sr_denorm, key="sr")
    print(f"  Saved SR to {output_path} (shape: {sr_denorm.shape})")
    return sr_denorm


def predict_all_test_scenes(
    model: torch.nn.Module,
    test_dir: str | Path,
    output_dir: str | Path,
    device: torch.device,
    scale: int = 4,
    tile_size: int = 200,
    overlap: int = 32,
    use_tta: bool = True,
) -> list[Path]:
    """Predict SR for all scenes in test_dir."""
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_files = sorted(test_dir.glob("*.h5"))
    if not test_files:
        print(f"No .h5 files found in {test_dir}")
        return []

    print(f"\nGenerating predictions for {len(test_files)} test scenes...")
    output_paths = []
    for i, f in enumerate(test_files):
        # Output filename: HR_00.h5, HR_01.h5, ...
        out_name = f"HR_{i:02d}.h5"
        out_path = output_dir / out_name
        print(f"\nScene {i + 1}/{len(test_files)}: {f.name} -> {out_name}")
        predict_scene(model, f, out_path, device, scale, tile_size, overlap, use_tta)
        output_paths.append(out_path)

    print(f"\nAll predictions saved to {output_dir}")
    return output_paths
