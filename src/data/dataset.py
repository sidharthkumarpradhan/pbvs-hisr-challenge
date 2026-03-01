"""PyTorch datasets for hyperspectral SR training and inference.

Memory-efficient design:
- HyperspectralPatchDataset: LAZY loading - reads patches from H5 files on demand.
  Only scene metadata (shapes, vmin/vmax) is kept in RAM. The actual pixel data is
  read from disk for each patch, so RAM usage is O(num_workers * patch_size) not
  O(num_scenes * scene_size).
- HyperspectralValDataset: Reads one scene at a time during validation.
"""
import random
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _discover_lr_hr_keys(f: h5py.File) -> Tuple[Optional[str], Optional[str]]:
    """Return (lr_key, hr_key) from an open H5 file, or (None, None) if ambiguous."""
    keys = list(f.keys())
    lr_key = hr_key = None
    for k in keys:
        kl = k.lower()
        if "lr" in kl:
            lr_key = k
        elif "hr" in kl:
            hr_key = k
    # Fallback: if no named keys, use positional order
    if lr_key is None and hr_key is None:
        if len(keys) == 2:
            lr_key, hr_key = keys[0], keys[1]
        elif len(keys) == 1:
            lr_key = keys[0]
    return lr_key, hr_key


def _get_scene_meta(path: Path) -> Optional[dict]:
    """Read only shapes and normalisation stats from an H5 file (no pixel data)."""
    try:
        with h5py.File(path, "r") as f:
            lr_key, hr_key = _discover_lr_hr_keys(f)
            if lr_key is None:
                return None
            lr_shape = f[lr_key].shape  # (H, W, C) or (C, H, W)
            hr_shape = f[hr_key].shape if hr_key else None

            # Read a small sample to compute vmin/vmax without loading full array
            # Use the HR as reference (or LR if no HR)
            ref_key = hr_key if hr_key else lr_key
            ref_ds = f[ref_key]
            # Sample up to 50k values for statistics
            flat_size = int(np.prod(ref_ds.shape))
            if flat_size <= 50_000:
                sample = ref_ds[:].astype(np.float32).ravel()
            else:
                # Read first slice only for speed
                sample = ref_ds[0].astype(np.float32).ravel()

            vmin = float(sample.min())
            vmax = float(sample.max())

        return {
            "path": path,
            "lr_key": lr_key,
            "hr_key": hr_key,
            "lr_shape": lr_shape,
            "hr_shape": hr_shape,
            "vmin": vmin,
            "vmax": vmax,
        }
    except Exception as e:
        print(f"  WARNING: Could not read meta from {path.name}: {e}")
        return None


class HyperspectralPatchDataset(Dataset):
    """Training dataset: random patches read LAZILY from H5 files.

    Only scene metadata (shapes, vmin, vmax) is held in RAM.
    Pixel data is read from disk for each __getitem__ call, keeping
    RAM proportional to DataLoader workers, not to number of scenes.
    """

    def __init__(
        self,
        scene_dir: str,
        patch_size: int = 64,
        scale: int = 4,
        num_patches_per_scene: int = 200,
        augment: bool = True,
        augment_cfg: Optional[dict] = None,
    ):
        self.scene_dir = Path(scene_dir)
        self.patch_size = patch_size
        self.scale = scale
        self.num_patches_per_scene = num_patches_per_scene
        self.augment = augment
        self.augment_cfg = augment_cfg or {}

        scene_files = sorted(self.scene_dir.glob("*.h5"))
        if not scene_files:
            raise FileNotFoundError(f"No .h5 files found in {self.scene_dir}")

        print(f"Scanning {len(scene_files)} scenes (lazy mode - no pixel data loaded)...")
        self.scenes: List[dict] = []
        for f in scene_files:
            meta = _get_scene_meta(f)
            if meta is None:
                print(f"  WARNING: {f.name} skipped (no usable keys).")
                continue
            if meta["hr_key"] is None:
                print(f"  WARNING: {f.name} has no HR key, skipping.")
                continue
            # Validate patch fits in scene
            lr_h, lr_w = meta["lr_shape"][0], meta["lr_shape"][1]
            if lr_h <= patch_size or lr_w <= patch_size:
                print(f"  WARNING: {f.name} LR {meta['lr_shape']} too small for patch {patch_size}, skipping.")
                continue
            self.scenes.append(meta)
            print(f"  {f.name}: LR={meta['lr_shape']}, HR={meta['hr_shape']}")

        if not self.scenes:
            raise RuntimeError("No valid scenes found after scanning.")
        print(f"Dataset: {len(self.scenes)} scenes, {len(self)} patches total (lazy).")

    def __len__(self) -> int:
        return len(self.scenes) * self.num_patches_per_scene

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        scene_idx = idx // self.num_patches_per_scene
        meta = self.scenes[scene_idx]
        ps = self.patch_size
        hr_ps = ps * self.scale

        lr_h, lr_w = meta["lr_shape"][0], meta["lr_shape"][1]
        vmin, vmax = meta["vmin"], meta["vmax"]
        scale_range = vmax - vmin if (vmax - vmin) > 1e-8 else 1.0

        # Random patch coordinates
        y = random.randint(0, lr_h - ps - 1)
        x = random.randint(0, lr_w - ps - 1)
        yr, xr = y * self.scale, x * self.scale

        # Read only the patch region from disk (HDF5 supports hyperslab reads)
        with h5py.File(meta["path"], "r", swmr=True) as f:
            lr_patch = f[meta["lr_key"]][y:y + ps, x:x + ps, :].astype(np.float32)
            hr_patch = f[meta["hr_key"]][yr:yr + hr_ps, xr:xr + hr_ps, :].astype(np.float32)

        # Normalise to [0, 1]
        lr_patch = (lr_patch - vmin) / scale_range
        hr_patch = (hr_patch - vmin) / scale_range
        lr_patch = np.clip(lr_patch, 0.0, 1.0)
        hr_patch = np.clip(hr_patch, 0.0, 1.0)

        if self.augment:
            lr_patch, hr_patch = self._augment(lr_patch, hr_patch)

        # (H, W, C) -> (C, H, W)
        lr_t = torch.from_numpy(lr_patch.transpose(2, 0, 1).copy())
        hr_t = torch.from_numpy(hr_patch.transpose(2, 0, 1).copy())
        return lr_t, hr_t

    def _augment(self, lr: np.ndarray, hr: np.ndarray):
        if self.augment_cfg.get("flip_horizontal", True) and random.random() > 0.5:
            lr = lr[:, ::-1, :]
            hr = hr[:, ::-1, :]
        if self.augment_cfg.get("flip_vertical", True) and random.random() > 0.5:
            lr = lr[::-1, :, :]
            hr = hr[::-1, :, :]
        if self.augment_cfg.get("rotate_90", True) and random.random() > 0.5:
            k = random.randint(1, 3)
            lr = np.rot90(lr, k=k, axes=(0, 1))
            hr = np.rot90(hr, k=k, axes=(0, 1))
        return lr, hr


class HyperspectralInferenceDataset(Dataset):
    """Inference dataset: single LR scene, no HR needed."""

    def __init__(self, lr_path: str):
        self.path = Path(lr_path)
        with h5py.File(self.path, "r") as f:
            keys = list(f.keys())
            lr = None
            for k in keys:
                if "lr" in k.lower():
                    lr = f[k][:].astype(np.float32)
                    break
            if lr is None:
                lr = f[keys[0]][:].astype(np.float32)
        self.lr = lr
        self.vmin = float(lr.min())
        self.vmax = float(lr.max())
        if self.vmax - self.vmin > 1e-8:
            self.lr_norm = (lr - self.vmin) / (self.vmax - self.vmin)
        else:
            self.lr_norm = lr

    def get_full_lr(self) -> np.ndarray:
        return self.lr_norm

    def get_stats(self) -> Tuple[float, float]:
        return self.vmin, self.vmax


class HyperspectralValDataset(Dataset):
    """Validation dataset: reads one scene at a time on demand (lazy)."""

    def __init__(self, scene_dir: str):
        self.scene_dir = Path(scene_dir)
        self.scene_files = sorted(self.scene_dir.glob("*.h5"))
        if not self.scene_files:
            raise FileNotFoundError(f"No .h5 files found in {self.scene_dir}")

    def __len__(self) -> int:
        return len(self.scene_files)

    def __getitem__(self, idx: int):
        """Returns (lr_norm, hr_norm, vmin, vmax, path_str). Reads from disk on demand."""
        path = self.scene_files[idx]
        with h5py.File(path, "r") as f:
            lr_key, hr_key = _discover_lr_hr_keys(f)
            lr = f[lr_key][:].astype(np.float32)
            hr = f[hr_key][:].astype(np.float32) if hr_key else None

        ref = hr if hr is not None else lr
        vmin = float(ref.min())
        vmax = float(ref.max())
        scale_range = (vmax - vmin) if (vmax - vmin) > 1e-8 else 1.0

        lr_n = (lr - vmin) / scale_range
        hr_n = ((hr - vmin) / scale_range) if hr is not None else None
        return lr_n, hr_n, vmin, vmax, str(path)
