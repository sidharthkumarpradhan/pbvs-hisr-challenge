"""PyTorch datasets for hyperspectral SR training and inference."""

import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_scene_arrays(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load LR and HR arrays from a scene H5 file."""
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        lr = hr = None
        for k in keys:
            kl = k.lower()
            arr = f[k][:]
            if "lr" in kl:
                lr = arr.astype(np.float32)
            elif "hr" in kl:
                hr = arr.astype(np.float32)
        if lr is None and hr is None:
            if len(keys) == 2:
                lr = f[keys[0]][:].astype(np.float32)
                hr = f[keys[1]][:].astype(np.float32)
            elif len(keys) == 1:
                lr = f[keys[0]][:].astype(np.float32)
    return lr, hr


def _normalize_scene(lr: np.ndarray, hr: np.ndarray | None):
    """Normalize LR and HR to [0, 1] based on HR max/min (or LR if no HR)."""
    ref = hr if hr is not None else lr
    vmin = float(ref.min())
    vmax = float(ref.max())
    if vmax - vmin < 1e-8:
        return lr, hr, vmin, vmax
    lr_n = (lr - vmin) / (vmax - vmin)
    hr_n = ((hr - vmin) / (vmax - vmin)) if hr is not None else None
    return lr_n, hr_n, vmin, vmax


class HyperspectralPatchDataset(Dataset):
    """Training dataset: random patches from LR-HR scene pairs."""

    def __init__(
        self,
        scene_dir: str | Path,
        patch_size: int = 64,
        scale: int = 4,
        num_patches_per_scene: int = 200,
        augment: bool = True,
        augment_cfg: dict = None,
    ):
        self.scene_dir = Path(scene_dir)
        self.patch_size = patch_size
        self.scale = scale
        self.num_patches_per_scene = num_patches_per_scene
        self.augment = augment
        self.augment_cfg = augment_cfg or {}

        self.scene_files = sorted(self.scene_dir.glob("*.h5"))
        if not self.scene_files:
            raise FileNotFoundError(f"No .h5 files found in {self.scene_dir}")

        print(f"Loading {len(self.scene_files)} scenes into memory...")
        self.scenes = []
        for f in self.scene_files:
            lr, hr = _load_scene_arrays(f)
            if hr is None:
                print(f"  WARNING: {f.name} has no HR, skipping.")
                continue
            lr_n, hr_n, vmin, vmax = _normalize_scene(lr, hr)
            self.scenes.append((lr_n, hr_n, vmin, vmax))
            print(f"  Loaded {f.name}: LR={lr.shape}, HR={hr.shape}")

        print(f"Dataset: {len(self.scenes)} scenes, ~{len(self)} patches total")

    def __len__(self) -> int:
        return len(self.scenes) * self.num_patches_per_scene

    def __getitem__(self, idx: int):
        scene_idx = idx // self.num_patches_per_scene
        lr, hr, _, _ = self.scenes[scene_idx]

        lr_h, lr_w, c = lr.shape
        ps = self.patch_size
        hr_ps = ps * self.scale

        if lr_h <= ps or lr_w <= ps:
            raise ValueError(f"Scene LR {lr.shape} smaller than patch size {ps}")
        y = random.randint(0, lr_h - ps - 1)
        x = random.randint(0, lr_w - ps - 1)
        lr_patch = lr[y:y + ps, x:x + ps, :]
        hr_patch = hr[y * self.scale:y * self.scale + hr_ps, x * self.scale:x * self.scale + hr_ps, :]

        if self.augment:
            lr_patch, hr_patch = self._augment(lr_patch, hr_patch)

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

    def __init__(self, lr_path: str | Path):
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

    def get_stats(self) -> tuple[float, float]:
        return self.vmin, self.vmax


class HyperspectralValDataset(Dataset):
    """Validation dataset: full scenes, returns (lr, hr, vmin, vmax)."""

    def __init__(self, scene_dir: str | Path):
        self.scene_dir = Path(scene_dir)
        self.scene_files = sorted(self.scene_dir.glob("*.h5"))
        if not self.scene_files:
            raise FileNotFoundError(f"No .h5 files found in {self.scene_dir}")

    def __len__(self) -> int:
        return len(self.scene_files)

    def __getitem__(self, idx: int):
        path = self.scene_files[idx]
        lr, hr = _load_scene_arrays(path)
        lr_n, hr_n, vmin, vmax = _normalize_scene(lr, hr)
        return lr_n, hr_n, vmin, vmax, str(path)
