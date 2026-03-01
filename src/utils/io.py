"""HDF5 I/O utilities for hyperspectral data."""

from pathlib import Path

import h5py
import numpy as np


def load_h5_scene(path: str | Path) -> dict[str, np.ndarray]:
    """Load all datasets from an HDF5 file into a dict."""
    path = Path(path)
    data = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            data[key] = f[key][:].astype(np.float32)
    return data


def load_lr_hr(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load LR and HR arrays from an HDF5 file.

    Returns (lr, hr) where hr may be None if not present.
    Arrays are shaped (H, W, C).
    """
    data = load_h5_scene(path)
    keys = list(data.keys())
    lr = hr = None
    for k in keys:
        kl = k.lower()
        if "lr" in kl:
            lr = data[k].astype(np.float32)
        elif "hr" in kl:
            hr = data[k].astype(np.float32)
    if lr is None and hr is None:
        if len(keys) >= 2:
            lr = data[keys[0]].astype(np.float32)
            hr = data[keys[1]].astype(np.float32)
        elif len(keys) == 1:
            lr = data[keys[0]].astype(np.float32)
    return lr, hr


def save_h5_prediction(path: str | Path, arr: np.ndarray, key: str = "sr") -> None:
    """Save a prediction array to HDF5. arr should be (H, W, C)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=arr.astype(np.float32), compression="gzip")


def normalize(
    arr: np.ndarray, vmin: float = None, vmax: float = None
) -> tuple[np.ndarray, float, float]:
    """Normalize array to [0, 1] range."""
    if vmin is None:
        vmin = float(arr.min())
    if vmax is None:
        vmax = float(arr.max())
    if vmax == vmin:
        return np.zeros_like(arr), vmin, vmax
    return (arr - vmin) / (vmax - vmin), vmin, vmax


def denormalize(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Denormalize array from [0, 1] back to original range."""
    return arr * (vmax - vmin) + vmin
