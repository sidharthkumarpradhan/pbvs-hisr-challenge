import numpy as np


def compute_psnr(sr: np.ndarray, hr: np.ndarray, data_range: float = None) -> float:
    """Compute PSNR between SR and HR arrays.

    Both arrays should be (H, W, C) float32.
    Returns PSNR in dB (higher is better).
    """
    if data_range is None:
        data_range = float(hr.max() - hr.min())
    if data_range == 0:
        return float("inf")
    mse = np.mean((sr.astype(np.float64) - hr.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def compute_sam(sr: np.ndarray, hr: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Spectral Angle Mapper (SAM) in degrees.

    Both arrays should be (H, W, C) float32.
    Returns average SAM in degrees (lower is better).
    """
    sr_f = sr.astype(np.float64)
    hr_f = hr.astype(np.float64)
    dot = np.sum(sr_f * hr_f, axis=-1)
    norm_sr = np.linalg.norm(sr_f, axis=-1)
    norm_hr = np.linalg.norm(hr_f, axis=-1)
    cos_angle = dot / (norm_sr * norm_hr + eps)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return float(np.mean(angle_deg))


def compute_ssim(sr: np.ndarray, hr: np.ndarray, data_range: float = None) -> float:
    """Compute average SSIM over all spectral bands.

    Both arrays (H, W, C). Returns average SSIM (higher is better).
    """
    from skimage.metrics import structural_similarity
    if data_range is None:
        data_range = float(hr.max() - hr.min())
    ssim_vals = []
    for c in range(sr.shape[2]):
        s = structural_similarity(
            sr[:, :, c].astype(np.float64),
            hr[:, :, c].astype(np.float64),
            data_range=data_range,
        )
        ssim_vals.append(s)
    return float(np.mean(ssim_vals))


def evaluate_patch(sr: np.ndarray, hr: np.ndarray) -> dict:
    """Compute all metrics for a single patch. Returns dict of metric values."""
    data_range = float(hr.max() - hr.min())
    psnr = compute_psnr(sr, hr, data_range=data_range)
    sam = compute_sam(sr, hr)
    return {"psnr": psnr, "sam": sam, "data_range": data_range}


def aggregate_metrics(results: list[dict]) -> dict:
    """Aggregate metric dicts from multiple patches into mean values."""
    keys = [k for k in results[0].keys() if k != "data_range"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in results]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    return agg
