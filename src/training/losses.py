"""Loss functions for hyperspectral super-resolution."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMLoss(nn.Module):
    """Spectral Angle Mapper loss.

    Measures the spectral angle between predicted and target pixel spectra.
    Lower is better. Range: [0, pi/2] radians.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, C, H, W)
        dot = (pred * target).sum(dim=1, keepdim=True)
        pred_norm = pred.norm(dim=1, keepdim=True).clamp(min=self.eps)
        target_norm = target.norm(dim=1, keepdim=True).clamp(min=self.eps)
        cos_angle = (dot / (pred_norm * target_norm)).clamp(-1 + self.eps, 1 - self.eps)
        angle = torch.acos(cos_angle)
        return angle.mean()


class SSIMLoss(nn.Module):
    """Structural Similarity loss (1 - SSIM), computed channel-wise."""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma

    def _gaussian_window(self, channels: int, device: torch.device) -> torch.Tensor:
        coords = torch.arange(self.window_size, dtype=torch.float32, device=device)
        coords -= self.window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.expand(channels, 1, self.window_size, self.window_size)
        return window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c = pred.size(1)
        window = self._gaussian_window(c, pred.device)
        mu_pred = F.conv2d(pred, window, padding=self.window_size // 2, groups=c)
        mu_target = F.conv2d(target, window, padding=self.window_size // 2, groups=c)
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target
        sigma_pred = F.conv2d(pred ** 2, window, padding=self.window_size // 2, groups=c) - mu_pred_sq
        sigma_target = F.conv2d(target ** 2, window, padding=self.window_size // 2, groups=c) - mu_target_sq
        sigma_cross = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=c) - mu_cross
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_cross + c1) * (2 * sigma_cross + c2)) / \
               ((mu_pred_sq + mu_target_sq + c1) * (sigma_pred + sigma_target + c2))
        return 1.0 - ssim.mean()


class CombinedLoss(nn.Module):
    """Combined L1 + SAM + optional SSIM loss."""

    def __init__(self, l1_weight: float = 1.0, sam_weight: float = 0.1, ssim_weight: float = 0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.sam_weight = sam_weight
        self.ssim_weight = ssim_weight
        self.l1 = nn.L1Loss()
        self.sam = SAMLoss()
        self.ssim = SSIMLoss() if ssim_weight > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.l1_weight * self.l1(pred, target)
        loss = loss + self.sam_weight * self.sam(pred, target)
        if self.ssim is not None and self.ssim_weight > 0:
            loss = loss + self.ssim_weight * self.ssim(pred, target)
        return loss


def build_loss(cfg: dict) -> nn.Module:
    """Build loss function from config."""
    loss_cfg = cfg.get("training", {})
    l1_w = float(loss_cfg.get("l1_weight", 1.0))
    sam_w = float(loss_cfg.get("sam_weight", 0.1))
    ssim_w = float(loss_cfg.get("ssim_weight", 0.0))

    if sam_w == 0 and ssim_w == 0:
        print("Using L1 loss only")
        return nn.L1Loss()

    print(f"Using combined loss: L1={l1_w}, SAM={sam_w}, SSIM={ssim_w}")
    return CombinedLoss(l1_weight=l1_w, sam_weight=sam_w, ssim_weight=ssim_w)
