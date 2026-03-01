"""HISRNet: Hyperspectral Image Super-Resolution Network."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Basic residual block with two 3x3 convolutions."""

    def __init__(self, num_features: int, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.res_scale * self.body(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, num_features: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_features, max(1, num_features // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, num_features // reduction), num_features, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SpectralGroupConv(nn.Module):
    """3D convolution across spectral dimension to capture band correlations."""

    def __init__(self, in_channels: int, out_channels: int, spectral_kernel: int = 5):
        super().__init__()
        pad = spectral_kernel // 2
        self.conv3d = nn.Conv3d(
            1, out_channels,
            kernel_size=(spectral_kernel, 3, 3),
            padding=(pad, 1, 1),
        )
        self.proj = nn.Conv2d(out_channels * in_channels, out_channels, 1)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        b, c, h, w = x.shape
        x3d = x.unsqueeze(1)  # (B, 1, C, H, W)
        out = self.conv3d(x3d)  # (B, out_ch, C, H, W)
        out = out.view(b, -1, h, w)  # (B, out_ch*C, H, W)
        out = self.proj(out)  # (B, out_ch, H, W)
        return out


class HISRNet(nn.Module):
    """Hyperspectral Image Super-Resolution Network."""

    def __init__(
        self,
        num_bands: int = 224,
        scale: int = 4,
        num_features: int = 128,
        num_blocks: int = 20,
        spectral_features: int = 32,
        spectral_kernel: int = 7,
        use_3d: bool = True,
        use_attention: bool = True,
        res_scale: float = 0.1,
    ):
        super().__init__()
        self.scale = scale
        self.num_bands = num_bands
        self.use_3d = use_3d

        in_features = num_bands
        if use_3d:
            self.spectral_head = SpectralGroupConv(num_bands, spectral_features, spectral_kernel)
            in_features = spectral_features
        self.head = nn.Conv2d(in_features, num_features, 3, 1, 1)

        body = []
        for i in range(num_blocks):
            body.append(ResBlock(num_features, res_scale=res_scale))
            if use_attention and i % 4 == 3:
                body.append(ChannelAttention(num_features))
        body.append(nn.Conv2d(num_features, num_features, 3, 1, 1))
        self.body = nn.Sequential(*body)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

        self.tail = nn.Conv2d(num_features, num_bands, 3, 1, 1)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)

        if self.use_3d:
            feat = self.spectral_head(x)
        else:
            feat = x
        feat = self.head(feat)

        res = self.body(feat)
        feat = feat + res

        feat = self.upsample(feat)
        out = self.tail(feat)
        return out + base


class EDSRHyperspectral(nn.Module):
    """Simplified EDSR baseline for hyperspectral SR."""

    def __init__(
        self,
        num_bands: int = 224,
        scale: int = 4,
        num_features: int = 64,
        num_blocks: int = 16,
        res_scale: float = 0.1,
    ):
        super().__init__()
        self.scale = scale

        self.head = nn.Conv2d(num_bands, num_features, 3, 1, 1)
        body = [ResBlock(num_features, res_scale=res_scale) for _ in range(num_blocks)]
        body.append(nn.Conv2d(num_features, num_features, 3, 1, 1))
        self.body = nn.Sequential(*body)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(num_features, num_bands, 3, 1, 1)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        feat = self.head(x)
        feat = feat + self.body(feat)
        feat = self.upsample(feat)
        return self.tail(feat) + base


class BicubicBaseline(nn.Module):
    """Simple bicubic upsampling baseline."""

    def __init__(self, scale: int = 4, **kwargs):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)


MODEL_REGISTRY = {
    "HISRNet": HISRNet,
    "EDSR": EDSRHyperspectral,
    "Bicubic": BicubicBaseline,
}


def build_model(cfg: dict) -> nn.Module:
    """Instantiate model from config dict."""
    model_cfg = cfg["model"]
    name = model_cfg["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY.keys())}")
    model_class = MODEL_REGISTRY[name]
    kwargs = {
        "num_bands": model_cfg.get("bands", cfg["data"].get("bands", 224)),
        "scale": model_cfg.get("scale", 4),
    }
    if name == "HISRNet":
        kwargs.update({
            "num_features": model_cfg.get("num_features", 128),
            "num_blocks": model_cfg.get("num_blocks", 20),
            "spectral_features": model_cfg.get("num_spectral_features", 32),
            "use_3d": model_cfg.get("use_3d_features", True),
            "use_attention": model_cfg.get("use_spectral_attention", True),
        })
    elif name == "EDSR":
        kwargs.update({
            "num_features": model_cfg.get("num_features", 64),
            "num_blocks": model_cfg.get("num_blocks", 16),
        })
    return model_class(**kwargs)
