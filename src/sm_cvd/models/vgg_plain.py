"""Vgg plain."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sm_cvd.models.msinet import (
    IMAGENET_MEAN_BGR,
    VGG16Encoder,
    Decoder,
    normalize_minmax,
)


class PlainBottleneck(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, in_ch: int = 1280, out_ch: int = 256) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.reduce(x))


class VGGPlain(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, subtract_imagenet_mean: bool = True) -> None:
        super().__init__()
        self.encoder = VGG16Encoder()
        self.bottleneck = PlainBottleneck(in_ch=1280, out_ch=256)
        self.decoder = Decoder()
        self.subtract_imagenet_mean = subtract_imagenet_mean
        self.register_buffer(
            "_mean",
            torch.tensor(IMAGENET_MEAN_BGR, dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.subtract_imagenet_mean:
            x = x - self._mean
        feats = self.encoder(x)                  # [B, 1280, H/8, W/8]
        ctx = self.bottleneck(feats)             # [B, 256, H/8, W/8]
        out = self.decoder(ctx)                  # [B, 1, H, W]
        out = normalize_minmax(out)
        return out

    def load_vgg16_imagenet(self, weights: dict | None = None) -> None:
        """(documentation omitted in the English release)."""
        if weights is None:
            from torchvision.models import vgg16, VGG16_Weights
            weights = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).state_dict()
        from collections import OrderedDict
        mapping = {
            "conv1_1": 0, "conv1_2": 2,
            "conv2_1": 5, "conv2_2": 7,
            "conv3_1": 10, "conv3_2": 12, "conv3_3": 14,
            "conv4_1": 17, "conv4_2": 19, "conv4_3": 21,
            "conv5_1": 24, "conv5_2": 26, "conv5_3": 28,
        }
        own = self.encoder.state_dict()
        new_state = OrderedDict(own)
        for name, idx in mapping.items():
            new_state[f"{name}.weight"] = weights[f"features.{idx}.weight"].clone()
            new_state[f"{name}.bias"] = weights[f"features.{idx}.bias"].clone()
        self.encoder.load_state_dict(new_state, strict=True)


__all__ = ["VGGPlain", "PlainBottleneck"]
