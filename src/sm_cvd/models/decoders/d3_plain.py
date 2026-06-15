"""D3 plain."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sm_cvd.models.msinet import (
    IMAGENET_MEAN_BGR,
    VGG16Encoder,
    normalize_minmax,
    _conv1x1,
    _conv3x3,
)


class PlainDecoder(nn.Module):
    """Plain decoder: 1x1 -> 3x3 -> 3x3 conv with bilinear upsample x3."""

    def __init__(self, in_ch: int = 512) -> None:
        super().__init__()
        self.reduce = _conv1x1(in_ch, 256)
        self.conv1 = _conv3x3(256, 128)
        self.conv2 = _conv3x3(128, 64)
        self.conv3 = _conv3x3(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.reduce(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.conv3(x)
        return x


class D3PlainNet(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, subtract_imagenet_mean: bool = True) -> None:
        super().__init__()
        self.encoder = VGG16Encoder()
        self.decoder = PlainDecoder(in_ch=512)
        self.subtract_imagenet_mean = subtract_imagenet_mean
        self.register_buffer(
            "_mean",
            torch.tensor(IMAGENET_MEAN_BGR, dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.subtract_imagenet_mean:
            x = x - self._mean




        feats_1280 = self.encoder(x)
        layer18 = feats_1280[:, 768:1280, :, :]
        out = self.decoder(layer18)
        out = normalize_minmax(out)
        return out
