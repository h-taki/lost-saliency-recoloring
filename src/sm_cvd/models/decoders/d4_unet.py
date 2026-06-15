"""D4 unet."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sm_cvd.models.msinet import (
    IMAGENET_MEAN_BGR,
    SamePadMaxPool2d,
    normalize_minmax,
    _conv1x1,
    _conv3x3,
)


class _VGG16SkipEncoder(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self) -> None:
        super().__init__()
        # block1: 1/1
        self.conv1_1 = _conv3x3(3, 64)
        self.conv1_2 = _conv3x3(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        # block2: 1/2
        self.conv2_1 = _conv3x3(64, 128)
        self.conv2_2 = _conv3x3(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        # block3: 1/4
        self.conv3_1 = _conv3x3(128, 256)
        self.conv3_2 = _conv3x3(256, 256)
        self.conv3_3 = _conv3x3(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        # block4: 1/8
        self.conv4_1 = _conv3x3(256, 512)
        self.conv4_2 = _conv3x3(512, 512)
        self.conv4_3 = _conv3x3(512, 512)
        self.pool4 = SamePadMaxPool2d(kernel_size=2, stride=1)
        # block5: 1/8 (dilation=2)
        self.conv5_1 = _conv3x3(512, 512, dilation=2)
        self.conv5_2 = _conv3x3(512, 512, dilation=2)
        self.conv5_3 = _conv3x3(512, 512, dilation=2)
        self.pool5 = SamePadMaxPool2d(kernel_size=2, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        x = F.relu(self.conv1_1(x))
        skip1 = F.relu(self.conv1_2(x))  # 64ch, 1/1
        x = self.pool1(skip1)
        x = F.relu(self.conv2_1(x))
        skip2 = F.relu(self.conv2_2(x))  # 128ch, 1/2
        x = self.pool2(skip2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        skip3 = F.relu(self.conv3_3(x))  # 256ch, 1/4
        x = self.pool3(skip3)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        skip4 = F.relu(self.conv4_3(x))  # 512ch, 1/8
        x = self.pool4(skip4)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))  # 512ch, 1/8
        bottleneck = self.pool5(x)  # 512ch, 1/8
        return skip1, skip2, skip3, skip4, bottleneck


class UNetDecoder(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self) -> None:
        super().__init__()

        self.dec4 = _conv3x3(512 + 512, 256)

        self.dec3 = _conv3x3(256 + 256, 128)

        self.dec2 = _conv3x3(128 + 128, 64)

        self.dec1 = _conv3x3(64 + 64, 32)

        self.final = _conv1x1(32, 1)

    def forward(self, skip1: torch.Tensor, skip2: torch.Tensor, skip3: torch.Tensor,
                skip4: torch.Tensor, bottleneck: torch.Tensor) -> torch.Tensor:
        # stage 4 (1/8): bottleneck + skip4
        x = torch.cat([bottleneck, skip4], dim=1)
        x = F.relu(self.dec4(x))
        # stage 3 (1/4): upsample x2 + skip3
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, skip3], dim=1)
        x = F.relu(self.dec3(x))
        # stage 2 (1/2): upsample x2 + skip2
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, skip2], dim=1)
        x = F.relu(self.dec2(x))
        # stage 1 (1/1): upsample x2 + skip1
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, skip1], dim=1)
        x = F.relu(self.dec1(x))

        out = self.final(x)
        return out


class D4UNet(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, subtract_imagenet_mean: bool = True) -> None:
        super().__init__()
        self.encoder = _VGG16SkipEncoder()
        self.decoder = UNetDecoder()
        self.subtract_imagenet_mean = subtract_imagenet_mean
        self.register_buffer(
            "_mean",
            torch.tensor(IMAGENET_MEAN_BGR, dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.subtract_imagenet_mean:
            x = x - self._mean
        skip1, skip2, skip3, skip4, bottleneck = self.encoder(x)
        out = self.decoder(skip1, skip2, skip3, skip4, bottleneck)
        out = normalize_minmax(out)
        return out
