"""Sam resnet."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sm_cvd.models.sam_vgg import (
    AttentiveConvLSTM,
    LearningPrior,
    _normalize_minmax,
)


class SAMResNetEncoder(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, pretrained_imagenet: bool = True) -> None:
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_imagenet else None
        base = resnet50(
            weights=weights,
            replace_stride_with_dilation=[False, True, True],
        )

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)         # [B, 2048, H/8, W/8]
        return x


class SAMResNet(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(
        self,
        n_lstm_steps: int = 4,
        nb_gaussian: int = 16,
        upsample_factor: int = 8,
        pretrained_imagenet: bool = True,
        reduction_channels: int = 512,
    ) -> None:
        super().__init__()
        self.encoder = SAMResNetEncoder(pretrained_imagenet=pretrained_imagenet)

        self.reduce = nn.Conv2d(2048, reduction_channels, kernel_size=1)
        self.reduce_bn = nn.BatchNorm2d(reduction_channels)

        self.convlstm = AttentiveConvLSTM(
            channels=reduction_channels,
            kernel_size=3,
            n_steps=n_lstm_steps,
        )

        # Prior 1
        self.prior1 = LearningPrior(nb_gaussian=nb_gaussian)
        self.atrous1 = nn.Conv2d(
            reduction_channels + nb_gaussian, reduction_channels,
            kernel_size=5, padding=8, dilation=4,
        )
        # Prior 2
        self.prior2 = LearningPrior(nb_gaussian=nb_gaussian)
        self.atrous2 = nn.Conv2d(
            reduction_channels + nb_gaussian, reduction_channels,
            kernel_size=5, padding=8, dilation=4,
        )
        # Head
        self.conv_out = nn.Conv2d(reduction_channels, 1, kernel_size=1)

        self.upsample_factor = upsample_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        feat = self.encoder(x)                  # [B, 2048, H/8, W/8]
        feat = F.relu(self.reduce_bn(self.reduce(feat)))   # [B, 512, H/8, W/8]

        h = self.convlstm(feat)                 # [B, 512, H/8, W/8]

        # Prior 1: concat + atrous conv
        p1 = self.prior1(h)
        h = torch.cat([h, p1], dim=1)
        h = F.relu(self.atrous1(h))             # [B, 512, H/8, W/8]

        # Prior 2: concat + atrous conv
        p2 = self.prior2(h)
        h = torch.cat([h, p2], dim=1)
        h = F.relu(self.atrous2(h))             # [B, 512, H/8, W/8]


        out = self.conv_out(h)                  # [B, 1, H/8, W/8]
        out = F.interpolate(out, scale_factor=self.upsample_factor,
                            mode="bilinear", align_corners=True)
        out = _normalize_minmax(out)
        return out


__all__ = ["SAMResNet", "SAMResNetEncoder"]
