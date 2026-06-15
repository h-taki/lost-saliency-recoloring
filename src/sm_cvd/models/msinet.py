"""MSI-Net saliency model (PyTorch re-implementation of the contextual encoder-decoder)."""
from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN_BGR: tuple[float, float, float] = (103.939, 116.779, 123.68)


def _conv3x3(in_ch: int, out_ch: int, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3,
                     padding=dilation, dilation=dilation)


def _conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1)


class SamePadMaxPool2d(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, kernel_size: int = 2, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:



        pad = self.kernel_size - self.stride
        if pad <= 0:
            return F.max_pool2d(x, self.kernel_size, self.stride)
        pad_l = pad // 2
        pad_r = pad - pad_l

        x = F.pad(x, [pad_l, pad_r, pad_l, pad_r])
        return F.max_pool2d(x, self.kernel_size, self.stride)


class VGG16Encoder(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self) -> None:
        super().__init__()
        # block1
        self.conv1_1 = _conv3x3(3, 64)
        self.conv1_2 = _conv3x3(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        # block2
        self.conv2_1 = _conv3x3(64, 128)
        self.conv2_2 = _conv3x3(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        # block3
        self.conv3_1 = _conv3x3(128, 256)
        self.conv3_2 = _conv3x3(256, 256)
        self.conv3_3 = _conv3x3(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4_1 = _conv3x3(256, 512)
        self.conv4_2 = _conv3x3(512, 512)
        self.conv4_3 = _conv3x3(512, 512)
        self.pool4 = SamePadMaxPool2d(kernel_size=2, stride=1)

        self.conv5_1 = _conv3x3(512, 512, dilation=2)
        self.conv5_2 = _conv3x3(512, 512, dilation=2)
        self.conv5_3 = _conv3x3(512, 512, dilation=2)
        self.pool5 = SamePadMaxPool2d(kernel_size=2, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        layer10 = self.pool3(x)                  # 256ch, 1/8
        x = F.relu(self.conv4_1(layer10))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        layer14 = self.pool4(x)                  # 512ch, 1/8 (pool stride 1)
        x = F.relu(self.conv5_1(layer14))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        layer18 = self.pool5(x)                  # 512ch, 1/8


        return torch.cat([layer10, layer14, layer18], dim=1)  # 1280ch


class ASPP(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, in_ch: int = 1280, mid_ch: int = 256) -> None:
        super().__init__()
        self.branch1 = _conv1x1(in_ch, mid_ch)
        self.branch2 = _conv3x3(in_ch, mid_ch, dilation=4)
        self.branch3 = _conv3x3(in_ch, mid_ch, dilation=8)
        self.branch4 = _conv3x3(in_ch, mid_ch, dilation=12)
        self.branch5_conv = _conv1x1(in_ch, mid_ch)
        self.fuse = _conv1x1(mid_ch * 5, mid_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = F.relu(self.branch1(x))
        b2 = F.relu(self.branch2(x))
        b3 = F.relu(self.branch3(x))
        b4 = F.relu(self.branch4(x))
        # Global avg pool → 1x1 conv → upsample
        b5 = x.mean(dim=(2, 3), keepdim=True)
        b5 = F.relu(self.branch5_conv(b5))
        b5 = F.interpolate(b5, size=x.shape[-2:], mode="bilinear", align_corners=True)
        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        return F.relu(self.fuse(out))


class Decoder(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = _conv3x3(256, 128)
        self.conv2 = _conv3x3(128, 64)
        self.conv3 = _conv3x3(64, 32)
        self.conv4 = _conv3x3(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


def normalize_minmax(maps: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """(documentation omitted in the English release)."""
    min_per = torch.amin(maps, dim=(1, 2, 3), keepdim=True)
    maps = maps - min_per
    max_per = torch.amax(maps, dim=(1, 2, 3), keepdim=True)
    return maps / (max_per + eps)


class MSINet(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, subtract_imagenet_mean: bool = True) -> None:
        super().__init__()
        self.encoder = VGG16Encoder()
        self.aspp = ASPP(in_ch=1280, mid_ch=256)
        self.decoder = Decoder()
        self.subtract_imagenet_mean = subtract_imagenet_mean


        self.register_buffer(
            "_mean",
            torch.tensor(IMAGENET_MEAN_BGR, dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.subtract_imagenet_mean:
            x = x - self._mean
        feats = self.encoder(x)
        ctx = self.aspp(feats)
        out = self.decoder(ctx)
        out = normalize_minmax(out)
        return out

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    def load_vgg16_imagenet(self, weights: dict | None = None) -> None:
        """(documentation omitted in the English release)."""
        if weights is None:
            from torchvision.models import vgg16, VGG16_Weights
            weights = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).state_dict()

        # torchvision VGG16: features.0=conv1_1, .2=conv1_2, .5=conv2_1, .7=conv2_2,
        #                    .10=conv3_1, .12=conv3_2, .14=conv3_3,
        #                    .17=conv4_1, .19=conv4_2, .21=conv4_3,
        #                    .24=conv5_1, .26=conv5_2, .28=conv5_3
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
            w_key = f"features.{idx}.weight"
            b_key = f"features.{idx}.bias"
            new_state[f"{name}.weight"] = weights[w_key].clone()
            new_state[f"{name}.bias"] = weights[b_key].clone()
        self.encoder.load_state_dict(new_state)


__all__ = ["MSINet", "VGG16Encoder", "ASPP", "Decoder", "normalize_minmax",
           "IMAGENET_MEAN_BGR"]
