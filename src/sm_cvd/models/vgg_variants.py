"""Vgg variants."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sm_cvd.models.msinet import (
    IMAGENET_MEAN_BGR,
    ASPP,
    Decoder,
    SamePadMaxPool2d,
    normalize_minmax,
    _conv3x3,
)



VGG_CONFIGS: dict[str, list[int]] = {
    "vgg11": [1, 1, 2, 2, 2],
    "vgg13": [2, 2, 2, 2, 2],
    "vgg16": [2, 2, 3, 3, 3],
    "vgg19": [2, 2, 4, 4, 4],
}

BLOCK_CHANNELS = [64, 128, 256, 512, 512]


class VGGVariantEncoder(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, variant: str = "vgg16") -> None:
        super().__init__()
        if variant not in VGG_CONFIGS:
            raise ValueError(f"unknown VGG variant: {variant}")
        self.variant = variant
        cfg = VGG_CONFIGS[variant]

        # Build blocks
        # block1
        b1 = []
        in_ch = 3
        for j in range(cfg[0]):
            b1.append(_conv3x3(in_ch, BLOCK_CHANNELS[0]))
            in_ch = BLOCK_CHANNELS[0]
        self.block1 = nn.ModuleList(b1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # block2
        b2 = []
        in_ch = BLOCK_CHANNELS[0]
        for j in range(cfg[1]):
            b2.append(_conv3x3(in_ch, BLOCK_CHANNELS[1]))
            in_ch = BLOCK_CHANNELS[1]
        self.block2 = nn.ModuleList(b2)
        self.pool2 = nn.MaxPool2d(2, 2)

        # block3
        b3 = []
        in_ch = BLOCK_CHANNELS[1]
        for j in range(cfg[2]):
            b3.append(_conv3x3(in_ch, BLOCK_CHANNELS[2]))
            in_ch = BLOCK_CHANNELS[2]
        self.block3 = nn.ModuleList(b3)
        self.pool3 = nn.MaxPool2d(2, 2)

        # block4 (pool stride=1)
        b4 = []
        in_ch = BLOCK_CHANNELS[2]
        for j in range(cfg[3]):
            b4.append(_conv3x3(in_ch, BLOCK_CHANNELS[3]))
            in_ch = BLOCK_CHANNELS[3]
        self.block4 = nn.ModuleList(b4)
        self.pool4 = SamePadMaxPool2d(kernel_size=2, stride=1)

        # block5 (dilation=2 + pool stride=1)
        b5 = []
        in_ch = BLOCK_CHANNELS[3]
        for j in range(cfg[4]):
            b5.append(_conv3x3(in_ch, BLOCK_CHANNELS[4], dilation=2))
            in_ch = BLOCK_CHANNELS[4]
        self.block5 = nn.ModuleList(b5)
        self.pool5 = SamePadMaxPool2d(kernel_size=2, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.block1:
            x = F.relu(conv(x))
        x = self.pool1(x)
        for conv in self.block2:
            x = F.relu(conv(x))
        x = self.pool2(x)
        for conv in self.block3:
            x = F.relu(conv(x))
        layer10 = self.pool3(x)  # 256ch, 1/8
        x = layer10
        for conv in self.block4:
            x = F.relu(conv(x))
        layer14 = self.pool4(x)  # 512ch, 1/8
        x = layer14
        for conv in self.block5:
            x = F.relu(conv(x))
        layer18 = self.pool5(x)  # 512ch, 1/8
        return torch.cat([layer10, layer14, layer18], dim=1)  # 1280ch


class MSINetVGGVariant(nn.Module):
    """MSI-Net with configurable VGG depth (VGG11/13/16/19)."""

    def __init__(self, variant: str = "vgg16", subtract_imagenet_mean: bool = True) -> None:
        super().__init__()
        self.encoder = VGGVariantEncoder(variant=variant)
        self.aspp = ASPP(in_ch=1280, mid_ch=256)
        self.decoder = Decoder()
        self.subtract_imagenet_mean = subtract_imagenet_mean
        self.register_buffer(
            "_mean",
            torch.tensor(IMAGENET_MEAN_BGR, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.subtract_imagenet_mean:
            x = x - self._mean
        feats = self.encoder(x)
        ctx = self.aspp(feats)
        out = self.decoder(ctx)
        return normalize_minmax(out)


def load_torchvision_vgg_weights(model: MSINetVGGVariant) -> int:
    """(documentation omitted in the English release)."""
    from torchvision.models import (
        vgg11, vgg13, vgg16, vgg19,
        VGG11_Weights, VGG13_Weights, VGG16_Weights, VGG19_Weights,
    )
    variant_to_loader = {
        "vgg11": (vgg11, VGG11_Weights.IMAGENET1K_V1),
        "vgg13": (vgg13, VGG13_Weights.IMAGENET1K_V1),
        "vgg16": (vgg16, VGG16_Weights.IMAGENET1K_V1),
        "vgg19": (vgg19, VGG19_Weights.IMAGENET1K_V1),
    }
    if model.variant not in variant_to_loader:
        return 0
    loader_fn, weights = variant_to_loader[model.variant]
    vgg = loader_fn(weights=weights)


    conv_layers_torch = [m for m in vgg.features if isinstance(m, nn.Conv2d)]

    encoder = model.encoder
    our_conv_lists = [encoder.block1, encoder.block2, encoder.block3,
                      encoder.block4, encoder.block5]
    our_convs = [conv for lst in our_conv_lists for conv in lst]

    if len(conv_layers_torch) != len(our_convs):
        print(f"[load_torchvision_vgg_weights/{model.variant}] count mismatch: "
              f"torch {len(conv_layers_torch)} vs ours {len(our_convs)}")
    loaded = 0
    for dst, src in zip(our_convs, conv_layers_torch):
        try:
            dst.weight.data.copy_(src.weight.data)
            if dst.bias is not None and src.bias is not None:
                dst.bias.data.copy_(src.bias.data)
            loaded += 1
        except Exception as e:
            print(f"  failed: {e}")
    return loaded


__all__ = ["VGGVariantEncoder", "MSINetVGGVariant", "load_torchvision_vgg_weights",
           "VGG_CONFIGS"]
