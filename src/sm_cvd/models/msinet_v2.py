"""Msinet v2."""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from sm_cvd.models.msinet import (
    MSINet,
    VGG16Encoder,
    IMAGENET_MEAN_BGR,
    normalize_minmax,
)


class VGG16Encoder4ch(VGG16Encoder):
    """(documentation omitted in the English release)."""

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()

        old_w = self.conv1_1.weight.data.clone()
        old_b = self.conv1_1.bias.data.clone() if self.conv1_1.bias is not None else None
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        with torch.no_grad():

            self.conv1_1.weight.data[:, :3, :, :] = old_w
            self.conv1_1.weight.data[:, 3:, :, :] = 0.0
            if self.conv1_1.bias is not None and old_b is not None:
                self.conv1_1.bias.data.copy_(old_b)


class MSINetV2(MSINet):
    """(documentation omitted in the English release)."""

    def __init__(self, subtract_imagenet_mean: bool = True,
                 residual_output: bool = False,
                 extra_channels: int = 1) -> None:

        super().__init__(subtract_imagenet_mean=subtract_imagenet_mean)


        if residual_output and extra_channels != 1:
            raise ValueError("residual_output=True is supported only with extra_channels=1")
        self.extra_channels = extra_channels
        self.encoder = VGG16Encoder4ch(in_channels=3 + extra_channels)




        self.residual_output = residual_output

    def forward(self, image: torch.Tensor, normal_sm: torch.Tensor) -> torch.Tensor:
        """(documentation omitted in the English release)."""
        if self.subtract_imagenet_mean:
            image = image - self._mean



        nsm = normal_sm
        if nsm.max() > 1.5:
            nsm = nsm / 255.0
        x4 = torch.cat([image, nsm], dim=1)         # (B, 4, H, W)
        feats = self.encoder(x4)
        ctx = self.aspp(feats)
        out = self.decoder(ctx)
        if self.residual_output:

            prior = nsm
            if prior.shape[-2:] != out.shape[-2:]:
                prior = F.interpolate(
                    prior, size=out.shape[-2:], mode="bilinear", align_corners=False
                )
            out = out + prior
        out = normalize_minmax(out)
        return out

    def load_vgg16_imagenet(self, weights: dict | None = None) -> None:
        """(documentation omitted in the English release)."""
        if weights is None:
            from torchvision.models import vgg16, VGG16_Weights
            weights = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).state_dict()

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
            tv_w = weights[w_key]
            if name == "conv1_1":

                cur = new_state["conv1_1.weight"].clone()
                cur[:, :3, :, :] = tv_w
                cur[:, 3:, :, :] = 0.0
                new_state["conv1_1.weight"] = cur
                new_state["conv1_1.bias"] = weights[b_key].clone()
            else:
                new_state[f"{name}.weight"] = tv_w.clone()
                new_state[f"{name}.bias"] = weights[b_key].clone()
        self.encoder.load_state_dict(new_state)


__all__ = ["MSINetV2", "VGG16Encoder4ch"]
