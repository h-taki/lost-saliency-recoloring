"""DINOv2 + LoRA saliency model: frozen backbone, low-rank adapters, 3-conv decoder."""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from sm_cvd.models.msinet import normalize_minmax


def _load_dinov2_backbone(backbone_name: str):
    """(documentation omitted in the English release)."""
    try:
        hub_dir = torch.hub.get_dir()
        local_repo = os.path.join(hub_dir, "facebookresearch_dinov2_main")
        if os.path.isdir(local_repo):
            return torch.hub.load(local_repo, backbone_name, source="local")
    except Exception as e:  # noqa: BLE001
        print(f"[dinov2] local hub load failed ({e}); falling back to online")
    return torch.hub.load("facebookresearch/dinov2", backbone_name)


class LoRALinear(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, std=1.0 / math.sqrt(r))
        nn.init.zeros_(self.lora_B.weight)


    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


class FiLMGenerator(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(hidden, channels * 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        self.channels = channels

    def forward(self, conf: torch.Tensor, feat_hw: tuple[int, int]) -> tuple:
        c = F.interpolate(conf, size=feat_hw, mode="bilinear", align_corners=False)
        c = F.relu(self.conv1(c))
        gb = self.conv_out(c)
        dgamma, beta = gb[:, :self.channels], gb[:, self.channels:]
        return dgamma, beta


class SaliencyHead(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, in_dim: int = 768, mid: int = 256, use_film: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, mid, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid, mid // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(mid // 2, 1, kernel_size=3, padding=1)
        self.use_film = use_film
        self.film = FiLMGenerator(mid) if use_film else None

    def forward(self, x: torch.Tensor, out_hw: tuple[int, int],
                conf: torch.Tensor | None = None) -> torch.Tensor:

        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)

        if self.use_film and conf is not None:
            dgamma, beta = self.film(conf, x.shape[-2:])
            x = (1.0 + dgamma) * x + beta
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        x = self.conv3(x)
        return x


class ProgressiveSaliencyHead(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, in_dim: int = 768, mid: int = 256) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_dim, mid, kernel_size=3, padding=1)

        self.up1 = nn.Conv2d(mid, mid * 4, kernel_size=3, padding=1)
        self.up2 = nn.Conv2d(mid, (mid // 2) * 4, kernel_size=3, padding=1)
        self.up3 = nn.Conv2d(mid // 2, (mid // 4) * 4, kernel_size=3, padding=1)
        self.up4 = nn.Conv2d(mid // 4, (mid // 8) * 4, kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(2)
        self.out = nn.Conv2d(mid // 8, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, out_hw: tuple[int, int],
                conf: torch.Tensor | None = None) -> torch.Tensor:
        x = F.relu(self.proj(x))                       # (B,256,18,24)
        x = F.relu(self.ps(self.up1(x)))               # ->(B,256,36,48)
        x = F.relu(self.ps(self.up2(x)))               # ->(B,128,72,96)
        x = F.relu(self.ps(self.up3(x)))               # ->(B,64,144,192)
        x = F.relu(self.ps(self.up4(x)))               # ->(B,32,288,384)
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        x = self.out(x)
        return x


class DINOv2Saliency(nn.Module):
    """(documentation omitted in the English release)."""

    PATCH = 14

    def __init__(self, backbone_name: str = "dinov2_vitb14_reg",
                 lora_rank: int = 8, lora_alpha: float = 16.0,
                 proc_hw: tuple[int, int] = (252, 336),
                 use_film: bool = False, dual_pathway: bool = False,
                 decoder_variant: str = "default") -> None:
        super().__init__()
        assert proc_hw[0] % self.PATCH == 0 and proc_hw[1] % self.PATCH == 0
        self.proc_hw = tuple(proc_hw)
        self.use_film = use_film
        self.decoder_variant = decoder_variant

        self.dual_pathway = dual_pathway

        self.backbone = _load_dinov2_backbone(backbone_name)
        embed_dim = self.backbone.embed_dim


        for p in self.backbone.parameters():
            p.requires_grad = False
        self.lora_rank = lora_rank
        if lora_rank > 0:
            for blk in self.backbone.blocks:
                blk.attn.qkv = LoRALinear(blk.attn.qkv, r=lora_rank, alpha=lora_alpha)



        if dual_pathway:
            self.diff_fuse = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
            nn.init.zeros_(self.diff_fuse.weight)
            nn.init.zeros_(self.diff_fuse.bias)

        if decoder_variant == "progressive":
            if use_film:
                raise ValueError("progressive decoder does not support use_film (D2 uses only one of loss/decoder)")
            self.head = ProgressiveSaliencyHead(in_dim=embed_dim)
        else:
            self.head = SaliencyHead(in_dim=embed_dim, use_film=use_film)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _patch_feats(self, x: torch.Tensor, B: int) -> torch.Tensor:
        if x.shape[-2:] != self.proc_hw:
            x = F.interpolate(x, size=self.proc_hw, mode="bilinear", align_corners=False)
        f = self.backbone.forward_features(x)["x_norm_patchtokens"]  # (B,N,C)
        gh, gw = self.proc_hw[0] // self.PATCH, self.proc_hw[1] // self.PATCH
        return f.transpose(1, 2).reshape(B, -1, gh, gw)             # (B,C,gh,gw)

    def forward(self, image: torch.Tensor,
                confusion: torch.Tensor | None = None,
                image_cvd: torch.Tensor | None = None) -> torch.Tensor:
        """(documentation omitted in the English release)."""
        B, _, H, W = image.shape
        feats = self._patch_feats(image, B)
        if self.dual_pathway and image_cvd is not None:
            feats_cvd = self._patch_feats(image_cvd, B)
            feats = feats + self.diff_fuse(feats - feats_cvd)
        out = self.head(feats, out_hw=(H, W), conf=confusion)        # (B, 1, H, W)
        return normalize_minmax(out)


__all__ = ["DINOv2Saliency", "LoRALinear", "SaliencyHead",
           "ProgressiveSaliencyHead", "FiLMGenerator"]
