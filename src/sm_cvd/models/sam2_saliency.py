"""Sam2 saliency."""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from torch import nn

from sm_cvd.models.dinov2_saliency import LoRALinear
from sm_cvd.models.msinet import normalize_minmax


def _inject_lora(module: nn.Module, r: int, alpha: float) -> int:
    n = 0
    for mod in module.modules():
        attn = getattr(mod, "attn", None)
        if attn is None:
            continue
        qkv = getattr(attn, "qkv", None)
        if isinstance(qkv, nn.Linear):
            attn.qkv = LoRALinear(qkv, r=r, alpha=alpha)
            n += 1
    return n


class FPNSaliencyDecoder(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, feat_dims: list[int], proj_dim: int = 128) -> None:
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(c, proj_dim, kernel_size=1) for c in feat_dims])
        self.fuse = nn.Sequential(
            nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(proj_dim, proj_dim // 2, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(proj_dim // 2, 1, kernel_size=3, padding=1)

    def forward(self, feats: list[torch.Tensor], out_hw: tuple[int, int]) -> torch.Tensor:

        target = max(feats, key=lambda f: f.shape[-1] * f.shape[-2]).shape[-2:]
        acc = None
        for proj, f in zip(self.projs, feats):
            p = proj(f)
            if p.shape[-2:] != target:
                p = F.interpolate(p, size=target, mode="bilinear", align_corners=False)
            acc = p if acc is None else acc + p
        x = self.fuse(acc)
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return self.out(x)


def _ensure_amp_shim() -> None:
    import torch.amp as _amp
    if not hasattr(_amp, "custom_fwd"):
        try:
            from torch.cuda.amp import custom_fwd, custom_bwd
            _amp.custom_fwd, _amp.custom_bwd = custom_fwd, custom_bwd
        except Exception:
            pass


class SAM2Saliency(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, config: str, ckpt: str,
                 proc_hw: tuple[int, int] = (512, 512),
                 lora_rank: int = 8, lora_alpha: float = 16.0,
                 proj_dim: int = 128) -> None:
        super().__init__()
        _ensure_amp_shim()
        from sam2.build_sam import build_sam2
        sam2 = build_sam2(config, ckpt, device="cpu")
        self.image_encoder = sam2.image_encoder
        self.proc_hw = tuple(proc_hw)

        for p in self.image_encoder.parameters():
            p.requires_grad = False
        if lora_rank > 0:
            n = _inject_lora(self.image_encoder, lora_rank, lora_alpha)
            print(f"[SAM2Saliency] LoRA injected into {n} attention qkv layers")


        with torch.no_grad():
            dummy = torch.zeros(1, 3, *self.proc_hw)
            feats = self._encode(dummy)
        self.feat_dims = [f.shape[1] for f in feats]
        print(f"[SAM2Saliency] backbone_fpn dims={self.feat_dims}, "
              f"shapes={[tuple(f.shape[-2:]) for f in feats]}")
        self.decoder = FPNSaliencyDecoder(self.feat_dims, proj_dim=proj_dim)

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = self.image_encoder(x)
        if isinstance(out, dict):
            feats = out.get("backbone_fpn")
            if feats is None:
                feats = [out["vision_features"]]
        elif isinstance(out, (list, tuple)):
            feats = list(out)
        else:
            feats = [out]
        return list(feats)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, image: torch.Tensor,
                confusion: torch.Tensor | None = None,
                image_cvd: torch.Tensor | None = None) -> torch.Tensor:
        B, _, H, W = image.shape
        x = image
        if (H, W) != self.proc_hw:
            x = F.interpolate(x, size=self.proc_hw, mode="bilinear", align_corners=False)
        feats = self._encode(x)
        out = self.decoder(feats, out_hw=(H, W))
        return normalize_minmax(out)


__all__ = ["SAM2Saliency", "FPNSaliencyDecoder"]
