"""DINOv3 saliency model variant (exploratory)."""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from torch import nn

from sm_cvd.models.dinov2_saliency import LoRALinear, SaliencyHead
from sm_cvd.models.msinet import normalize_minmax


def _ensure_torch_amp_shim() -> None:
    """(documentation omitted in the English release)."""
    import torch.amp as _amp
    if not hasattr(_amp, "custom_fwd"):
        try:
            from torch.cuda.amp import custom_fwd, custom_bwd
            _amp.custom_fwd = custom_fwd
            _amp.custom_bwd = custom_bwd
        except Exception:
            pass


def _load_dinov3_backbone(repo_dir: str, entrypoint: str, weights: str):
    """(documentation omitted in the English release)."""
    _ensure_torch_amp_shim()
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from dinov3.hub import backbones as bb
    fn = getattr(bb, entrypoint, None)
    if fn is None:
        raise AttributeError(
            f"entrypoint '{entrypoint}' not found in dinov3.hub.backbones. "
            f"available: {[n for n in dir(bb) if n.startswith('dinov3')]}"
        )
    return fn(weights=weights)


def _inject_lora(backbone: nn.Module, r: int, alpha: float) -> int:
    """(documentation omitted in the English release)."""
    n = 0
    for mod in backbone.modules():
        attn = getattr(mod, "attn", None)
        if attn is None:
            continue
        qkv = getattr(attn, "qkv", None)
        if isinstance(qkv, nn.Linear):
            attn.qkv = LoRALinear(qkv, r=r, alpha=alpha)
            n += 1
    return n


class DINOv3Saliency(nn.Module):
    """(documentation omitted in the English release)."""

    PATCH = 16

    def __init__(self, repo_dir: str, weights: str,
                 entrypoint: str = "dinov3_vitb16",
                 lora_rank: int = 8, lora_alpha: float = 16.0,
                 proc_hw: tuple[int, int] = (288, 384)) -> None:
        super().__init__()
        assert proc_hw[0] % self.PATCH == 0 and proc_hw[1] % self.PATCH == 0, \
            f"proc_hw must be multiple of {self.PATCH}, got {proc_hw}"
        self.proc_hw = tuple(proc_hw)
        self.backbone = _load_dinov3_backbone(repo_dir, entrypoint, weights)

        embed_dim = getattr(self.backbone, "embed_dim", None)
        if embed_dim is None:
            embed_dim = getattr(getattr(self.backbone, "norm", None), "normalized_shape",
                                [768])[0]
        self.embed_dim = embed_dim

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.lora_rank = lora_rank
        if lora_rank > 0:
            n = _inject_lora(self.backbone, lora_rank, lora_alpha)
            print(f"[DINOv3Saliency] LoRA injected into {n} attention qkv layers")

        self.head = SaliencyHead(in_dim=embed_dim, use_film=False)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _patch_grid(self, x: torch.Tensor, B: int) -> torch.Tensor:
        gh, gw = self.proc_hw[0] // self.PATCH, self.proc_hw[1] // self.PATCH
        out = self.backbone.forward_features(x)
        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            feats = out["x_norm_patchtokens"]                       # (B, N, C)
        elif isinstance(out, dict) and "x_prenorm" in out:

            feats = out["x_prenorm"][:, -gh * gw:, :]
        else:
            t = out if torch.is_tensor(out) else out["last_hidden_state"]
            feats = t[:, -gh * gw:, :]
        return feats.transpose(1, 2).reshape(B, -1, gh, gw)         # (B, C, gh, gw)

    def forward(self, image: torch.Tensor,
                confusion: torch.Tensor | None = None,
                image_cvd: torch.Tensor | None = None) -> torch.Tensor:
        """(documentation omitted in the English release)."""
        B, _, H, W = image.shape
        x = image
        if (H, W) != self.proc_hw:
            x = F.interpolate(x, size=self.proc_hw, mode="bilinear", align_corners=False)
        feats = self._patch_grid(x, B)
        out = self.head(feats, out_hw=(H, W))
        return normalize_minmax(out)


__all__ = ["DINOv3Saliency"]
