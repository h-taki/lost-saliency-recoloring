"""Saliency metrics: CC, SIM, KLD, NSS (PyTorch)."""
from __future__ import annotations

import torch


EPSILON: float = 2.2204e-16


def _spatial_sum(t: torch.Tensor) -> torch.Tensor:
    """(documentation omitted in the English release)."""
    return t.sum(dim=(1, 2, 3), keepdim=True)


def _spatial_mean(t: torch.Tensor) -> torch.Tensor:
    return t.mean(dim=(1, 2, 3), keepdim=True)


def _spatial_min(t: torch.Tensor) -> torch.Tensor:

    return torch.amin(t, dim=(1, 2, 3), keepdim=True)


def _spatial_max(t: torch.Tensor) -> torch.Tensor:
    return torch.amax(t, dim=(1, 2, 3), keepdim=True)


# ----------------------------------------------------------------------
# KLD (Kullback-Leibler Divergence)
# ----------------------------------------------------------------------
def kld(y_true: torch.Tensor, y_pred: torch.Tensor,
        eps: float = EPSILON, reduce: bool = True) -> torch.Tensor:
    """(documentation omitted in the English release)."""
    y_pred = y_pred / (_spatial_sum(y_pred) + eps)
    y_true = y_true / (_spatial_sum(y_true) + eps)

    div = y_true * torch.log(eps + y_true / (y_pred + eps))
    div = div.sum(dim=(1, 2, 3))   # [B]
    return div.mean() if reduce else div


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def cc(y_true: torch.Tensor, y_pred: torch.Tensor,
       eps: float = EPSILON, reduce: bool = True) -> torch.Tensor:
    gt = y_true - _spatial_mean(y_true)
    pr = y_pred - _spatial_mean(y_pred)

    num = (gt * pr).sum(dim=(1, 2, 3))
    den = torch.sqrt(
        (gt ** 2).sum(dim=(1, 2, 3)) * (pr ** 2).sum(dim=(1, 2, 3)) + eps
    )
    out = num / den
    return out.mean() if reduce else out


# ----------------------------------------------------------------------
# SIM (Similarity / Histogram Intersection)
# ----------------------------------------------------------------------
def sim(y_true: torch.Tensor, y_pred: torch.Tensor,
        eps: float = EPSILON, reduce: bool = True) -> torch.Tensor:
    """(documentation omitted in the English release)."""


    min_t = _spatial_min(y_true)
    max_t = _spatial_max(y_true)
    y_true = (y_true - min_t) / (max_t - min_t + eps)

    min_p = _spatial_min(y_pred)
    max_p = _spatial_max(y_pred)
    y_pred = (y_pred - min_p) / (max_p - min_p + eps)


    y_true = y_true / (_spatial_sum(y_true) + eps)
    y_pred = y_pred / (_spatial_sum(y_pred) + eps)


    score = torch.minimum(y_true, y_pred).sum(dim=(1, 2, 3))
    return score.mean() if reduce else score


# ----------------------------------------------------------------------
# NSS (Normalized Scanpath Saliency)
# ----------------------------------------------------------------------
def nss(fix_map: torch.Tensor, pred_map: torch.Tensor,
        eps: float = EPSILON, reduce: bool = True) -> torch.Tensor:
    """(documentation omitted in the English release)."""
    mu = _spatial_mean(pred_map)
    std = torch.sqrt(((pred_map - mu) ** 2).mean(dim=(1, 2, 3), keepdim=True) + eps)
    pred_z = (pred_map - mu) / (std + eps)

    mask = fix_map > 0
    if not mask.any():
        return pred_map.new_zeros(()) if reduce else pred_map.new_zeros(pred_map.shape[0])

    if reduce:
        return pred_z.masked_select(mask).mean()


    out = []
    for i in range(pred_map.shape[0]):
        m = mask[i]
        if m.any():
            out.append(pred_z[i].masked_select(m).mean())
        else:
            out.append(pred_map.new_zeros(()))
    return torch.stack(out)


__all__ = ["cc", "sim", "kld", "nss", "EPSILON"]
