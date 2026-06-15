"""Composite saliency loss (KLD, CC, SIM) used to train the saliency models."""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from sm_cvd.metrics.saliency import cc, kld, sim


class MSINetLoss(nn.Module):
    """(documentation omitted in the English release)."""

    def __init__(self, variant: Literal["method1", "method3"] = "method1") -> None:
        super().__init__()
        if variant not in ("method1", "method3"):
            raise ValueError(f"variant must be 'method1' or 'method3', got {variant}")
        self.variant = variant

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.variant == "method1":
            return kld(y_true, y_pred)
        # method3
        return 10.0 * kld(y_true, y_pred) - 2.0 * cc(y_true, y_pred) - sim(y_true, y_pred)
