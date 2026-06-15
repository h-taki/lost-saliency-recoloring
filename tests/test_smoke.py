"""Minimal smoke test that the model forward pass runs with correct shapes."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from sm_cvd.models import MSINet


def test_msinet_forward_shape():
    model = MSINet(subtract_imagenet_mean=True)
    model.eval()
    x = torch.rand(2, 3, 240, 320) * 255.0
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 240, 320), y.shape

    assert y.min().item() >= -1e-6
    assert y.max().item() <= 1.0 + 1e-6


if __name__ == "__main__":
    test_msinet_forward_shape()
    print("OK")
