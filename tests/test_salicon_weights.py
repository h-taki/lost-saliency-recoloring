"""Check that the SALICON-pretrained weights load into MSI-Net and run."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from sm_cvd.models import MSINet

WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "weights" / "msinet_salicon_pytorch.pth"


@pytest.mark.skipif(
    not WEIGHTS_PATH.exists(),
    reason=f"SALICON weights not generated: {WEIGHTS_PATH}. "
           "run `python scripts/convert_hf_weights.py` first",
)
def test_salicon_weights_load_into_msinet():
    """(documentation omitted in the English release)."""
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    assert isinstance(state, dict)
    assert len(state) > 0, "state_dict is empty"

    model = MSINet(subtract_imagenet_mean=True)
    missing, unexpected = model.load_state_dict(state, strict=False)



    real_missing = [k for k in missing if not k.endswith("_mean")]
    assert not real_missing, f"unmapped parameters: {real_missing}"
    assert not unexpected, f"extra keys (mapping error?): {unexpected}"


@pytest.mark.skipif(
    not WEIGHTS_PATH.exists(),
    reason=f"SALICON weights not generated: {WEIGHTS_PATH}",
)
def test_salicon_inference_smoke():
    """(documentation omitted in the English release)."""
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model = MSINet(subtract_imagenet_mean=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    x = torch.rand(1, 3, 240, 320) * 255.0
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 240, 320), y.shape

    assert y.min().item() >= -1e-6
    assert y.max().item() <= 1.0 + 1e-6

    assert y.std().item() > 1e-4, "output is nearly constant - weights may not be effective"
