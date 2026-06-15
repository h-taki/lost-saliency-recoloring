"""Unit tests for MSINetLoss (method1 and method3 loss configurations)."""
from __future__ import annotations

import math

import pytest
import torch

from sm_cvd.losses import MSINetLoss
from sm_cvd.metrics import cc, kld, sim


# ----------------------------------------------------------------------
# fixture
# ----------------------------------------------------------------------
@pytest.fixture
def maps_identical() -> tuple[torch.Tensor, torch.Tensor]:
    """(documentation omitted in the English release)."""
    torch.manual_seed(0)
    m = torch.rand(2, 1, 32, 32) + 0.1
    return m.clone(), m.clone()


@pytest.fixture
def maps_random() -> tuple[torch.Tensor, torch.Tensor]:
    """(documentation omitted in the English release)."""
    g = torch.Generator().manual_seed(1)
    y_true = torch.rand(2, 1, 32, 32, generator=g) + 0.1
    g2 = torch.Generator().manual_seed(2)
    y_pred = torch.rand(2, 1, 32, 32, generator=g2) + 0.1
    return y_true, y_pred


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def test_method1_is_pure_kld(maps_random):
    y_true, y_pred = maps_random
    loss_fn = MSINetLoss(variant="method1")
    actual = loss_fn(y_pred, y_true)
    expected = kld(y_true, y_pred)
    assert torch.allclose(actual, expected, atol=1e-7)


def test_method1_zero_on_identical_maps(maps_identical):
    y_true, y_pred = maps_identical
    loss_fn = MSINetLoss(variant="method1")
    actual = loss_fn(y_pred, y_true)

    assert actual.item() == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------
# method3: 10·KLD - 2·CC - SIM
# ----------------------------------------------------------------------
def test_method3_decomposition(maps_random):
    """method3(y_pred, y_true) == 10·kld(y_true, y_pred) - 2·cc(...) - sim(...)."""
    y_true, y_pred = maps_random
    loss_fn = MSINetLoss(variant="method3")
    actual = loss_fn(y_pred, y_true)
    expected = (
        10.0 * kld(y_true, y_pred)
        - 2.0 * cc(y_true, y_pred)
        - sim(y_true, y_pred)
    )
    assert torch.allclose(actual, expected, atol=1e-6)


def test_method3_identical_maps_yields_minus_three(maps_identical):
    """(documentation omitted in the English release)."""
    y_true, y_pred = maps_identical
    loss_fn = MSINetLoss(variant="method3")
    actual = loss_fn(y_pred, y_true)
    assert actual.item() == pytest.approx(-3.0, abs=1e-4)


def test_method3_is_lower_for_better_prediction():
    """(documentation omitted in the English release)."""
    torch.manual_seed(0)
    y_true = torch.rand(1, 1, 32, 32) + 0.1
    y_far = torch.rand(1, 1, 32, 32) + 0.1
    y_close = y_true + 0.05 * torch.randn(1, 1, 32, 32)
    y_close = y_close.clamp(min=1e-3)

    loss_fn = MSINetLoss(variant="method3")
    loss_far = loss_fn(y_far, y_true)
    loss_close = loss_fn(y_close, y_true)
    assert loss_close.item() < loss_far.item(), (
        f"a map closer to GT should have lower loss: close={loss_close}, far={loss_far}"
    )


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def test_invalid_variant_raises():
    with pytest.raises(ValueError):
        MSINetLoss(variant="method2")  # type: ignore[arg-type]


def test_method_choices_are_method1_and_method3():
    """(documentation omitted in the English release)."""
    for name in ("method1", "method3"):
        MSINetLoss(variant=name)  # type: ignore[arg-type]


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def test_method3_more_negative_than_method1_when_prediction_matches():
    """(documentation omitted in the English release)."""
    torch.manual_seed(42)
    y_true = torch.rand(1, 1, 32, 32) + 0.1
    y_pred = y_true + 0.01 * torch.randn(1, 1, 32, 32)
    y_pred = y_pred.clamp(min=1e-3)

    l1 = MSINetLoss(variant="method1")(y_pred, y_true)
    l3 = MSINetLoss(variant="method3")(y_pred, y_true)
    assert l3.item() < l1.item()
