"""Hand-computed checks of the saliency metrics (numpy reference vs the PyTorch implementation)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from sm_cvd.metrics.saliency import EPSILON, cc, kld, nss, sim


def _np_kld(y_true: np.ndarray, y_pred: np.ndarray, eps=EPSILON) -> float:
    y_pred = y_pred / (y_pred.sum(axis=(1, 2, 3), keepdims=True) + eps)
    y_true = y_true / (y_true.sum(axis=(1, 2, 3), keepdims=True) + eps)
    div = y_true * np.log(eps + y_true / (y_pred + eps))
    return div.sum(axis=(1, 2, 3)).mean()


def _np_cc(y_true, y_pred, eps=EPSILON) -> float:
    gt = y_true - y_true.mean(axis=(1, 2, 3), keepdims=True)
    pr = y_pred - y_pred.mean(axis=(1, 2, 3), keepdims=True)
    num = (gt * pr).sum(axis=(1, 2, 3))
    den = np.sqrt((gt ** 2).sum(axis=(1, 2, 3)) * (pr ** 2).sum(axis=(1, 2, 3)) + eps)
    return (num / den).mean()


def _np_sim(y_true, y_pred, eps=EPSILON) -> float:
    def _mm(t):
        mn = t.min(axis=(1, 2, 3), keepdims=True)
        mx = t.max(axis=(1, 2, 3), keepdims=True)
        return (t - mn) / (mx - mn + eps)
    y_true = _mm(y_true); y_pred = _mm(y_pred)
    y_true = y_true / (y_true.sum(axis=(1, 2, 3), keepdims=True) + eps)
    y_pred = y_pred / (y_pred.sum(axis=(1, 2, 3), keepdims=True) + eps)
    return np.minimum(y_true, y_pred).sum(axis=(1, 2, 3)).mean()


def test_metrics_match_numpy_reference():
    rng = np.random.RandomState(0)
    y_true = rng.rand(2, 1, 16, 24).astype(np.float32) * 255.0
    y_pred = rng.rand(2, 1, 16, 24).astype(np.float32)
    fix = (rng.rand(2, 1, 16, 24) > 0.95).astype(np.float32) * 255.0

    t_true = torch.from_numpy(y_true.copy())
    t_pred = torch.from_numpy(y_pred.copy())
    t_fix = torch.from_numpy(fix.copy())

    np.testing.assert_allclose(float(kld(t_true, t_pred)), _np_kld(y_true, y_pred), rtol=1e-5)
    np.testing.assert_allclose(float(cc(t_true, t_pred)),  _np_cc(y_true, y_pred),  rtol=1e-5)
    np.testing.assert_allclose(float(sim(t_true, t_pred)), _np_sim(y_true, y_pred), rtol=1e-5)


    mu = y_pred.mean(axis=(1, 2, 3), keepdims=True)
    sd = np.sqrt(((y_pred - mu) ** 2).mean(axis=(1, 2, 3), keepdims=True) + EPSILON)
    z = (y_pred - mu) / (sd + EPSILON)
    expected = z[fix > 0].mean()
    np.testing.assert_allclose(float(nss(t_fix, t_pred)), expected, rtol=1e-5)


if __name__ == "__main__":
    test_metrics_match_numpy_reference()
    print("OK")
