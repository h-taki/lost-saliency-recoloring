"""Demonstrate the fixed-pixel vs one-degree smoothing confound on external datasets with proxy predictors."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

for _o, _n in (("string_", "bytes_"), ("unicode_", "str_"), ("bool8", "bool_"),
               ("float_", "float64"), ("int_", "int64")):
    if not hasattr(np, _o) and hasattr(np, _n):
        setattr(np, _o, getattr(np, _n))

from scipy.ndimage import gaussian_filter  # noqa
from PIL import Image  # noqa

REPO = Path(__file__).resolve().parent.parent
WG = 256
EPS = 2.2204e-16
PRED_SMOOTH, PRED_SHARP, PROPER, FIXED_PX = 0.06, 0.02, 0.03, 30.0


def cc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (math.sqrt((a * a).sum() * (b * b).sum()) + EPS))


def mapof(pts, sig):
    g = np.zeros((WG, WG))
    if len(pts) == 0:
        return g
    yi = np.clip((pts[:, 0] * WG).astype(int), 0, WG - 1)
    xi = np.clip((pts[:, 1] * WG).astype(int), 0, WG - 1)
    np.add.at(g, (yi, xi), 1.0)
    s = gaussian_filter(g, sig * WG); m = s.max()
    return s / m if m > 1e-8 else g


def boot(d, nb=10000, seed=0):
    rng = np.random.default_rng(seed); a = np.asarray(d)
    b = a[rng.integers(0, len(a), size=(nb, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def iter_toronto(loc):
    import pysaliency
    stim, fix = pysaliency.get_toronto(location=loc)
    xs, ys, ns = np.array(fix.x), np.array(fix.y), np.array(fix.n)
    for i in range(len(stim.stimuli)):
        h, w = stim.stimuli[i].shape[:2]
        m = ns == i
        if m.sum() < 8:
            continue
        yield np.stack([ys[m] / h, xs[m] / w], 1)


def iter_etto(locs_dir):
    d = Path(locs_dir)
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        F = np.asarray(Image.open(p).convert("L"))
        ys, xs = np.nonzero(F)
        if len(ys) < 8:
            continue
        H, W = F.shape
        yield np.stack([ys / H, xs / W], 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["toronto", "etto"], required=True)
    ap.add_argument("--toronto-loc", default="/tmp/sal2")
    ap.add_argument("--etto-locs", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    src = iter_toronto(args.toronto_loc) if args.dataset == "toronto" \
        else iter_etto(args.etto_locs)

    rows = []
    for P in src:
        Wi = rng.uniform(200, 700); miscal = FIXED_PX / Wi
        idx = rng.permutation(len(P)); half = len(P) // 2
        tr, te = P[idx[:half]], P[idx[half:]]
        ps, ph = mapof(tr, PRED_SMOOTH), mapof(tr, PRED_SHARP)
        for gtn, sig in (("miscal", miscal), ("proper", PROPER)):
            gt = mapof(te, sig)
            rows.append({"gt": gtn, "cc_smooth": cc(ps, gt), "cc_sharp": cc(ph, gt),
                         "miscal_pct": round(miscal * 100, 2)})
    name = "toronto" if args.dataset == "toronto" else "ETTO"
    out = REPO / "paper" / "tables" / "raw" / f"sigma_confound_{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["gt", "cc_smooth", "cc_sharp", "miscal_pct"])
        w.writeheader(); w.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault(r["gt"], []).append(r)
    n = len(by["miscal"])
    print(f"{name}: n={n}, mean miscal sigma {np.mean([r['miscal_pct'] for r in rows]):.1f}% (proper=3%)")
    adv = {}
    for gt in ("miscal", "proper"):
        sm = [r["cc_smooth"] for r in by[gt]]; sh = [r["cc_sharp"] for r in by[gt]]
        a = [x - y for x, y in zip(sm, sh)]; m, lo, hi = boot(a); adv[gt] = a
        print(f"  {gt:7s}: smooth {np.mean(sm):.3f} sharp {np.mean(sh):.3f} | "
              f"smooth-adv {m:+.4f} [{lo:+.4f},{hi:+.4f}]")
    inter = [m - p for m, p in zip(adv["miscal"], adv["proper"])]
    mi, lo, hi = boot(inter)
    print(f"  interaction (miscal_adv - proper_adv) = {mi:+.4f} [{lo:+.4f},{hi:+.4f}] (CI excl 0 = confound)")
    print(f"raw: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
