"""Binomial split-half reliability floor for the temporal windows."""
from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parent.parent
WORK_W = 256
EPS = 2.2204e-16


def block_sum(F, f):
    if f <= 1:
        return F
    H, W = F.shape
    ys = np.arange(0, H, f); xs = np.arange(0, W, f)
    return np.add.reduceat(np.add.reduceat(F, ys, axis=0), xs, axis=1)


def cc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (math.sqrt((a * a).sum() * (b * b).sum()) + EPS))


def dens(F):
    f = max(1, round(F.shape[1] / WORK_W))
    Fs = block_sum(F, f)
    s = gaussian_filter(Fs.astype(np.float64), 0.03 * Fs.shape[1]); m = s.max()
    return s / m if m > 1e-8 else s, Fs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-root", required=True)
    args = ap.parse_args()
    locs = Path(args.public_root) / "ETOCVD_ALLFIXATIONLOCS"
    rng = np.random.default_rng(0)

    def load(sub, win, stem):
        g = glob.glob(str(locs / sub / win / f"{stem}.*"))
        return np.asarray(Image.open(g[0]).convert("L"), dtype=np.int64) if g else None

    normd = locs / "fixationLocsNoCVD" / "full"
    stems = sorted(p.stem for p in normd.iterdir()
                   if p.suffix.lower() in (".jpeg", ".jpg", ".png", ".bmp"))
    rows = []
    for t, cvd in (("d", "fixationLocsCVD_D"), ("p", "fixationLocsCVD_P")):
        for stem in stems:
            nf = load("fixationLocsNoCVD", "full", stem)
            ne = load("fixationLocsNoCVD", "excluding_200ms", stem)
            cf = load(cvd, "full", stem)
            ce = load(cvd, "excluding_200ms", stem)
            if any(x is None for x in (nf, ne, cf, ce)):
                continue
            for win in ("early", "late"):
                N = np.clip(nf - ne, 0, None) if win == "early" else ne
                C = np.clip(cf - ce, 0, None) if win == "early" else ce
                Nd, NFs = dens(N); Cd, _ = dens(C)

                A = rng.binomial(NFs, 0.5); B = NFs - A
                fa = gaussian_filter(A.astype(np.float64), 0.03 * A.shape[1])
                fb = gaussian_filter(B.astype(np.float64), 0.03 * B.shape[1])
                rows.append({"type": t, "window": win, "stem": stem,
                             "cc_normal_cvd": cc(Nd, Cd),
                             "splithalf_floor": cc(fa, fb)})
    out = REPO / "paper" / "tables" / "raw" / "temporal_noise_floor.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["type", "window", "stem",
                                          "cc_normal_cvd", "splithalf_floor"])
        w.writeheader(); w.writerows(rows)
    print("window  type  normal-CVD CC   split-half floor (raw points)")
    for t in ("d", "p"):
        for win in ("early", "late"):
            rr = [r for r in rows if r["type"] == t and r["window"] == win]
            nc = np.mean([r["cc_normal_cvd"] for r in rr])
            fl = np.mean([r["splithalf_floor"] for r in rr])
            print(f"  {win:5s} {t.upper()}   {nc:.3f}           {fl:.3f}")
    print(f"raw: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
