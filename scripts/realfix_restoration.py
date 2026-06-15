"""Real-fixation analysis: test whether recoloring moves CVD attention toward normal vision."""
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


def nss(pr, fixbin):
    s = pr.std()
    return float(((pr - pr.mean()) / s)[fixbin > 0].mean()) if s > 1e-8 and fixbin.sum() > 0 else 0.0


def load_density(path):
    """(documentation omitted in the English release)."""
    F = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    f = max(1, round(F.shape[1] / WORK_W))
    Fs = block_sum(F, f)
    s = gaussian_filter(Fs, 0.03 * Fs.shape[1]); m = s.max()
    dens = s / m if m > 1e-8 else s
    return dens, (Fs > 0).astype(np.float64)


def boot(d, nb=10000, seed=0):
    rng = np.random.default_rng(seed); a = np.asarray(d)
    b = a[rng.integers(0, len(a), size=(nb, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-root", required=True)
    args = ap.parse_args()
    pub = Path(args.public_root)
    locs = pub / "ETOCVD_ALLFIXATIONLOCS"

    def f1(sub, stem, suffix=""):
        g = glob.glob(str(locs / sub / f"{stem}{suffix}.*"))
        return g[0] if g else None

    rows = []
    for t, cvd, T in (("d", "CVD_D", "D"), ("p", "CVD_P", "P")):
        normd = locs / "fixationLocsNoCVD" / "full"
        stems = sorted(p.stem for p in normd.iterdir()
                       if p.suffix.lower() in (".jpeg", ".jpg", ".png", ".bmp"))
        for stem in stems:
            np_ = f1("fixationLocsNoCVD/full", stem)
            co = f1(f"fixationLocs{cvd}/full", stem)
            cr = (f1(f"fixationLocsRecolored{cvd}/full", stem, f".jpeg_{T}")
                  or f1(f"fixationLocsRecolored{cvd}/full", stem))
            if not (np_ and co and cr):
                continue
            nd, _ = load_density(np_)
            cod, _ = load_density(co)
            crd, _ = load_density(cr)
            rows.append({"type": t, "stem": stem,
                         "cc_orig": cc(cod, nd), "cc_recol": cc(crd, nd),
                         "nss_orig": nss(nd, (cod > np.percentile(cod, 95))),
                         "nss_recol": nss(nd, (crd > np.percentile(crd, 95)))})
    out = REPO / "paper" / "tables" / "raw" / "realfix_restoration.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["type", "stem", "cc_orig", "cc_recol",
                                          "nss_orig", "nss_recol"])
        w.writeheader(); w.writerows(rows)
    for t in ("d", "p"):
        rr = [r for r in rows if r["type"] == t]
        print(f"== {t.upper()} (n={len(rr)}) restoration (recol - orig) ==")
        for mt in ("cc", "nss"):
            o = [r[f"{mt}_orig"] for r in rr]; rc = [r[f"{mt}_recol"] for r in rr]
            mo, *_ = boot(o); mr, *_ = boot(rc)
            dm, lo, hi = boot([b - a for a, b in zip(o, rc)])
            sig = "(CI excl 0)" if (lo > 0 or hi < 0) else "ns"
            print(f"  {mt.upper()}: orig {mo:.4f} recol {mr:.4f} | restoration {dm:+.4f} [{lo:+.4f},{hi:+.4f}] {sig}")
    print(f"raw: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
