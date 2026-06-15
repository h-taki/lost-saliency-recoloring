"""Split-half reliability and matched-N reproducibility of the pooled fixations."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parent.parent
GROUP_DIR = {
    "d": "/PATH/TO/EToCVD/D",
    "p": "/PATH/TO/EToCVD/P",
    "normal": "/PATH/TO/EToCVD/normal",
}
WORK_W = 320
SIGMA_PCT = 3.0
EPS = 2.2204e-16


def block_sum(F: np.ndarray, f: int) -> np.ndarray:
    """(documentation omitted in the English release)."""
    if f <= 1:
        return F
    H, W = F.shape
    ys = np.arange(0, H, f); xs = np.arange(0, W, f)
    return np.add.reduceat(np.add.reduceat(F, ys, axis=0), xs, axis=1)



def cc(gt, pr):
    g = gt - gt.mean(); p = pr - pr.mean()
    den = np.sqrt((g * g).sum() * (p * p).sum() + EPS)
    return float((g * p).sum() / den)


def sim(gt, pr):
    def norm(x):
        x = (x - x.min()) / (x.max() - x.min() + EPS)
        return x / (x.sum() + EPS)
    g = norm(gt); p = norm(pr)
    return float(np.minimum(g, p).sum())


def kld(gt, pr):
    g = gt / (gt.sum() + EPS); p = pr / (pr.sum() + EPS)
    return float((g * np.log(EPS + g / (p + EPS))).sum())


def nss(fix_bin, pr):
    mu = pr.mean(); sd = np.sqrt(((pr - mu) ** 2).mean() + EPS)
    z = (pr - mu) / (sd + EPS)
    m = fix_bin > 0
    return float(z[m].mean()) if m.any() else 0.0


def boot_ci(vals, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["d", "p", "normal"], required=True)
    ap.add_argument("--n-split", type=int, default=20)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.dataset_root) if args.dataset_root else Path(GROUP_DIR[args.group])
    fixdir = root / "fixation_maps"
    rng = np.random.default_rng(args.seed)
    raw_dir = REPO / "paper" / "tables" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in fixdir.iterdir()
                   if p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"})
    rows = []
    for fp in files:
        Ffull = np.asarray(Image.open(fp).convert("L")).astype(np.int64)
        H0, W0 = Ffull.shape
        factor = max(1, round(W0 / WORK_W))
        F = block_sum(Ffull, factor)
        Hs, Ws = F.shape
        sigma = SIGMA_PCT / 100.0 * Ws
        ccs, sims, klds, nsss = [], [], [], []
        for _ in range(args.n_split):

            FA = rng.binomial(F, 0.5)
            FB = F - FA
            if FA.sum() < 1 or FB.sum() < 1:
                continue
            sA = gaussian_filter(FA.astype(np.float64), sigma)
            sB = gaussian_filter(FB.astype(np.float64), sigma)
            fixB = (FB > 0).astype(np.float64)
            ccs.append(cc(sB, sA))
            sims.append(sim(sB, sA))
            klds.append(kld(sB, sA))
            nsss.append(nss(fixB, sA))
        if not ccs:
            continue
        rows.append({"stem": fp.stem, "W": W0, "H": H0, "fix_mass": int(Ffull.sum()),
                     "fix_nnz": int((Ffull > 0).sum()),
                     "cc": np.mean(ccs), "sim": np.mean(sims),
                     "kld": np.mean(klds), "nss": np.mean(nsss)})

    out = raw_dir / f"ceiling_splithalf_{args.group}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "W", "H", "fix_mass", "fix_nnz",
                                          "cc", "sim", "kld", "nss"])
        w.writeheader(); w.writerows(rows)

    print(f"=== ceiling split-half ({args.group}) n_img={len(rows)} "
          f"n_split={args.n_split} sigma={SIGMA_PCT}%W ===")
    print(f"mean fix_mass={np.mean([r['fix_mass'] for r in rows]):.0f} "
          f"nnz={np.mean([r['fix_nnz'] for r in rows]):.0f}")
    for mt in ("cc", "sim", "kld", "nss"):
        vals = [r[mt] for r in rows]
        lo, hi = boot_ci(vals)
        print(f"  ceiling {mt.upper():4s} = {np.mean(vals):.4f} [{lo:.4f},{hi:.4f}]")
    print(f"\nraw: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
