"""Regenerate ground-truth saliency maps by smoothing fixations at a chosen sigma (one-degree calibration)."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parent.parent
ROOTS = {
    "d": "/PATH/TO/EToCVD/D",
    "p": "/PATH/TO/EToCVD/P",
}
VALID = (".png", ".jpg", ".jpeg", ".bmp")


def regen_saliency(fix_path: Path, sigma_pct: float) -> Image.Image:
    """(documentation omitted in the English release)."""
    fix = np.asarray(Image.open(fix_path).convert("L"), dtype=np.float32)
    H, W = fix.shape
    sigma = sigma_pct / 100.0 * W
    sal = gaussian_filter(fix, sigma)
    mx = sal.max()
    sal = sal / mx if mx > 1e-8 else sal
    return Image.fromarray((sal * 255).astype(np.uint8), mode="L")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["d", "p"], required=True)
    ap.add_argument("--sigma-pct", type=float, required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[1])
    ap.add_argument("--dataset-root", default=None, help="EToCVD root override (for sandbox)")
    args = ap.parse_args()
    root = Path(args.dataset_root) if args.dataset_root else Path(ROOTS[args.type])
    src_splits = root / "splits_fold5"
    tag = f"sigma{str(args.sigma_pct).replace('.', 'p')}"
    out_root = REPO / "data" / f"etocvd_{args.type}_{tag}" / "splits_fold5"

    n_sal = 0
    for fold in args.folds:
        for split in ("train", "val"):
            sdir = src_splits / f"fold_{fold}" / split
            if not sdir.exists():
                continue
            dst = out_root / f"fold_{fold}" / split

            for kind in ("images", "fixation_maps"):
                (dst / kind).mkdir(parents=True, exist_ok=True)
                for f in (sdir / kind).iterdir():
                    if f.is_file():
                        t = dst / kind / f.name
                        if not t.exists():
                            shutil.copy2(f, t)

            (dst / "saliency_maps").mkdir(parents=True, exist_ok=True)
            fixdir = sdir / "fixation_maps"
            for fx in fixdir.iterdir():
                if not (fx.is_file() and fx.suffix.lower() in VALID):
                    continue
                sal = regen_saliency(fx, args.sigma_pct)
                sal.save(dst / "saliency_maps" / f"{fx.stem}.png")
                n_sal += 1
        print(f"fold_{fold}: regenerated")
    print(f"\nregenerated {n_sal} saliency (σ={args.sigma_pct}% width) -> {out_root}")
    print(f"Step2: retrain baseline+winner on this dataset_root (fold_1) and compare against the old GT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
