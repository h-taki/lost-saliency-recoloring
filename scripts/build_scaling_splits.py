"""Build training-set-size subsets (20/40/60/80) for the learning-curve check."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = {"d": REPO / "data" / "etocvd_d_sigma3p0",
       "p": REPO / "data" / "etocvd_p_native_sigma3p0"}
VALID = (".png", ".jpg", ".jpeg", ".bmp")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["d", "p"], required=True)
    ap.add_argument("--sizes", nargs="+", type=int, default=[20, 40, 60])
    ap.add_argument("--folds", nargs="+", type=int, default=[1])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import random
    src = SRC[args.type] / "splits_fold5"
    for N in args.sizes:
        out = REPO / "data" / f"etocvd_{args.type}_scale{N}" / "splits_fold5"
        for fold in args.folds:
            tr_img = src / f"fold_{fold}" / "train" / "images"
            if not tr_img.exists():
                print(f"[skip] {tr_img}"); continue
            stems = sorted(p.stem for p in tr_img.iterdir() if p.suffix.lower() in VALID)
            rng = random.Random(args.seed + fold)
            keep = set(rng.sample(stems, min(N, len(stems))))
            # train: keep N stems ; val: copy all
            for split in ("train", "val"):
                for sub in ("images", "saliency_maps", "fixation_maps"):
                    sd = src / f"fold_{fold}" / split / sub
                    dd = out / f"fold_{fold}" / split / sub
                    dd.mkdir(parents=True, exist_ok=True)
                    if not sd.exists():
                        continue
                    for f in sd.iterdir():
                        if f.suffix.lower() not in VALID:
                            continue
                        if split == "train" and f.stem not in keep:
                            continue
                        shutil.copy2(f, dd / f.name)
        print(f"  scale{N} ({args.type}) folds {args.folds} built (train<= {N})")
    print(f"[scaling] type={args.type} sizes={args.sizes} -> data/etocvd_{args.type}_scale*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
