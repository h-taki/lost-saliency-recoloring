"""Build native-resolution protan splits (resolution-matched)."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parent.parent
E_P = Path("/PATH/TO/EToCVD/P")
REF_PART = REPO / "data" / "etocvd_p_sigma3p0" / "splits_fold5"
SIGMA_PCT = 3.0
VALID = (".png", ".jpg", ".jpeg", ".bmp")


def find(d: Path, stem: str):
    for e in VALID:
        if (d / f"{stem}{e}").exists():
            return d / f"{stem}{e}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--e-root", default=None, help="/PATH/TO/EToCVD/P")
    args = ap.parse_args()
    e_p = Path(args.e_root) if args.e_root else E_P
    src_img = e_p / "images"
    src_fix = e_p / "fixation_maps"
    out_root = REPO / "data" / "etocvd_p_native_sigma3p0" / "splits_fold5"

    n = 0
    for fold in args.folds:
        for split in ("train", "val"):
            ref = REF_PART / f"fold_{fold}" / split / "images"
            if not ref.exists():
                print(f"[skip] {ref} not found"); continue
            dst = out_root / f"fold_{fold}" / split
            for sub in ("images", "saliency_maps", "fixation_maps"):
                (dst / sub).mkdir(parents=True, exist_ok=True)
            for rp in ref.iterdir():
                if rp.suffix.lower() not in VALID:
                    continue
                stem = rp.stem
                ip = find(src_img, stem)
                fp = find(src_fix, stem)
                if ip is None or fp is None:
                    print(f"[miss] {stem}: native img/fix not found"); continue
                # native image copy
                shutil.copy2(ip, dst / "images" / ip.name)
                # native fixation copy + saliency regen at native sigma=3%W
                fix = np.asarray(Image.open(fp).convert("L"), dtype=np.float32)
                W = fix.shape[1]
                sal = gaussian_filter(fix, SIGMA_PCT / 100.0 * W)
                mx = sal.max()
                sal = (sal / mx * 255).astype(np.uint8) if mx > 1e-8 else sal.astype(np.uint8)
                Image.fromarray(sal, mode="L").save(dst / "saliency_maps" / f"{stem}.png")
                Image.open(fp).convert("L").save(dst / "fixation_maps" / f"{stem}.png")
                n += 1
        print(f"  fold_{fold} done")
    print(f"[native-p] built {n} triplets (native res, sigma={SIGMA_PCT}%W) -> "
          f"data/etocvd_p_native_sigma3p0")
    print("Next: train baseline/winner with dataset_root=data/etocvd_p_native_sigma3p0 and compare type dependence against native D")
    return 0


if __name__ == "__main__":
    sys.exit(main())
