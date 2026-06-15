"""Build early/late temporal-window splits from the time-windowed fixations."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parent.parent
REF_SPLITS = {"d": REPO / "data" / "etocvd_d_sigma3p0",
              "p": REPO / "data" / "etocvd_p_sigma3p0"}
SIGMA_PCT = 3.0
VALID = (".png", ".jpg", ".jpeg", ".bmp")


def find(d: Path, stem: str):
    for e in VALID:
        p = d / f"{stem}{e}"
        if p.exists():
            return p
    # also try stem.jpeg style public naming
    for p in d.glob(f"{stem}.*"):
        return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["d", "p"], required=True)
    ap.add_argument("--public-root", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    t = args.type; T = t.upper()
    pub = Path(args.public_root)
    locs = pub / "ETOCVD_ALLFIXATIONLOCS" / f"fixationLocsCVD_{T}"
    img_dir = pub / "ETOCVD_ALLSTIMULI" / "images"
    ref = REF_SPLITS[t] / "splits_fold5"

    def load_pts(window, stem):
        # window: 'full' or 'excluding_200ms'
        p = find(locs / window, stem)
        if p is None:
            return None
        return np.asarray(Image.open(p).convert("L"), dtype=np.int64)

    n = 0
    for window in ("early", "late"):
        out_root = REPO / "data" / f"etocvd_{t}_{window}" / "splits_fold5"
        for fold in args.folds:
            for split in ("train", "val"):
                ref_img = ref / f"fold_{fold}" / split / "images"
                if not ref_img.exists():
                    continue
                dst = out_root / f"fold_{fold}" / split
                for sub in ("images", "saliency_maps", "fixation_maps"):
                    (dst / sub).mkdir(parents=True, exist_ok=True)
                for ip in ref_img.iterdir():
                    if ip.suffix.lower() not in VALID:
                        continue
                    stem = ip.stem
                    full = load_pts("full", stem)
                    excl = load_pts("excluding_200ms", stem)
                    if full is None or excl is None:
                        continue
                    pts = np.clip(full - excl, 0, None) if window == "early" else excl
                    W = pts.shape[1]
                    sal = gaussian_filter(pts.astype(np.float32), SIGMA_PCT / 100.0 * W)
                    mx = sal.max()
                    sal = (sal / mx * 255).astype(np.uint8) if mx > 1e-8 else sal.astype(np.uint8)
                    fix = ((pts > 0).astype(np.uint8) * 255)
                    # image: copy public stimulus
                    pub_img = find(img_dir, stem)
                    if pub_img is None:
                        continue
                    shutil.copy2(pub_img, dst / "images" / pub_img.name)
                    Image.fromarray(sal, mode="L").save(dst / "saliency_maps" / f"{stem}.png")
                    Image.fromarray(fix, mode="L").save(dst / "fixation_maps" / f"{stem}.png")
                    n += 1
            print(f"  {window} fold_{fold} done")
    print(f"[temporal] type={t} built {n} (image,sal,fix) triplets -> data/etocvd_{t}_{{early,late}}")
    print("Next: train winner/baseline on the early/late dataset_root (fold_1) and compare early vs late")
    return 0


if __name__ == "__main__":
    sys.exit(main())
