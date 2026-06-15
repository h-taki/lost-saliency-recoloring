"""Build splits using CVD-simulated input images."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

DEFAULT_ROOT = "/PATH/TO/EToCVD/D"
REPO = Path(__file__).resolve().parent.parent
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def find_cvd(cvd_dir: Path, stem: str) -> Path | None:
    for e in VALID_EXTS:
        p = cvd_dir / f"{stem}_D_100{e}"
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DEFAULT_ROOT)
    ap.add_argument("--out-root", default=str(REPO / "data" / "etocvd_d_cvdinput"))
    ap.add_argument("--paired-out", default=str(REPO / "data" / "etocvd_d_cvd_paired_png"))
    args = ap.parse_args()

    root = Path(args.dataset_root)
    src_splits = root / "splits_fold5"
    cvd_dir = root / "images_for_D"
    out_root = Path(args.out_root) / "splits_fold5"
    paired_out = Path(args.paired_out)
    paired_out.mkdir(parents=True, exist_ok=True)


    n_img = 0
    missing: list[str] = []
    for fold_dir in sorted(src_splits.glob("fold_*")):
        for split in ("train", "val"):
            sdir = fold_dir / split
            if not sdir.exists():
                continue
            dst = out_root / fold_dir.name / split

            for kind in ("saliency_maps", "fixation_maps"):
                (dst / kind).mkdir(parents=True, exist_ok=True)
                for f in (sdir / kind).iterdir():
                    if f.is_file():
                        t = dst / kind / f.name
                        if not t.exists():
                            shutil.copy2(f, t)

            (dst / "images").mkdir(parents=True, exist_ok=True)
            for f in (sdir / "images").iterdir():
                if not (f.is_file() and f.suffix.lower() in VALID_EXTS):
                    continue
                cvd = find_cvd(cvd_dir, f.stem)
                if cvd is None:
                    missing.append(f"{fold_dir.name}/{split}/{f.name}")
                    continue
                t = dst / "images" / f"{f.stem}{cvd.suffix}"
                if not t.exists():
                    shutil.copy2(cvd, t)
                n_img += 1
    print(f"[1] D-cvd splits: {n_img} images -> {out_root}")
    if missing:
        print(f"  [warn] missing I_cvd for: {missing[:5]} ({len(missing)} total)")


    n_png = 0
    for f in sorted(cvd_dir.iterdir()):
        if not (f.is_file() and f.suffix.lower() in VALID_EXTS):
            continue
        stem = f.stem
        if stem.endswith("_D_100"):
            stem = stem[: -len("_D_100")]
        t = paired_out / f"{stem}.png"
        if not t.exists():
            Image.open(f).convert("RGB").save(t)
        n_png += 1
    print(f"[2] paired PNG: {n_png} -> {paired_out}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
