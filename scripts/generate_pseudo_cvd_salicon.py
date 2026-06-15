"""Generate pseudo-CVD saliency targets on SALICON for pretraining."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sm_cvd.data import EToCVDDataset
from sm_cvd.lightning_modules import MSINetLightning


def find_best_ckpt(run_dir: Path, fold: int) -> Path:
    fold_dir = run_dir / f"fold_{fold}" / "lightning_logs"
    versions = sorted(
        (d for d in fold_dir.iterdir() if d.is_dir() and d.name.startswith("version_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not versions:
        raise FileNotFoundError(f"No version_* under {fold_dir}")
    ckpts = sorted((versions[-1] / "checkpoints").glob("best-*.ckpt"))
    if not ckpts:

        ckpts = sorted((versions[-1] / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No ckpt in {versions[-1] / 'checkpoints'}")
    return ckpts[0]


@torch.no_grad()
def predict_to_png(model: MSINetLightning, image_path: Path, out_path: Path,
                   image_size: tuple[int, int], device: str) -> None:
    """(documentation omitted in the English release)."""


    from sm_cvd.data.dataset import resize_and_pad, image_to_tensor
    pil = Image.open(image_path).convert("RGB")
    pil = resize_and_pad(pil, image_size, is_image=True)
    t = image_to_tensor(pil).unsqueeze(0).to(device)
    pred = model(t).squeeze().cpu().numpy()
    lo, hi = pred.min(), pred.max()
    if hi - lo > 1e-12:
        pred = (pred - lo) / (hi - lo)
    else:
        pred = np.zeros_like(pred)
    pred_u8 = np.asarray(pred * 255.0, dtype=np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred_u8, mode="L").save(out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cvd-src-root", type=Path,
                   default=Path("data/salicon_2017_d_cvd/processed/single_split/fold_1"),
                   help="parent of the input image directory for the SALICON deutan CVD-converted version")
    p.add_argument("--out-root", type=Path,
                   default=Path("data/salicon_2017_d_cvd/pseudo_cvd_gt"),
                   help="output path for pseudo-CVD GT PNG (<split>/<stem>.png)")
    p.add_argument("--baseline-run", type=Path,
                   default=Path("runs/msi_d_hf_method3"),
                   help="baseline MSI-Net 20-fold run directory")
    p.add_argument("--ckpt-fold", type=int, default=1,
                   help="which fold's baseline ckpt to use (default: fold 1).")
    p.add_argument("--image-size", type=int, nargs=2, default=[240, 320])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = find_best_ckpt(args.baseline_run, args.ckpt_fold)
    print(f"Loading baseline MSINet ckpt: {ckpt}")
    model = MSINetLightning.load_from_checkpoint(ckpt, map_location="cpu").to(args.device).eval()

    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"out_root: {args.out_root.resolve()}\n")

    for split in ("train", "val"):
        src_dir = args.cvd_src_root / split / "images"
        out_split = args.out_root / split
        if not src_dir.exists():
            print(f"SKIP {split}: {src_dir} not found")
            continue
        files = sorted(src_dir.iterdir())
        n_new = 0
        n_skip = 0
        print(f"--- {split}: {len(files)} images ---")
        for i, f in enumerate(files):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            out_path = out_split / f"{f.stem}.png"
            if out_path.exists():
                n_skip += 1
                continue
            predict_to_png(model, f, out_path,
                           image_size=tuple(args.image_size), device=args.device)
            n_new += 1
            if (i + 1) % 500 == 0:
                print(f"  ... {i+1}/{len(files)}")
        print(f"  new: {n_new}, skipped: {n_skip}")

    print(f"\n>> done. out_root: {args.out_root.resolve()}")


if __name__ == "__main__":
    main()
