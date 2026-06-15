"""Dump per-fold out-of-fold dichromat saliency predictions (S_cvd)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sm_cvd.data.transforms import resize_and_pad, image_to_tensor_imagenet  # noqa
from sm_cvd.lightning_modules import DINOv2SaliencyLightning  # noqa

ROOTS = {
    "d": ("/PATH/TO/EToCVD/D", "runs/dinov2_lora_d"),
    "p": ("/PATH/TO/EToCVD/P", "runs/dinov2_lora_p"),
}


def find_ckpt(run: Path, fold: int):
    p = run / f"fold_{fold}" / "lightning_logs"
    vs = sorted([d for d in p.iterdir() if d.name.startswith("version_")],
                key=lambda d: int(d.name.split("_")[1]))
    cks = sorted((vs[-1] / "checkpoints").glob("best-*.ckpt")) or \
        sorted((vs[-1] / "checkpoints").glob("*.ckpt"))
    return cks[-1] if cks else None


@torch.no_grad()
def predict(model, img_path, hw, device):
    pil = resize_and_pad(Image.open(img_path).convert("RGB"), hw, is_image=True)
    t = image_to_tensor_imagenet(pil).unsqueeze(0).to(device)
    p = model(t).squeeze().cpu().numpy()
    lo, hi = p.min(), p.max()
    p = (p - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(p)
    return (p * 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["d", "p"], required=True)
    ap.add_argument("--image-size", type=int, nargs=2, default=[240, 320])
    ap.add_argument("--run", default=None,
                    help="winner run dir override (e.g. runs/gtsigma3_dinov2_d)")
    ap.add_argument("--dataset-root", default=None,
                    help="root override when using the new-GT splits (e.g. data/etocvd_d_sigma3p0)")
    ap.add_argument("--out", default=None, help="output dir override")
    args = ap.parse_args()
    root, run = ROOTS[args.type]
    root = Path(args.dataset_root) if args.dataset_root else Path(root)
    run = REPO / (args.run if args.run else run)
    hw = tuple(args.image_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out) if args.out else REPO / "data" / f"etocvd_{args.type}_scvd_perfold"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for fold in range(1, 6):
        ck = find_ckpt(run, fold)
        if ck is None:
            print(f"[skip] fold_{fold}: no ckpt"); continue
        model = DINOv2SaliencyLightning.load_from_checkpoint(ck, map_location="cpu").to(device).eval()
        val_img_dir = root / "splits_fold5" / f"fold_{fold}" / "val" / "images"
        imgs = sorted(p for p in val_img_dir.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        for ip in imgs:
            sal = predict(model, ip, hw, device)
            Image.fromarray(sal, mode="L").save(out_dir / f"{ip.stem}.png")
            n += 1
        print(f"fold_{fold}: {len(imgs)} val imgs predicted")
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"\ntotal {n} leak-free S_cvd -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
