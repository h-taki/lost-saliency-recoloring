"""Generate CIELAB a* confusion maps (optional color-opponent input)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = "/PATH/TO/EToCVD/D"
VALID = (".jpg", ".jpeg", ".png", ".bmp")
WIN = 15
OUT_HW = (240, 320)


def local_std(a: np.ndarray, k: int) -> np.ndarray:
    """(documentation omitted in the English release)."""
    from scipy.ndimage import uniform_filter
    mean = uniform_filter(a, size=k, mode="reflect")
    mean2 = uniform_filter(a ** 2, size=k, mode="reflect")
    var = np.clip(mean2 - mean ** 2, 0.0, None)
    return np.sqrt(var)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=str(REPO / "data" / "etocvd_d_confusion"))
    ap.add_argument("--png-out", default=str(REPO / "data" / "etocvd_d_confusion_png"),
                    help="output path for native-resolution PNG (read by the teacher_saliency mechanism; "
                         "spatially aligned via the same resize_and_pad + HF aug as the image)")
    ap.add_argument("--window", type=int, default=WIN)
    ap.add_argument("--viz", type=int, default=2, help="visualize the first N images as PNG")
    args = ap.parse_args()

    img_dir = Path(args.dataset_root) / "images"
    out_dir = Path(args.out)
    png_dir = Path(args.png_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in VALID)
    print(f"images: {len(stems)} -> npy:{out_dir} png:{png_dir}")

    means = []
    for idx, p in enumerate(stems):
        rgb = np.asarray(Image.open(p).convert("RGB"))
        lab = rgb2lab(rgb)                       # H,W,3
        a = lab[:, :, 1].astype(np.float64)
        conf = local_std(a, args.window)
        mx = conf.max()
        conf_n = (conf / mx) if mx > 1e-8 else conf

        Image.fromarray((conf_n * 255).astype(np.uint8), mode="L").save(
            png_dir / f"{p.stem}.png")

        conf_img = Image.fromarray((conf_n * 255).astype(np.uint8)).resize(
            (OUT_HW[1], OUT_HW[0]), Image.BILINEAR)
        conf_final = np.asarray(conf_img, dtype=np.float32) / 255.0
        np.save(out_dir / f"{p.stem}.npy", conf_final)
        means.append(float(conf_final.mean()))
        if idx < args.viz:
            Image.fromarray((conf_final * 255).astype(np.uint8)).save(
                out_dir / f"_viz_{p.stem}.png")

    print(f"written: {len(stems)} npy + {len(stems)} png")
    print(f"image-mean confusion: min={min(means):.4f} mean={sum(means)/len(means):.4f} "
          f"max={max(means):.4f}")
    print(f"viz PNG: {args.viz} images (_viz_*.png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
