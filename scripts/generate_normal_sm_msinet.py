"""Generate predicted normal-vision saliency maps with MSI-Net (for the fully-predicted-M deployment)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sm_cvd.lightning_modules import MSINetLightning  # noqa: E402
from sm_cvd.data.dataset import resize_and_pad, image_to_tensor  # noqa: E402


def collect_unique_images(dataset_root: Path, splits_folder: str,
                          max_fold: int = 20) -> dict[str, Path]:
    """(documentation omitted in the English release)."""
    out: dict[str, Path] = {}
    for k in range(1, max_fold + 1):
        for split in ("train", "val"):
            img_dir = dataset_root / splits_folder / f"fold_{k}" / split / "images"
            if not img_dir.exists():
                continue
            for f in img_dir.iterdir():
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    if f.stem not in out:
                        out[f.stem] = f
    return out


@torch.no_grad()
def predict_saliency(model, image_pil: Image.Image,
                     image_size: tuple[int, int], device: str) -> Image.Image:
    img = image_pil.convert("RGB")
    img = resize_and_pad(img, image_size, is_image=True)
    t = image_to_tensor(img).unsqueeze(0).to(device)
    pred = model(t).squeeze().cpu().numpy()
    lo, hi = pred.min(), pred.max()
    if hi - lo > 1e-12:
        pred = (pred - lo) / (hi - lo)
    else:
        pred = np.zeros_like(pred)
    return Image.fromarray(np.asarray(pred * 255.0, dtype=np.uint8), mode="L")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path,
                   default=Path("/PATH/TO/EToCVD/D"))
    p.add_argument("--splits-folder", default="splits_30_original")
    p.add_argument("--out-dir", type=Path,
                   default=Path("data/etocvd_d_normal_sm_msinet"))
    p.add_argument("--pretrained-path", type=Path,
                   default=Path("weights/msinet_salicon_pytorch.pth"),
                   help="MSI-Net ckpt (trained on SALICON, disjoint from EToCVD)")
    p.add_argument("--image-size", type=int, nargs=2, default=[240, 320])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()



    print(f"Loading MSINet (SALICON) from {args.pretrained_path}")
    model = MSINetLightning(
        learning_rate=1e-5, loss_variant="method3",
        pretrained_vgg16=False,
        pretrained_path=str(args.pretrained_path),
    ).to(args.device).eval()

    images = collect_unique_images(args.dataset_root, args.splits_folder)
    print(f"unique EToCVD deutan images: {len(images)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_new, n_skip = 0, 0
    for i, (stem, src) in enumerate(sorted(images.items())):
        out_path = args.out_dir / f"{stem}.png"
        if out_path.exists():
            n_skip += 1
            continue
        with Image.open(src) as im:
            sm = predict_saliency(model, im,
                                  image_size=tuple(args.image_size),
                                  device=args.device)
        sm.save(out_path)
        n_new += 1
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(images)}: {stem}.png")
    print(f"\n>> done. new: {n_new}, skipped: {n_skip}, out_dir: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
