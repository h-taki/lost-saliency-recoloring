"""Cross-dataset evaluation of the smoothing/resolution calibration confound (Toronto, ETTO)."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from math import erf, sqrt
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sm_cvd.data.dataset import resize_and_pad, image_to_tensor
from sm_cvd.lightning_modules import MSINetLightning, SAMVGGLightning
from sm_cvd.metrics import cc as cc_metric
from sm_cvd.metrics import kld, nss, sim


def list_images(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return sorted([
        f for f in dir_path.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])


@torch.no_grad()
def predict(model, image_path: Path, image_size: tuple[int, int],
            device: str) -> torch.Tensor:
    """(documentation omitted in the English release)."""
    pil = Image.open(image_path).convert("RGB")
    pil = resize_and_pad(pil, image_size, is_image=True)
    t = image_to_tensor(pil).unsqueeze(0).to(device)
    pred = model(t).squeeze().cpu()
    if pred.dim() == 0:
        return pred
    lo, hi = pred.min(), pred.max()
    if hi - lo > 1e-8:
        pred = (pred - lo) / (hi - lo)
    else:
        pred = torch.zeros_like(pred)
    return pred  # (H, W) 0..1


def load_gt(gt_path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    """(documentation omitted in the English release)."""
    pil = Image.open(gt_path).convert("L")
    pil = resize_and_pad(pil, image_size, is_image=False)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.tensor(arr)


def paired_t(diffs: list[float]) -> tuple[float, float, float, float]:
    n = len(diffs)
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    t = m / (sd / math.sqrt(n)) if sd > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))) if sd > 0 else 1.0
    return m, sd, t, p


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="root of an EToCVD-compatible layout (images/, saliency_maps/, fixation_maps/)")
    p.add_argument("--msinet-ckpt", type=Path, required=True,
                   help="MSI-Net iter-7 fold-K best ckpt")
    p.add_argument("--sam-vgg-ckpt", type=Path, required=True,
                   help="SAM-VGG iter-7 fold-K best ckpt")
    p.add_argument("--image-size", type=int, nargs=2, default=[240, 320])
    p.add_argument("--ens-weight", type=float, default=0.5,
                   help="ensemble weight for the MSI side")
    p.add_argument("--out-csv", type=Path,
                   default=Path("runs/cross_dataset_eval/results.csv"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None,
                   help="limit the number of evaluated images (for debugging)")
    args = p.parse_args()

    img_dir = args.dataset_root / "images"
    sal_dir = args.dataset_root / "saliency_maps"
    fix_dir = args.dataset_root / "fixation_maps"
    images = list_images(img_dir)
    if not images:
        raise SystemExit(f"No images in {img_dir}")
    if args.limit:
        images = images[: args.limit]
    print(f"\n=== Cross-dataset evaluation ===")
    print(f"  dataset_root: {args.dataset_root}")
    print(f"  images: {len(images)}")
    has_sal = sal_dir.exists()
    has_fix = fix_dir.exists()
    print(f"  saliency_maps available: {has_sal}")
    print(f"  fixation_maps available: {has_fix}")
    print(f"  device: {args.device}")
    print(f"  ens weight: {args.ens_weight}\n")

    print(f"  Loading MSI-Net iter-7: {args.msinet_ckpt}")
    msi = MSINetLightning.load_from_checkpoint(args.msinet_ckpt, map_location="cpu")
    msi.to(args.device).eval()
    print(f"  Loading SAM-VGG iter-7: {args.sam_vgg_ckpt}")
    sam = SAMVGGLightning.load_from_checkpoint(args.sam_vgg_ckpt, map_location="cpu")
    sam.to(args.device).eval()

    rows: list[dict] = []
    H, W = args.image_size
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(images):
        sal_path = sal_dir / f"{img_path.stem}.png" if has_sal else None
        fix_path = fix_dir / f"{img_path.stem}.png" if has_fix else None

        msi_pred = predict(msi, img_path, (H, W), args.device)
        sam_pred = predict(sam, img_path, (H, W), args.device)
        ens_pred = args.ens_weight * msi_pred + (1.0 - args.ens_weight) * sam_pred
        lo, hi = ens_pred.min(), ens_pred.max()
        if hi - lo > 1e-8:
            ens_pred = (ens_pred - lo) / (hi - lo)

        row: dict = {"image": img_path.name}

        if has_sal and sal_path is not None and sal_path.exists():
            sal_gt = load_gt(sal_path, (H, W))
            sal_gt_b = sal_gt.unsqueeze(0).unsqueeze(0)
            for name, pred in [("msi", msi_pred), ("sam", sam_pred), ("ens", ens_pred)]:
                pb = pred.unsqueeze(0).unsqueeze(0)
                row[f"{name}_cc"] = float(cc_metric(sal_gt_b, pb))
                row[f"{name}_sim"] = float(sim(sal_gt_b, pb))
                row[f"{name}_kld"] = float(kld(sal_gt_b, pb))
        if has_fix and fix_path is not None and fix_path.exists():
            fix_gt = load_gt(fix_path, (H, W))
            fix_gt_b = fix_gt.unsqueeze(0).unsqueeze(0)
            for name, pred in [("msi", msi_pred), ("sam", sam_pred), ("ens", ens_pred)]:
                pb = pred.unsqueeze(0).unsqueeze(0)
                row[f"{name}_nss"] = float(nss(fix_gt_b, pb))

        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(images)}")


    if not rows:
        print("No rows.")
        return
    fieldnames = list(rows[0].keys())
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n  out: {args.out_csv}")


    print("\n=== Summary (mean per metric) ===")
    metric_cols = [c for c in fieldnames if c not in {"image"}]
    means = {}
    for col in metric_cols:
        vals = [r[col] for r in rows if col in r]
        if vals:
            means[col] = statistics.mean(vals)
            print(f"  {col:>10}: mean={means[col]:.4f}, "
                  f"n={len(vals)}, min={min(vals):.4f}, max={max(vals):.4f}")

    # paired comparison (ensemble vs single models)
    if "ens_cc" in means and "msi_cc" in means:
        d_msi = [r["ens_cc"] - r["msi_cc"] for r in rows if "ens_cc" in r and "msi_cc" in r]
        d_sam = [r["ens_cc"] - r["sam_cc"] for r in rows if "ens_cc" in r and "sam_cc" in r]
        m1, sd1, t1, p1 = paired_t(d_msi)
        m2, sd2, t2, p2 = paired_t(d_sam)
        print(f"\n=== Paired t-test (Ensemble vs single) ===")
        print(f"  Ens vs MSI: diff {m1:+.4f} +/- {sd1:.4f}, t={t1:.3f}, p~{p1:.4f}")
        print(f"  Ens vs SAM: diff {m2:+.4f} +/- {sd2:.4f}, t={t2:.3f}, p~{p2:.4f}")


    print(f"\n=== Comparison against EToCVD (in-distribution) ===")
    print(f"  EToCVD MSI iter-7: 0.9248 (fold-mean), Cross-dataset: {means.get('msi_cc', 0):.4f}")
    print(f"  EToCVD SAM iter-7: 0.9251 (fold-mean), Cross-dataset: {means.get('sam_cc', 0):.4f}")
    print(f"  EToCVD Ens w=0.5:  0.9404 (fold-mean), Cross-dataset: {means.get('ens_cc', 0):.4f}")
    if "ens_cc" in means:
        drop = 0.9404 - means["ens_cc"]
        print(f"\n  Ensemble drop: {drop:+.4f}")
        if drop < 0.05:
            print(f"  -> true generalization (drop < 0.05)")
        elif drop < 0.15:
            print(f"  -> moderate generalization (drop 0.05-0.15), distribution shift present")
        else:
            print(f"  -> *** sign of EToCVD overfitting (drop > 0.15)")


if __name__ == "__main__":
    main()
