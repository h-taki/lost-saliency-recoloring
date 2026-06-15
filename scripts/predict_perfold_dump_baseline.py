"""Dump per-fold out-of-fold predictions for the generic baseline (MSI-Net)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from sm_cvd.data.dataset import EToCVDDataset  # noqa
from sm_cvd.lightning_modules import MSINetLightning  # noqa

DATASET_ROOT = {"d": REPO / "data" / "etocvd_d_sigma3p0",
                "p": REPO / "data" / "etocvd_p_sigma3p0"}
RUN = {"d": "runs/gtsigma3_baseline_d", "p": "runs/gtsigma3_baseline_p"}
HW = (240, 320)


def find_ckpt(run: Path, fold: int):
    p = run / f"fold_{fold}" / "lightning_logs"
    if not p.exists():
        return None
    vs = sorted([d for d in p.iterdir() if d.name.startswith("version_")],
                key=lambda d: int(d.name.split("_")[1]))
    for v in reversed(vs):
        cks = sorted((v / "checkpoints").glob("best-*.ckpt")) or \
            sorted((v / "checkpoints").glob("*.ckpt"))
        if cks:
            return cks[-1]
    return None


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["d", "p"], required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t = args.type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = REPO / (args.run if args.run else RUN[t])
    droot = Path(args.dataset_root) if args.dataset_root else DATASET_ROOT[t]
    out = Path(args.out) if args.out else REPO / "data" / f"etocvd_{t}_scvd_perfold_baseline"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for fold in range(1, 6):
        ck = find_ckpt(run, fold)
        if ck is None:
            print(f"[skip] fold{fold}: no ckpt"); continue
        model = MSINetLightning.load_from_checkpoint(ck, map_location="cpu").to(device).eval()
        ds = EToCVDDataset(droot / "splits_fold5" / f"fold_{fold}" / "val",
                           image_size=HW, preprocess_mode="msinet", return_fixation=False)
        for i in range(len(ds)):
            img = ds[i]["image"].unsqueeze(0).to(device)
            pr = model(img).squeeze().cpu().numpy()
            lo, hi = pr.min(), pr.max()
            pr = (pr - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(pr)
            stem = Path(ds._img_files[i]).stem
            Image.fromarray((pr * 255).astype(np.uint8), mode="L").save(out / f"{stem}.png")
            n += 1
        model.to("cpu")
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  fold{fold}: {len(ds)} dumped")
    print(f"[baseline-OOF] type={t} dumped {n} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
