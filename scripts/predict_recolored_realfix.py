"""Evaluate the predictor against the second-session real fixations on recolored images."""
from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from sm_cvd.data.transforms import resize_and_pad, image_to_tensor_imagenet  # noqa
from sm_cvd.lightning_modules import DINOv2SaliencyLightning  # noqa

DATASET_ROOT = {"d": REPO / "data" / "etocvd_d_sigma3p0",
                "p": REPO / "data" / "etocvd_p_sigma3p0"}
RUN = {"d": "runs/gtsigma3_dinov2_d", "p": "runs/gtsigma3_dinov2_p"}
HW = (240, 320)
SIGMA_PCT = 3.0
EPS = 2.2204e-16
VALID = (".png", ".jpg", ".jpeg", ".bmp")


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


def stem_to_fold(dataset_root: Path):
    m = {}
    for fold in range(1, 6):
        vd = dataset_root / "splits_fold5" / f"fold_{fold}" / "val" / "images"
        if vd.exists():
            for p in vd.iterdir():
                if p.suffix.lower() in VALID:
                    m[p.stem] = fold
    return m


def cc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (math.sqrt((a * a).sum() * (b * b).sum()) + EPS))


def sim_m(a, b):
    a = a / (a.sum() + EPS); b = b / (b.sum() + EPS)
    return float(np.minimum(a, b).sum())


def nss(pr, fixbin):
    s = pr.std()
    return float(((pr - pr.mean()) / s)[fixbin > 0].mean()) if s > 1e-8 and fixbin.sum() > 0 else 0.0


def load_locs_density(path, hw=HW):
    F = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    W = F.shape[1]
    sal = gaussian_filter(F, SIGMA_PCT / 100.0 * W)
    sal = np.asarray(Image.fromarray((sal / (sal.max() + 1e-8) * 255).astype(np.uint8)).resize(
        (hw[1], hw[0]), Image.BILINEAR), dtype=np.float64) / 255.0
    fixbin = (np.asarray(Image.fromarray((F > 0).astype(np.uint8) * 255).resize(
        (hw[1], hw[0]), Image.NEAREST)) > 0).astype(np.float64)
    return sal, fixbin


@torch.no_grad()
def predict(model, img_path, device):
    pil = resize_and_pad(Image.open(img_path).convert("RGB"), HW, is_image=True)
    t = image_to_tensor_imagenet(pil).unsqueeze(0).to(device)
    p = model(t).squeeze().cpu().numpy()
    lo, hi = p.min(), p.max()
    return (p - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(p)


def find_one(pattern):
    g = glob.glob(pattern)
    return g[0] if g else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["d", "p"], required=True)
    ap.add_argument("--public-root", required=True,
                    help="public EToCVD dir (containing ETOCVD_ALLSTIMULI / ETOCVD_ALLFIXATIONLOCS)")
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    t = args.type; T = t.upper()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pub = Path(args.public_root)
    run = REPO / (args.run if args.run else RUN[t])
    s2f = stem_to_fold(DATASET_ROOT[t])

    rec_img = pub / "ETOCVD_ALLSTIMULI" / f"imagesRecoloredCVD_{T}"
    orig_img = pub / "ETOCVD_ALLSTIMULI" / "images"
    rec_locs = pub / "ETOCVD_ALLFIXATIONLOCS" / f"fixationLocsRecoloredCVD_{T}" / "full"
    orig_locs = pub / "ETOCVD_ALLFIXATIONLOCS" / f"fixationLocsCVD_{T}" / "full"

    rows = []
    models = {}
    for stem, fold in sorted(s2f.items()):
        if fold not in models:
            ck = find_ckpt(run, fold)
            if ck is None:
                print(f"[skip] fold{fold}: no ckpt"); models[fold] = None; continue
            models[fold] = DINOv2SaliencyLightning.load_from_checkpoint(
                ck, map_location="cpu").to(device).eval()
        model = models[fold]
        if model is None:
            continue

        rp = find_one(str(rec_img / f"{stem}.*_{T}.*"))
        rl = find_one(str(rec_locs / f"{stem}.*_{T}.*")) or find_one(str(rec_locs / f"{stem}.*"))

        op = find_one(str(orig_img / f"{stem}.*"))
        ol = find_one(str(orig_locs / f"{stem}.*"))
        row = {"type": t, "stem": stem, "fold": fold}
        if rp and rl:
            pred = predict(model, rp, device)
            gt, fb = load_locs_density(rl)
            row.update(cc_recol=cc(pred, gt), sim_recol=sim_m(pred, gt), nss_recol=nss(pred, fb))
        if op and ol:
            pred = predict(model, op, device)
            gt, fb = load_locs_density(ol)
            row.update(cc_orig=cc(pred, gt), sim_orig=sim_m(pred, gt), nss_orig=nss(pred, fb))
        if len(row) > 3:
            rows.append(row)
    out = REPO / "paper" / "tables" / "raw" / "realfix_predictor_calibration.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    # append-friendly: write per-type, merge by reading existing
    existing = []
    if out.exists():
        existing = [r for r in csv.DictReader(open(out)) if r.get("type") != t]
    fields = ["type", "stem", "fold", "cc_recol", "sim_recol", "nss_recol",
              "cc_orig", "sim_orig", "nss_orig"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, "") for k in fields})
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    def mean(k):
        v = [r[k] for r in rows if k in r]
        return sum(v) / len(v) if v else float("nan")
    print(f"=== {T} predictor calibration on REAL recolored fixation (n={len(rows)}) ===")
    print(f"  recolored: CC {mean('cc_recol'):.4f} SIM {mean('sim_recol'):.4f} NSS {mean('nss_recol'):.4f}")
    print(f"  original : CC {mean('cc_orig'):.4f} SIM {mean('sim_orig'):.4f} NSS {mean('nss_orig'):.4f}")
    print(f"raw: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
