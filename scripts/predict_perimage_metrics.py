"""Compute per-image saliency metrics (CC/SIM/KLD/NSS)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from sm_cvd.data.dataset import EToCVDDataset  # noqa
from sm_cvd.lightning_modules import MSINetLightning, DINOv2SaliencyLightning  # noqa
from sm_cvd.metrics.saliency import cc, sim, kld, nss  # noqa

DATASET_ROOT = {"d": REPO / "data" / "etocvd_d_sigma3p0",
                "p": REPO / "data" / "etocvd_p_sigma3p0"}
SEEDS = {42: "", 21: "_seed21", 84: "_seed84"}
# model -> (run prefix, lightning class, preprocess_mode)
MODELS = {
    "baseline": ("gtsigma3_baseline", MSINetLightning, "msinet"),
    "winner": ("gtsigma3_dinov2", DINOv2SaliencyLightning, "transalnet"),
}


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[perimage] device={device}")
    out = REPO / "paper" / "tables" / "raw" / "saliency_perimage_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in ("d", "p"):
        for mkey, (prefix, cls, pre) in MODELS.items():
            for seed, suf in SEEDS.items():
                run = REPO / "runs" / f"{prefix}_{t}{suf}"
                for fold in range(1, 6):
                    ck = find_ckpt(run, fold)
                    if ck is None:
                        print(f"[skip] {run.name} fold{fold}: no ckpt"); continue
                    model = cls.load_from_checkpoint(ck, map_location="cpu").to(device).eval()
                    val_dir = DATASET_ROOT[t] / "splits_fold5" / f"fold_{fold}" / "val"
                    ds = EToCVDDataset(val_dir, image_size=(240, 320),
                                       preprocess_mode=pre, return_fixation=True)
                    for i in range(len(ds)):
                        s = ds[i]
                        img = s["image"].unsqueeze(0).to(device)
                        gt = s["saliency"].unsqueeze(0).to(device)
                        fix = s["fixation"].unsqueeze(0).to(device)
                        pr = model(img)
                        if pr.dim() == 3:
                            pr = pr.unsqueeze(0)
                        stem = Path(ds._img_files[i]).stem
                        rows.append({"type": t, "model": mkey, "seed": seed, "fold": fold,
                                     "stem": stem,
                                     "cc": float(cc(gt, pr)), "sim": float(sim(gt, pr)),
                                     "kld": float(kld(gt, pr)), "nss": float(nss(fix, pr))})
                    model.to("cpu")
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    print(f"  {t} {mkey} seed{seed} fold{fold}: {len(ds)} imgs")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["type", "model", "seed", "fold", "stem",
                                          "cc", "sim", "kld", "nss"])
        w.writeheader(); w.writerows(rows)
    print(f"\n[perimage] {len(rows)} rows -> {out}\nNext: scripts/analyze_imagelevel_saliency.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
