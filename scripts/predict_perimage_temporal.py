"""Per-image metrics on the early/late temporal windows."""
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

SEEDS = {42: "", 21: "_seed21", 84: "_seed84"}
MODELS = {"baseline": (MSINetLightning, "msinet"), "winner": (DINOv2SaliencyLightning, "transalnet")}


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
    print(f"[temporal-perimage] device={device}")
    rows = []
    for t in ("d", "p"):
        for window in ("early", "late"):
            droot = REPO / "data" / f"etocvd_{t}_{window}"
            for mkey, (cls, pre) in MODELS.items():
                for seed, suf in SEEDS.items():
                    run = REPO / "runs" / f"temporal_{mkey}_{t}_{window}{suf}"
                    for fold in range(1, 6):
                        ck = find_ckpt(run, fold)
                        if ck is None:
                            print(f"[skip] {run.name} fold{fold}: no ckpt"); continue
                        model = cls.load_from_checkpoint(ck, map_location="cpu").to(device).eval()
                        vdir = droot / "splits_fold5" / f"fold_{fold}" / "val"
                        ds = EToCVDDataset(vdir, image_size=(240, 320),
                                           preprocess_mode=pre, return_fixation=True)
                        for i in range(len(ds)):
                            s = ds[i]
                            img = s["image"].unsqueeze(0).to(device)
                            gt = s["saliency"].unsqueeze(0).to(device)
                            fix = s["fixation"].unsqueeze(0).to(device)
                            pr = model(img)
                            if pr.dim() == 3:
                                pr = pr.unsqueeze(0)
                            rows.append({"type": t, "window": window, "model": mkey,
                                         "seed": seed, "fold": fold,
                                         "stem": Path(ds._img_files[i]).stem,
                                         "cc": float(cc(gt, pr)), "sim": float(sim(gt, pr)),
                                         "kld": float(kld(gt, pr)), "nss": float(nss(fix, pr))})
                        model.to("cpu")
                        if device == "cuda":
                            torch.cuda.empty_cache()
                    print(f"  {t} {window} {mkey} seed{seed}: done")
    out = REPO / "paper" / "tables" / "raw" / "temporal_perimage_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["type", "window", "model", "seed", "fold",
                                          "stem", "cc", "sim", "kld", "nss"])
        w.writeheader(); w.writerows(rows)
    print(f"\n[temporal-perimage] {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
