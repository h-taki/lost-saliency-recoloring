"""Extract CC/SIM/KLD/NSS from per-run metrics CSV files."""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parent.parent


RUN_NAMES = [
    # MSI-Net
    "msi_d_hf_method3_fold5",            # baseline
    "msi_d_hf_method3_fold5_iter1",
    "msi_d_hf_method3_fold5_iter2",
    "msi_d_hf_method3_fold5_iter3",
    "msi_d_hf_method3_fold5_iter4",
    "msi_d_hf_method3_fold5_iter5",
    "msi_d_hf_method3_fold5_iter6",
    "msi_d_hf_method3_fold5_iter7",
    "msi_d_hf_method3_fold5_iter7_seed21",
    "msi_d_hf_method3_fold5_iter7_seed84",
    # SAM-VGG
    "sam_vgg_d_hf_fold5",
    "sam_vgg_d_hf_fold5_iter1",
    "sam_vgg_d_hf_fold5_iter2",
    "sam_vgg_d_hf_fold5_iter3",
    "sam_vgg_d_hf_fold5_iter4",
    "sam_vgg_d_hf_fold5_iter5",
    "sam_vgg_d_hf_fold5_iter6",
    "sam_vgg_d_hf_fold5_iter7",
    "sam_vgg_d_hf_fold5_iter7_seed21",
    "sam_vgg_d_hf_fold5_iter7_seed84",
    # SAM-ResNet50
    "sam_resnet_d_hf_fold5",
    "sam_resnet_d_hf_fold5_iter1",
    "sam_resnet_d_hf_fold5_iter2",
    "sam_resnet_d_hf_fold5_iter3",
    "sam_resnet_d_hf_fold5_iter4",
    "sam_resnet_d_hf_fold5_iter5",
    "sam_resnet_d_hf_fold5_iter6",
    "sam_resnet_d_hf_fold5_iter7",
    # ★ rigorous Option A strict chain (paper chat #18 revision)
    "sam_resnet_strict_fold5_iter1",
    "sam_resnet_strict_fold5_iter2",
    "sam_resnet_strict_fold5_iter3",
    "sam_resnet_strict_fold5_iter4",
    "sam_resnet_strict_fold5_iter5",
    "sam_resnet_strict_fold5_iter6",
    "sam_resnet_strict_fold5_iter7",
    # TranSalNet
    "transalnet_d_hf_fold5_blocks1",
    "transalnet_d_hf_fold5_blocks2",
    "transalnet_d_hf_fold5_blocks3",
    "transalnet_d_hf_fold5_iter1_blocks1",  # paper chat #18 Table 4 fill (Phase 1)
    "transalnet_d_hf_fold5_iter1_blocks2",
    "transalnet_d_hf_fold5_iter1_blocks3",
    "transalnet_d_hf_fold5_iter2_blocks1",
    "transalnet_d_hf_fold5_iter2_blocks2",
    "transalnet_d_hf_fold5_iter2_blocks3",
    "transalnet_d_hf_fold5_iter3_blocks1",
    "transalnet_d_hf_fold5_iter3_blocks2",
    "transalnet_d_hf_fold5_iter3_blocks3",
    "transalnet_d_hf_fold5_iter4_blocks1",
    "transalnet_d_hf_fold5_iter4_blocks2",
    "transalnet_d_hf_fold5_iter4_blocks3",
    "transalnet_d_hf_fold5_iter5_blocks1",
    "transalnet_d_hf_fold5_iter5_blocks2",
    "transalnet_d_hf_fold5_iter5_blocks3",
    "transalnet_d_hf_fold5_iter6_blocks1",
    "transalnet_d_hf_fold5_iter6_blocks2",
    "transalnet_d_hf_fold5_iter6_blocks3",
    "transalnet_d_hf_fold5_iter7_blocks1",
    "transalnet_d_hf_fold5_iter7_blocks2",
    "transalnet_d_hf_fold5_iter7_blocks3",
    # ★ rigorous Option A strict chain (paper chat #18 revision)
    "transalnet_strict_fold5_iter1_blocks1",
    "transalnet_strict_fold5_iter1_blocks2",
    "transalnet_strict_fold5_iter1_blocks3",
    "transalnet_strict_fold5_iter2_blocks1",
    "transalnet_strict_fold5_iter2_blocks2",
    "transalnet_strict_fold5_iter2_blocks3",
    "transalnet_strict_fold5_iter3_blocks1",
    "transalnet_strict_fold5_iter3_blocks2",
    "transalnet_strict_fold5_iter3_blocks3",
    "transalnet_strict_fold5_iter4_blocks1",
    "transalnet_strict_fold5_iter4_blocks2",
    "transalnet_strict_fold5_iter4_blocks3",
    "transalnet_strict_fold5_iter5_blocks1",
    "transalnet_strict_fold5_iter5_blocks2",
    "transalnet_strict_fold5_iter5_blocks3",
    "transalnet_strict_fold5_iter6_blocks1",
    "transalnet_strict_fold5_iter6_blocks2",
    "transalnet_strict_fold5_iter6_blocks3",
    "transalnet_strict_fold5_iter7_blocks1",
    "transalnet_strict_fold5_iter7_blocks2",
    "transalnet_strict_fold5_iter7_blocks3",

    "decoder_ablation_d3_fold5_baseline",
    "decoder_ablation_d3_fold5_iter1",
    "decoder_ablation_d3_fold5_iter2",
    "decoder_ablation_d3_fold5_iter3",
    "decoder_ablation_d3_fold5_iter4",
    "decoder_ablation_d3_fold5_iter5",
    "decoder_ablation_d3_fold5_iter6",
    "decoder_ablation_d3_fold5_iter7",
    "decoder_ablation_d4_fold5_baseline",
    "decoder_ablation_d4_fold5_iter1",
    "decoder_ablation_d4_fold5_iter2",
    "decoder_ablation_d4_fold5_iter3",
    "decoder_ablation_d4_fold5_iter4",
    "decoder_ablation_d4_fold5_iter5",
    "decoder_ablation_d4_fold5_iter6",
    "decoder_ablation_d4_fold5_iter7",
    # VGG depth
    "vgg_depth_vgg11_fold5_baseline",
    "vgg_depth_vgg13_fold5_baseline",
    "vgg_depth_vgg19_fold5_baseline",

    "msi_p_hf_method3_fold5",
    "msi_p_hf_method3_fold5_iter1",
    "msi_p_hf_method3_fold5_iter2",
    "msi_p_hf_method3_fold5_iter3",
    "msi_p_hf_method3_fold5_iter4",
    "msi_p_hf_method3_fold5_iter5",
    "msi_p_hf_method3_fold5_iter6",
    "msi_p_hf_method3_fold5_iter7",

    "msi_d_hf_method3_fold5_iter7_adamw_wd1e-5",
    "msi_d_hf_method3_fold5_iter7_adamw_wd1e-4",
    "msi_d_hf_method3_fold5_iter7_adamw_wd1e-3",

    "msi_d_hf_method3_fold5_iter5_aug",
    "msi_d_hf_method3_fold5_iter6_aug",
    "msi_d_hf_method3_fold5_iter7_aug",
]


def iter_folds(run_name: str) -> Iterator[tuple[int, Path]]:
    """(documentation omitted in the English release)."""
    run_dir = REPO / "runs" / run_name
    if not run_dir.exists():
        return
    for k in range(1, 21):
        fold_dir = run_dir / f"fold_{k}" / "lightning_logs"
        if not fold_dir.exists():
            continue
        versions = sorted(
            [d for d in fold_dir.iterdir() if d.is_dir() and d.name.startswith("version_")],
            key=lambda d: int(d.name.split("_")[1]),
        )
        if not versions:
            continue
        metrics_csv = versions[-1] / "metrics.csv"
        if metrics_csv.exists():
            yield k, metrics_csv


def best_epoch_metrics(metrics_csv: Path) -> dict | None:
    """(documentation omitted in the English release)."""
    rows = list(csv.DictReader(open(metrics_csv, encoding="utf-8")))

    by_epoch: dict[int, dict[str, float]] = {}
    for row in rows:
        ep_raw = row.get("epoch", "").strip()
        if not ep_raw:
            continue
        try:
            ep = int(ep_raw)
        except ValueError:
            continue
        d = by_epoch.setdefault(ep, {})
        for key in ("val/cc_epoch", "val/sim_epoch", "val/kld_epoch", "val/nss_epoch"):
            v = row.get(key, "").strip()
            if v:
                try:
                    d[key] = float(v)
                except ValueError:
                    pass

    complete = {ep: d for ep, d in by_epoch.items()
                if {"val/cc_epoch", "val/sim_epoch", "val/kld_epoch", "val/nss_epoch"} <= d.keys()}
    if not complete:
        return None
    best_ep = max(complete, key=lambda e: complete[e]["val/cc_epoch"])
    return {
        "best_epoch": best_ep,
        "CC": complete[best_ep]["val/cc_epoch"],
        "SIM": complete[best_ep]["val/sim_epoch"],
        "KLD": complete[best_ep]["val/kld_epoch"],
        "NSS": complete[best_ep]["val/nss_epoch"],
    }


def main() -> int:
    out_csv = REPO / "paper" / "tables" / "raw" / "per_fold_4metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_missing = 0
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt_name", "fold", "best_epoch", "CC", "SIM", "KLD", "NSS"])
        for name in RUN_NAMES:
            n_found = 0
            for fold, metrics_csv in iter_folds(name):
                metrics = best_epoch_metrics(metrics_csv)
                if metrics is None:
                    print(f"  [warn] {name}/fold_{fold}: no complete metrics")
                    n_missing += 1
                    continue
                w.writerow([
                    name, fold, metrics["best_epoch"],
                    f"{metrics['CC']:.6f}",
                    f"{metrics['SIM']:.6f}",
                    f"{metrics['KLD']:.6f}",
                    f"{metrics['NSS']:.6f}",
                ])
                n_total += 1
                n_found += 1
            if n_found > 0:
                print(f"  {name}: {n_found} folds")
            else:
                print(f"  [miss] {name}: no folds found")
                n_missing += 1

    print(f"\n>> total: {n_total} rows, missing: {n_missing}")
    print(f">> saved: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
