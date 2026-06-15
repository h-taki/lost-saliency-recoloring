"""Aggregate MSI-Net cross-validation metrics across folds and conditions."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

TYPES = ["c", "d", "p"]
TYPE_LABELS = {"c": "normal", "d": "deutan", "p": "protan"}

CONDITIONS = ["baseline", "hf", "rrc", "re", "hf_rrc", "hf_rrc_re"]
COND_LABELS = {
    "baseline":   "baseline",
    "hf":         "HF",
    "rrc":        "RRC",
    "re":         "RE",
    "hf_rrc":     "HF+RRC",
    "hf_rrc_re":  "HF+RRC+RE",
}

METRICS: List[Tuple[str, Tuple[str, ...], str]] = [
    ("CC",  ("val/cc_epoch",  "val/cc"),  "max"),
    ("SIM", ("val/sim_epoch", "val/sim"), "max"),
    ("KLD", ("val/kld_epoch", "val/kld"), "min"),
    ("NSS", ("val/nss_epoch", "val/nss"), "max"),
]


def find_version_dir(fold_root: Path) -> Path | None:
    candidates = sorted(fold_root.glob("lightning_logs/version_*"))
    if not candidates:
        candidates = sorted(fold_root.glob("**/version_*"))
    return candidates[-1] if candidates else None


def _read_best_from_metrics_csv(csv_path: Path) -> Dict[str, float | None]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    out: Dict[str, float | None] = {}
    for name, candidates, mode in METRICS:
        best: float | None = None
        for tag in candidates:
            if tag not in df.columns:
                continue
            ser = df[tag].dropna()
            if not len(ser):
                continue
            best = float(ser.max()) if mode == "max" else float(ser.min())
            break
        out[name] = best
    return out


def _read_best_from_tb_events(version_dir: Path) -> Dict[str, float | None]:
    try:
        from tensorboard.backend.event_processing.event_accumulator \
            import EventAccumulator
    except ImportError as exc:
        raise RuntimeError("tensorboard is required: pip install tensorboard") from exc
    ea = EventAccumulator(str(version_dir))
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out: Dict[str, float | None] = {}
    for name, candidates, mode in METRICS:
        best: float | None = None
        for tag in candidates:
            if tag not in available:
                continue
            values = [e.value for e in ea.Scalars(tag)]
            if not values:
                continue
            best = max(values) if mode == "max" else min(values)
            break
        out[name] = best
    return out


def read_best_metrics(version_dir: Path) -> Dict[str, float | None]:
    csv_path = version_dir / "metrics.csv"
    if csv_path.exists():
        return _read_best_from_metrics_csv(csv_path)
    return _read_best_from_tb_events(version_dir)


def aggregate_experiment(exp_root: Path, num_folds: int) -> Dict[str, object]:
    per_fold: List[Dict[str, float | None]] = []
    for k in range(1, num_folds + 1):
        fold_root = exp_root / f"fold_{k}"
        vd = find_version_dir(fold_root)
        if vd is None:
            per_fold.append({name: None for name, *_ in METRICS})
            continue
        per_fold.append(read_best_metrics(vd))
    mean: Dict[str, float | None] = {}
    counts: Dict[str, int] = {}
    for name, *_ in METRICS:
        vals = [r[name] for r in per_fold if r[name] is not None]
        mean[name] = (sum(vals) / len(vals)) if vals else None
        counts[name] = len(vals)
    return {"per_fold": per_fold, "mean": mean, "counts": counts}


def _fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "  -   "


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--num-folds", type=int, default=20)
    parser.add_argument("--types", type=str, default=",".join(TYPES),
                        help="comma-separated subset of c/d/p")
    parser.add_argument("--conditions", type=str, default=",".join(CONDITIONS),
                        help="comma-separated subset of baseline/hf/rrc/re/hf_rrc/hf_rrc_re")
    parser.add_argument("--method", choices=["method1", "method2"], default="method1",
                        help="method1 (raw) / method2 (trained on CVD-simulated images)")
    parser.add_argument("--arch", choices=["msi", "transalnet"], default="msi",
                        help="model-name prefix (msi -> msi_<type>_<cond>, "
                             "transalnet → transalnet_<type>_<cond>)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path for the summary CSV (default: runs/<arch>_summary[_method2].csv)")
    parser.add_argument("--include-per-fold", action="store_true")
    args = parser.parse_args()

    selected_types = [t.strip() for t in args.types.split(",") if t.strip()]
    selected_conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    method_suffix = "" if args.method == "method1" else "_method2"

    if args.out is None:
        args.out = REPO_ROOT / "runs" / f"{args.arch}_summary{method_suffix}.csv"

    rows = []
    all_per_fold: Dict[str, List[Dict[str, float | None]]] = {}

    print(f"\n{args.arch} aggregation [num_folds={args.num_folds}]  method={args.method}  "
          f"types={selected_types}  conds={selected_conds}")
    print("-" * 86)
    print(f"{'type':<10} {'condition':<12} | {'CC':>7} | {'SIM':>7} | "
          f"{'KLD':>7} | {'NSS':>7} | n")
    print("-" * 86)

    for t in selected_types:
        type_label = TYPE_LABELS.get(t, t)
        for c in selected_conds:
            exp_name = f"{args.arch}_{t}_{c}{method_suffix}"
            cond_label = COND_LABELS.get(c, c)
            exp_root = args.runs_root / exp_name
            if not exp_root.exists():
                print(f"{type_label:<10} {cond_label:<12} | "
                      f"(missing: {exp_root.name})")
                rows.append({
                    "type": t, "condition": c,
                    "method": args.method, "arch": args.arch,
                    "CC": None, "SIM": None, "KLD": None, "NSS": None, "n": 0,
                })
                all_per_fold[f"{t}/{c}"] = []
                continue
            agg = aggregate_experiment(exp_root, args.num_folds)
            m = agg["mean"]
            cnt = agg["counts"]
            n = min(cnt.values()) if cnt else 0
            print(f"{type_label:<10} {cond_label:<12} | "
                  f"{_fmt(m['CC'])} | {_fmt(m['SIM'])} | "
                  f"{_fmt(m['KLD'])} | {_fmt(m['NSS'])} | {n}")
            rows.append({
                "type": t, "condition": c,
                "method": args.method, "arch": args.arch,
                "CC": m["CC"], "SIM": m["SIM"],
                "KLD": m["KLD"], "NSS": m["NSS"],
                "n": n,
            })
            all_per_fold[f"{t}/{c}"] = agg["per_fold"]
    print("-" * 86)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["arch", "type", "condition", "method",
                        "CC", "SIM", "KLD", "NSS", "n"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f">> summary CSV: {args.out}")

    if args.include_per_fold:
        detail_path = args.out.with_name(args.out.stem + "_per_fold.csv")
        with open(detail_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["arch", "type", "condition", "method", "fold",
                            "CC", "SIM", "KLD", "NSS"],
            )
            w.writeheader()
            for t in selected_types:
                for c in selected_conds:
                    key = f"{t}/{c}"
                    for k, rec in enumerate(all_per_fold.get(key, []), start=1):
                        w.writerow({
                            "arch": args.arch, "type": t, "condition": c,
                            "method": args.method, "fold": k,
                            "CC": rec.get("CC"), "SIM": rec.get("SIM"),
                            "KLD": rec.get("KLD"), "NSS": rec.get("NSS"),
                        })
        print(f">> detail CSV: {detail_path}")


if __name__ == "__main__":
    main()
