"""Run k-fold cross-validation training."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_with_tee(cmd: list[str], cwd: Path, env: dict, log_path: Path) -> int:
    """(documentation omitted in the English release)."""
    with open(log_path, "w", encoding="utf-8", buffering=1) as logf:

        logf.write("# $ " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
        finally:
            proc.wait()
    return proc.returncode


def load_default_root_dir(config_path: Path) -> Path:
    """(documentation omitted in the English release)."""
    import yaml
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Path(cfg["trainer"]["default_root_dir"])


def best_metric_from_tb_events(version_dir: Path,
                               metric_candidates: tuple[str, ...] = ("val/cc_epoch", "val/cc")
                               ) -> float | None:
    """(documentation omitted in the English release)."""

    csv_path = version_dir / "metrics.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        for col in metric_candidates:
            if col in df.columns:
                ser = df[col].dropna()
                if len(ser):
                    return float(ser.max())


    if not list(version_dir.glob("events.out.tfevents.*")):
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[warn] cannot read event files because tensorboard is missing: pip install tensorboard")
        return None
    ea = EventAccumulator(str(version_dir))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    for t in metric_candidates:
        if t in tags:
            values = [e.value for e in ea.Scalars(t)]
            if values:
                return max(values)
    return None


def find_version_dir(fold_root: Path) -> Path | None:
    """(documentation omitted in the English release)."""
    candidates = sorted(fold_root.glob("lightning_logs/version_*"))
    if not candidates:
        candidates = sorted(fold_root.glob("**/version_*"))
    return candidates[-1] if candidates else None


def summarize(base_root: Path, num_folds: int, start_fold: int = 1) -> None:
    """(documentation omitted in the English release)."""
    rows: list[dict] = []
    for fold in range(start_fold, num_folds + 1):
        fold_root = base_root / f"fold_{fold}"
        vd = find_version_dir(fold_root)
        cc = best_metric_from_tb_events(vd) if vd else None
        rows.append({"fold": fold, "best_val_cc": cc})
        print(f"  fold {fold}: best val/cc = {cc}")

    summary_path = base_root / "kfold_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "best_val_cc"])
        writer.writeheader()
        writer.writerows(rows)

    ccs = [r["best_val_cc"] for r in rows if r["best_val_cc"] is not None]
    if ccs:
        avg = sum(ccs) / len(ccs)
        print(f"\n>> mean best val/cc = {avg:.4f}  (n={len(ccs)})")
    print(f">> {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="LightningCLI config YAML")
    parser.add_argument("--num-folds", type=int, default=20)
    parser.add_argument("--start-fold", type=int, default=1)
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="absolute path of the python used in subprocesses")
    parser.add_argument("--phase", choices=["fit", "test"], default="fit")
    parser.add_argument("--only-summarize", action="store_true",
                        help="do not train; only aggregate from existing events")
    args = parser.parse_args()

    cfg_path: Path = args.config.resolve()
    base_root = load_default_root_dir(cfg_path)

    if args.only_summarize:
        summarize(base_root, args.num_folds, args.start_fold)
        return

    summary_rows: list[dict] = []
    for fold in range(args.start_fold, args.num_folds + 1):
        fold_root = base_root / f"fold_{fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {args.phase} fold {fold}/{args.num_folds} -> {fold_root} ===",
              flush=True)

        cmd = [
            args.python,
            str(REPO_ROOT / "scripts" / "train.py"),
            args.phase,
            "--config", str(cfg_path),
            "--data.fold", str(fold),
            "--trainer.default_root_dir", str(fold_root),
        ]
        print("$", " ".join(cmd), flush=True)

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"




        log_path = fold_root / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        returncode = _run_with_tee(cmd, cwd=REPO_ROOT, env=env, log_path=log_path)
        if returncode != 0:
            print(f"!! fold {fold} failed with code {returncode} "
                  f"(see {log_path})", flush=True)
            continue

        vd = find_version_dir(fold_root)
        best_cc = best_metric_from_tb_events(vd) if vd else None
        summary_rows.append({"fold": fold, "best_val_cc": best_cc})


    summary_path = base_root / "kfold_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "best_val_cc"])
        writer.writeheader()
        writer.writerows(summary_rows)


    ccs = [r["best_val_cc"] for r in summary_rows if r["best_val_cc"] is not None]
    if ccs:
        avg = sum(ccs) / len(ccs)
        print(f"\n>> aggregation: mean best val/cc = {avg:.4f}  (n={len(ccs)})", flush=True)
    print(f">> detail: {summary_path}")


if __name__ == "__main__":
    main()
