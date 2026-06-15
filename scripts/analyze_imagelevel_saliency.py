"""Image-level statistics (paired bootstrap, mixed-effects) for the saliency metrics."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "paper" / "tables" / "raw" / "saliency_perimage_metrics.csv"
METRICS = ["cc", "sim", "kld", "nss"]
HIGHER = {"cc": True, "sim": True, "kld": False, "nss": True}


def boot_paired(diffs, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(diffs)
    bs = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> int:
    if not SRC.exists():
        print(f"[err] {SRC} not found. Run predict_perimage_metrics.py on GPU first"); return 1
    rows = list(csv.DictReader(open(SRC)))
    # index: (type, model, stem) -> list over seeds
    agg = {}
    for r in rows:
        k = (r["type"], r["model"], r["stem"])
        agg.setdefault(k, {m: [] for m in METRICS})
        for m in METRICS:
            agg[k][m].append(float(r[m]))
    # per-image seed-mean
    img = {}
    for (t, model, stem), d in agg.items():
        img.setdefault((t, stem), {})[model] = {m: np.mean(d[m]) for m in METRICS}

    out = [["type", "metric", "baseline_mean", "winner_mean", "mean_diff",
            "ci_lo", "ci_hi", "per_image_d", "n_img", "higher_is_better", "winner_better"]]
    print("=== image-level winner - baseline (paired bootstrap, n=images) ===")
    for t in ("d", "p"):
        keys = [k for k in img if k[0] == t and "winner" in img[k] and "baseline" in img[k]]
        print(f"\n-- {t.upper()} (n_img={len(keys)}) --")
        for m in METRICS:
            wv = [img[k]["winner"][m] for k in keys]
            bv = [img[k]["baseline"][m] for k in keys]
            diffs = [w - b for w, b in zip(wv, bv)]
            md, lo, hi = boot_paired(diffs)
            sd = np.std(diffs, ddof=1)
            d = md / sd if sd > 0 else 0.0
            better = (md > 0) == HIGHER[m]
            sig = (lo > 0 or hi < 0)
            print(f"  {m.upper():4s} base {np.mean(bv):.4f} win {np.mean(wv):.4f} "
                  f"diff {md:+.4f} [{lo:+.4f},{hi:+.4f}] d={d:+.2f} "
                  f"{'(CI excl 0)' if sig else '(ns)'} {'win+' if better else 'base+'}")
            out.append([t, m, f"{np.mean(bv):.4f}", f"{np.mean(wv):.4f}", f"{md:+.4f}",
                        f"{lo:+.4f}", f"{hi:+.4f}", f"{d:+.2f}", len(keys),
                        HIGHER[m], better])
    op = REPO / "paper" / "tables" / "raw" / "stats_imagelevel_saliency.csv"
    with open(op, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out)
    print(f"\nsaved {op}")

    # optional mixed-effects
    try:
        import statsmodels.formula.api as smf
        import pandas as pd
        df = pd.DataFrame(rows)
        for c in METRICS:
            df[c] = df[c].astype(float)
        print("\n=== mixed-effects (metric ~ model, groups=image, re seed) p-values ===")
        for t in ("d", "p"):
            sub = df[df.type == t].copy()
            sub["img"] = sub["type"] + "_" + sub["stem"]
            for m in METRICS:
                try:
                    md = smf.mixedlm(f"{m} ~ C(model)", sub, groups=sub["img"]).fit(reml=False)
                    coef = md.params.get("C(model)[T.winner]", float("nan"))
                    p = md.pvalues.get("C(model)[T.winner]", float("nan"))
                    print(f"  {t.upper()} {m.upper():4s}: winner coef {coef:+.4f} p={p:.4g}")
                except Exception as e:  # noqa
                    print(f"  {t.upper()} {m.upper()}: mixedlm failed ({e})")
    except Exception:
        print("\n[note] statsmodels not found -> skip mixed-effects (rely on bootstrap CI)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
