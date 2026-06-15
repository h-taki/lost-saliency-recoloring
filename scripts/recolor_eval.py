"""Evaluate recolored images with predictor-independent CIEDE2000 metrics (naturalness cost and local-contrast distinguishability gain)."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from daltonlens import simulate
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity as ssim

REPO = Path(__file__).resolve().parent.parent
DATA_ROOTS = {"d": "/PATH/TO/EToCVD/D",
              "p": "/PATH/TO/EToCVD/P"}
SCVD = {"d": REPO / "data" / "etocvd_d_scvd_perfold_newgt",
        "p": REPO / "data" / "etocvd_p_scvd_perfold_newgt"}
NORMAL_SM = REPO / "data" / "etocvd_normal_sm_sigma3p0"
DEF = {"d": simulate.Deficiency.DEUTAN, "p": simulate.Deficiency.PROTAN}
HW = (240, 320)
VALID = (".png", ".jpg", ".jpeg", ".bmp")
SIM = simulate.Simulator_Machado2009()
MASK_THR = 0.3


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB").resize((HW[1], HW[0]), Image.BILINEAR),
                      dtype=np.uint8)


def load_w(p):
    return np.asarray(Image.open(p).convert("L").resize((HW[1], HW[0]), Image.BILINEAR),
                      dtype=np.float32) / 255.0


def local_contrast(rgb, defi):
    sim = SIM.simulate_cvd(rgb, defi, 1.0).astype(np.uint8)
    lab = rgb2lab(sim)
    dx = np.zeros(rgb.shape[:2]); dy = np.zeros(rgb.shape[:2])
    dx[:, :-1] = deltaE_ciede2000(lab[:, :-1], lab[:, 1:])
    dy[:-1, :] = deltaE_ciede2000(lab[:-1, :], lab[1:, :])
    return (dx + dy) / 2.0


def boot_ci(vals, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed); a = np.asarray(vals)
    m = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def betai(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lb = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lb) / a
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30: d = 1e-30
    d = 1 / d; h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2)); d = 1 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1 / d; h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1)); d = 1 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1 / d; de = d * c; h *= de
        if abs(de - 1) < 1e-10: break
    return front * h


def paired(diffs):
    n = len(diffs); m = sum(diffs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0; t = m / se if se > 0 else 0.0
    df = n - 1; p = betai(0.5 * df, 0.5, df / (df + t * t)) if se > 0 else 1.0
    return m, t, p, (m / sd if sd > 0 else 0.0), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["d", "p"], required=True)
    ap.add_argument("--methods", nargs="+", default=["guided", "normalguide", "uniform"])
    ap.add_argument("--lams", nargs="+", type=float, default=[0.05, 0.15, 0.4])
    ap.add_argument("--dataset-root", default=None)
    args = ap.parse_args()
    t = args.type; defi = DEF[t]
    root = Path(args.dataset_root) if args.dataset_root else Path(DATA_ROOTS[t])
    img_dir = root / "images"
    opt_base = REPO / "data" / f"etocvd_{t}_recolor_opt"
    raw_dir = REPO / "paper" / "tables" / "raw"; raw_dir.mkdir(parents=True, exist_ok=True)

    def find(d, s):
        for e in VALID:
            if (d / f"{s}{e}").exists(): return d / f"{s}{e}"
        return None

    stems = [s for s in (p.stem for p in sorted(img_dir.iterdir()) if p.suffix.lower() in VALID)
             if (SCVD[t] / f"{s}.png").exists() and (NORMAL_SM / f"{s}.png").exists()]

    # conditions: (method,lam) + daltonize
    conds = [(m, lam) for m in args.methods for lam in args.lams] + [("daltonize", None)]
    rows = []
    # precompute base (orig) contrast + M per image
    base = {}
    for s in stems:
        orig = load_rgb(find(img_dir, s))
        M = np.clip(load_w(NORMAL_SM / f"{s}.png") - load_w(SCVD[t] / f"{s}.png"), 0, None)
        M = M / (M.max() + 1e-8)
        base[s] = (orig, rgb2lab(orig), local_contrast(orig, defi), M > MASK_THR, M)

    for (m, lam) in conds:
        cdir = opt_base / ("daltonize" if m == "daltonize" else f"{m}_lam{lam}")
        if not cdir.exists():
            print(f"[skip] {cdir} not found"); continue
        for s in stems:
            rp = cdir / f"{s}.png"
            if not rp.exists():
                continue
            orig, orig_lab, base_con, mask, M = base[s]
            rec = load_rgb(rp)
            natur = float(deltaE_ciede2000(orig_lab, rgb2lab(rec)).mean())
            rec_con = local_contrast(rec, defi)
            gain = float((rec_con[mask] - base_con[mask]).mean()) if mask.sum() > 0 else 0.0
            gain_full = float((rec_con - base_con).mean())
            ss = float(ssim(orig, rec, channel_axis=2))
            rows.append({"method": m, "lam": (lam if lam is not None else ""),
                         "stem": s, "naturalness_cost": natur,
                         "distinguish_gain": gain, "distinguish_gain_full": gain_full,
                         "ssim": ss})

    out = raw_dir / f"recolor_method_{t}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "lam", "stem", "naturalness_cost",
                                          "distinguish_gain", "distinguish_gain_full", "ssim"])
        w.writeheader(); w.writerows(rows)

    # summary per condition
    print(f"=== recolor method eval ({t.upper()}) n={len(stems)} ===")
    print(f"{'cond':22s} {'natcost(low)':>13} {'gain(high)':>11} {'ssim':>7}")
    agg = {}
    for (m, lam) in conds:
        key = m if m == "daltonize" else f"{m}_lam{lam}"
        vr = [r for r in rows if r["method"] == m and (m == "daltonize" or float(r["lam"]) == lam)]
        if not vr: continue
        nc = np.mean([r["naturalness_cost"] for r in vr])
        gn = np.mean([r["distinguish_gain"] for r in vr])
        sm = np.mean([r["ssim"] for r in vr])
        agg[key] = (nc, gn, sm, vr)
        print(f"{key:22s} {nc:>13.3f} {gn:>11.3f} {sm:>7.4f}")

    # Pareto / paired: guided vs daltonize at matched-distinguishability (closest gain)
    if "daltonize" in agg:
        dnc, dgn, dsm, dvr = agg["daltonize"]
        print(f"\n=== Pareto: does guided reach the daltonize gain={dgn:.3f} at lower naturalness cost? ===")
        gkeys = [k for k in agg if k.startswith("guided_lam")]
        if gkeys:
            best = min(gkeys, key=lambda k: abs(agg[k][1] - dgn))
            gnc, ggn, gsm, gvr = agg[best]
            dmap = {r["stem"]: r for r in dvr}; gmap = {r["stem"]: r for r in gvr}
            common = sorted(set(dmap) & set(gmap))
            dnat = [gmap[s]["naturalness_cost"] - dmap[s]["naturalness_cost"] for s in common]
            mn, tt, p, d, n = paired(dnat); lo, hi = boot_ci(dnat)
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {best}: gain {ggn:.3f} (vs daltonize {dgn:.3f}), natcost {gnc:.3f} vs {dnc:.3f}")
            print(f"  paired natcost(guided-daltonize): {mn:+.3f} [{lo:+.3f},{hi:+.3f}] "
                  f"t={tt:.2f} p={p:.4f} d={d:.2f} {star}  (<0 = guided more natural at = gain)")
    print(f"\nraw: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
