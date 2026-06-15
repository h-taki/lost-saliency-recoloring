"""End-to-end recoloring pipeline: build the lost-saliency map M and run targeted recoloring."""
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
DATA_ROOTS = {
    "d": ("/PATH/TO/EToCVD/D",
          simulate.Deficiency.DEUTAN),
    "p": ("/PATH/TO/EToCVD/P",
          simulate.Deficiency.PROTAN),
}
NORMAL_SAL = "/PATH/TO/EToCVD/normal/saliency_maps"
HW = (240, 320)
VARIANTS = ("guided", "agnostic", "uniform")


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((HW[1], HW[0]),
                      Image.BILINEAR), dtype=np.uint8)


def load_gray01(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("L").resize((HW[1], HW[0]),
                   Image.BILINEAR), dtype=np.float32) / 255.0
    return a



_ERR2MOD = np.array([[0.0, 0.0, 0.0],
                     [0.7, 1.0, 0.0],
                     [0.7, 0.0, 1.0]], dtype=np.float64)
_SIM = simulate.Simulator_Machado2009()


def daltonize(img: np.ndarray, deficiency) -> np.ndarray:
    """(documentation omitted in the English release)."""
    imgf = img.astype(np.float64)
    isim = _SIM.simulate_cvd(img, deficiency, severity=1.0).astype(np.float64)
    err = imgf - isim
    corr = err @ _ERR2MOD.T
    return np.clip(imgf + corr, 0, 255).astype(np.uint8)


def cvd_sim(img: np.ndarray, deficiency) -> np.ndarray:
    return np.clip(_SIM.simulate_cvd(img, deficiency, severity=1.0), 0, 255).astype(np.uint8)


def local_contrast_ciede(rgb: np.ndarray) -> np.ndarray:
    """(documentation omitted in the English release)."""
    lab = rgb2lab(rgb)
    dx = np.zeros(rgb.shape[:2], dtype=np.float64)
    dy = np.zeros(rgb.shape[:2], dtype=np.float64)
    dx[:, :-1] = deltaE_ciede2000(lab[:, :-1], lab[:, 1:])
    dy[:-1, :] = deltaE_ciede2000(lab[:-1, :], lab[1:, :])
    return (dx + dy) / 2.0


def recolor(img: np.ndarray, T: np.ndarray, M: np.ndarray, variant: str) -> np.ndarray:
    w = {"guided": M, "agnostic": np.ones_like(M),
         "uniform": np.full_like(M, float(M.mean()))}[variant][..., None]
    out = (1 - w) * img.astype(np.float32) + w * T.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---- metrics (saliency) ----
def cc(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-8 else 0.0


def sim_metric(a, b):
    a = a / (a.sum() + 1e-8); b = b / (b.sum() + 1e-8)
    return float(np.minimum(a, b).sum())


def nss(pred, fix_bin):
    s = pred.std()
    if s < 1e-8 or fix_bin.sum() < 1:
        return 0.0
    z = (pred - pred.mean()) / s
    return float(z[fix_bin > 0.5].mean())


def paired_t(diffs):
    n = len(diffs); m = sum(diffs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1)) if n > 1 else 0
    se = sd / math.sqrt(n) if n else 0
    t = m / se if se > 0 else 0

    return m, sd, t, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["d", "p"], required=True)
    ap.add_argument("--e2", action="store_true", help="E2 aggregation (after S_cvd' is prepared)")
    ap.add_argument("--scvd-dir", default=None)
    ap.add_argument("--normal-dir", default=None,
                    help="S_normal dir override (new GT: data/etocvd_normal_sm_sigma3p0)")
    ap.add_argument("--out-recolor", default=None, help="output path for recolored images")
    ap.add_argument("--scvd-recolored-prefix", default=None,
                    help="E2: dir prefix for re-prediction on recolored images (e.g. data/etocvd_d_scvd_recolored_newgt)")
    ap.add_argument("--tag", default="", help="output CSV suffix (e.g. newgt)")
    args = ap.parse_args()

    root, deficiency = DATA_ROOTS[args.type]
    root = Path(root)
    img_dir = root / "images"
    scvd_dir = Path(args.scvd_dir) if args.scvd_dir else \
        REPO / "data" / f"etocvd_{args.type}_scvd_perfold"
    normal_dir = Path(args.normal_dir) if args.normal_dir else \
        (Path(NORMAL_SAL) if args.type == "d" else Path(str(NORMAL_SAL)))
    out_recolor = Path(args.out_recolor) if args.out_recolor else \
        REPO / "data" / f"etocvd_{args.type}_recolored"
    raw_dir = REPO / "paper" / "tables" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in img_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    if not args.e2:
        # ---- E1 ----
        rows = []
        for v in VARIANTS:
            (out_recolor / v).mkdir(parents=True, exist_ok=True)
        for stem in stems:
            ip = next((img_dir / f"{stem}{e}" for e in (".jpeg", ".jpg", ".png", ".bmp")
                       if (img_dir / f"{stem}{e}").exists()), None)
            sp = scvd_dir / f"{stem}.png"
            np_ = normal_dir / f"{stem}.png"
            if ip is None or not sp.exists() or not np_.exists():
                print(f"[skip] {stem} (missing input)"); continue
            img = load_rgb(ip)
            s_cvd = load_gray01(sp)
            s_nrm = load_gray01(np_)
            M = np.clip(s_nrm - s_cvd, 0, None)
            mx = M.max(); M = M / mx if mx > 1e-8 else M
            T = daltonize(img, deficiency)
            orig_sim = cvd_sim(img, deficiency)
            mask = M > 0.3
            base_contrast = local_contrast_ciede(orig_sim)
            for v in VARIANTS:
                rec = recolor(img, T, M, v)
                Image.fromarray(rec).save(out_recolor / v / f"{stem}.png")
                rec_sim = cvd_sim(rec, deficiency)
                rec_contrast = local_contrast_ciede(rec_sim)
                if mask.sum() > 0:
                    gain = float((rec_contrast[mask] - base_contrast[mask]).mean())
                else:
                    gain = 0.0
                natcost = float(deltaE_ciede2000(rgb2lab(img), rgb2lab(rec)).mean())
                ss = float(ssim(img, rec, channel_axis=2))
                rows.append({"stem": stem, "variant": v, "distinguish_gain": gain,
                             "naturalness_cost": natcost, "ssim": ss})
        out = raw_dir / f"E1_masked_daltonize_{args.type}{('_'+args.tag) if args.tag else ''}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["stem", "variant", "distinguish_gain",
                                              "naturalness_cost", "ssim"])
            w.writeheader(); w.writerows(rows)
        # summary + guided vs agnostic paired t
        print(f"=== E1 ({args.type}) per-variant mean ===")
        for v in VARIANTS:
            vr = [r for r in rows if r["variant"] == v]
            g = sum(r["distinguish_gain"] for r in vr) / len(vr)
            c = sum(r["naturalness_cost"] for r in vr) / len(vr)
            s = sum(r["ssim"] for r in vr) / len(vr)
            print(f"  {v:9s}: gain={g:.3f}  natcost={c:.3f}  ssim={s:.4f}  (n={len(vr)})")
        gd = {r["stem"]: r for r in rows if r["variant"] == "guided"}
        ag = {r["stem"]: r for r in rows if r["variant"] == "agnostic"}
        common = sorted(set(gd) & set(ag))
        dcost = [gd[s]["naturalness_cost"] - ag[s]["naturalness_cost"] for s in common]
        dgain = [gd[s]["distinguish_gain"] - ag[s]["distinguish_gain"] for s in common]
        m, sd, t, n = paired_t(dcost)
        print(f"\nguided - agnostic naturalness_cost: mean {m:+.3f} (t={t:.2f}, n={n})  "
              f"(<0 = guided more natural ★)")
        m, sd, t, n = paired_t(dgain)
        print(f"guided - agnostic distinguish_gain: mean {m:+.3f} (t={t:.2f}, n={n})")
        print(f"\nraw: {out}\nrecolored imgs: {out_recolor}/<variant>/  -> proceed to E2 re-prediction")
        return 0

    # ---- E2 ----
    print(f"=== E2 ({args.type}) restoration: S_cvd'(recolored) vs S_normal ===")
    rows = []
    e2pref = args.scvd_recolored_prefix or f"data/etocvd_{args.type}_scvd_recolored"
    for v in VARIANTS:
        pred_dir = REPO / f"{e2pref}_{v}"
        if not pred_dir.exists():
            print(f"[skip] {v}: {pred_dir} not found (predict_recolored_scvd.py not run)"); continue
        for stem in stems:
            pp = pred_dir / f"{stem}.png"
            np_ = normal_dir / f"{stem}.png"
            if not pp.exists() or not np_.exists():
                continue
            scvd2 = load_gray01(pp)
            snrm = load_gray01(np_)
            thr = np.percentile(snrm, 95)
            fix = (snrm > thr).astype(np.float32)
            rows.append({"stem": stem, "variant": v,
                         "cc": cc(scvd2, snrm), "sim": sim_metric(scvd2, snrm),
                         "nss": nss(scvd2, fix)})
    if rows:
        out = raw_dir / f"E2_saliency_restoration_{args.type}{('_'+args.tag) if args.tag else ''}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["stem", "variant", "cc", "sim", "nss"])
            w.writeheader(); w.writerows(rows)
        print(f"{'variant':9s} {'CC':>8} {'SIM':>8} {'NSS':>8}")
        for v in VARIANTS:
            vr = [r for r in rows if r["variant"] == v]
            if not vr: continue
            print(f"{v:9s} {sum(r['cc'] for r in vr)/len(vr):>8.4f} "
                  f"{sum(r['sim'] for r in vr)/len(vr):>8.4f} "
                  f"{sum(r['nss'] for r in vr)/len(vr):>8.4f}")
        print(f"\nraw: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
