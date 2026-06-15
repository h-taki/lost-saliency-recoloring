"""Simplified Kuhn-style mass-spring recoloring baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from daltonlens import simulate
from skimage.color import rgb2lab, lab2rgb

REPO = Path(__file__).resolve().parent.parent
DATA_ROOTS = {"d": "/PATH/TO/EToCVD/D",
              "p": "/PATH/TO/EToCVD/P"}
DEF = {"d": simulate.Deficiency.DEUTAN, "p": simulate.Deficiency.PROTAN}
HW = (240, 320)
VALID = (".png", ".jpg", ".jpeg", ".bmp")
SIM = simulate.Simulator_Machado2009()


def kmeans_palette(lab, k=16, iters=10, seed=0):
    rng = np.random.default_rng(seed)
    pts = lab.reshape(-1, 3)
    idx = rng.choice(len(pts), k, replace=False)
    C = pts[idx].copy()
    for _ in range(iters):
        d = ((pts[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = pts[m].mean(0)
    return C, a


def cvd_sim_lab(rgb_u8, defi):
    s = SIM.simulate_cvd(rgb_u8, defi, 1.0).astype(np.uint8)
    return rgb2lab(s)


def mass_spring(C, C_sim, k_n=0.30, k_c=0.10, iters=120, dt=0.5, damp=0.85):
    """(documentation omitted in the English release)."""
    k = len(C)
    ab = C[:, 1:].copy()
    ab0 = C[:, 1:].copy()
    sim_ab = C_sim[:, 1:]

    dorig = np.sqrt(((C[:, None, 1:] - C[None, :, 1:]) ** 2).sum(-1)) + 1e-6
    dsim = np.sqrt(((sim_ab[:, None] - sim_ab[None]) ** 2).sum(-1)) + 1e-6
    confuse = np.clip(dorig - dsim, 0, None)
    vel = np.zeros_like(ab)
    for _ in range(iters):
        F = -k_n * (ab - ab0)  # naturalness spring

        db = ab[:, None, 1] - ab[None, :, 1]
        sgn = np.sign(db) + (db == 0) * 1.0
        rep_b = (k_c * confuse * sgn)
        np.fill_diagonal(rep_b, 0)
        F[:, 1] += rep_b.sum(1)
        vel = damp * (vel + dt * F)
        ab = ab + dt * vel
        ab = np.clip(ab, -127, 127)
    out = C.copy(); out[:, 1:] = ab
    return out


def recolor_image(lab, assign, C, C_new):
    H, W, _ = lab.shape
    disp = (C_new - C)[:, 1:]

    pts = lab.reshape(-1, 3)
    d = ((pts[:, None, 1:] - C[None, :, 1:]) ** 2).sum(-1)
    w = np.exp(-d / (2 * (15.0 ** 2)))
    w = w / (w.sum(1, keepdims=True) + 1e-8)
    shift = w @ disp
    out = pts.copy(); out[:, 1:] += shift
    return out.reshape(H, W, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["d", "p"], required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--k-c", type=float, default=0.04, help="contrast repulsion strength (naturalness trade-off)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset-root", default=None)
    args = ap.parse_args()
    t = args.type; defi = DEF[t]
    root = Path(args.dataset_root) if args.dataset_root else Path(DATA_ROOTS[t])
    img_dir = root / "images"
    out_dir = REPO / "data" / f"etocvd_{t}_recolor_kuhn"; out_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in img_dir.iterdir() if p.suffix.lower() in VALID)
    if args.limit:
        stems = stems[: args.limit]

    def find(d, s):
        for e in VALID:
            if (d / f"{s}{e}").exists():
                return d / f"{s}{e}"
        return None

    n = 0
    for s in stems:
        ip = find(img_dir, s)
        if ip is None:
            continue
        rgb = np.asarray(Image.open(ip).convert("RGB").resize((HW[1], HW[0]), Image.BILINEAR),
                         dtype=np.uint8)
        lab = rgb2lab(rgb)
        C, assign = kmeans_palette(lab, k=args.k)

        C_rgb = np.clip(lab2rgb(C.reshape(1, -1, 3)).reshape(-1, 3) * 255, 0, 255).astype(np.uint8)
        C_sim = rgb2lab(SIM.simulate_cvd(C_rgb.reshape(1, -1, 3), defi, 1.0).astype(np.uint8)
                        ).reshape(-1, 3)
        C_new = mass_spring(C, C_sim, k_c=args.k_c)
        lab2 = recolor_image(lab, assign, C, C_new)
        rec = np.clip(lab2rgb(lab2) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(rec).save(out_dir / f"{s}.png")
        n += 1
        if n % 25 == 0:
            print(f"  {n} done")
    print(f"[kuhn] type={t} processed {n} -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
