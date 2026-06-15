"""Targeted CVD recoloring as a per-image optimization in CIELAB (L* fixed, a*/b* free): maximize local contrast under Machado CVD simulation inside the lost-saliency region while penalizing normal-vision color change."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
REPO = Path(__file__).resolve().parent.parent
DATA_ROOTS = {"d": "/PATH/TO/EToCVD/D",
              "p": "/PATH/TO/EToCVD/P"}
SCVD = {"d": REPO / "data" / "etocvd_d_scvd_perfold_newgt",
        "p": REPO / "data" / "etocvd_p_scvd_perfold_newgt"}
NORMAL_SM = REPO / "data" / "etocvd_normal_sm_sigma3p0"
HW = (240, 320)
VALID = (".png", ".jpg", ".jpeg", ".bmp")
MACHADO = {
    "d": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
          [-0.011820, 0.042940, 0.968881]],
    "p": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
          [-0.003882, -0.048116, 1.051998]],
}

MACHADO_SEV = {
    ("d", 0.6): [[0.498864, 0.674741, -0.173604], [0.205199, 0.754872, 0.039929], [-0.011131, 0.030969, 0.980162]],
    ("d", 0.8): [[0.422823, 0.781057, -0.203881], [0.245752, 0.709602, 0.044646], [-0.011843, 0.037423, 0.974421]],
    ("d", 1.0): [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    ("p", 0.6): [[0.385450, 0.769005, -0.154455], [0.100526, 0.829802, 0.069673], [-0.007442, -0.022190, 1.029632]],
    ("p", 0.8): [[0.259411, 0.923008, -0.182420], [0.110296, 0.804340, 0.085364], [-0.006276, -0.034346, 1.040622]],
    ("p", 1.0): [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
}
ERR2MOD = [[0.0, 0.0, 0.0], [0.7, 1.0, 0.0], [0.7, 0.0, 1.0]]
_MXYZ = [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
_WHITE = [0.95047, 1.0, 1.08883]


def _dev(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


def srgb_to_linear(c):
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = c.clamp(1e-8, 1)
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1 / 2.4)) - 0.055)


def _mm(img, M):
    return torch.einsum("...c,kc->...k", img, M)


def srgb_to_lab(img, MX, WH):
    xyz = _mm(srgb_to_linear(img), MX) / WH
    d = 6 / 29
    f = torch.where(xyz > d ** 3, xyz.clamp(min=1e-8) ** (1 / 3), xyz / (3 * d * d) + 4 / 29)
    return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                        200 * (f[..., 1] - f[..., 2])], dim=-1)


def lab_to_srgb(lab, MXI, WH):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116; fx = fy + a / 500; fz = fy - b / 200; d = 6 / 29
    inv = lambda t: torch.where(t > d, t ** 3, 3 * d * d * (t - 4 / 29))
    xyz = torch.stack([inv(fx), inv(fy), inv(fz)], dim=-1) * WH
    return linear_to_srgb(_mm(xyz, MXI))


def machado_sim(img, M):
    return linear_to_srgb(_mm(srgb_to_linear(img), M).clamp(0, 1))


def gradmag(lab):
    dx = torch.zeros(lab.shape[:-1], device=lab.device)
    dy = torch.zeros(lab.shape[:-1], device=lab.device)
    dx[..., :, :-1] = torch.linalg.norm(lab[..., :, 1:, :] - lab[..., :, :-1, :], dim=-1)
    dy[..., :-1, :] = torch.linalg.norm(lab[..., 1:, :, :] - lab[..., :-1, :, :], dim=-1)
    return (dx + dy) / 2.0


def daltonize(img, Mmac, Merr):
    return (img + _mm(img - machado_sim(img, Mmac), Merr)).clamp(0, 1)


def optimize_batch(imgs, weights, Mmac, MX, MXI, WH, lam, iters, lr=0.05, free_L=False):
    """(documentation omitted in the English release)."""
    lab0 = srgb_to_lab(imgs, MX, WH)
    L0 = lab0[..., 0:1]
    base = gradmag(srgb_to_lab(machado_sim(imgs, Mmac), MX, WH)).detach()
    w = weights.clamp(min=0)
    if free_L:
        lab = lab0.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([lab], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            srgb = lab_to_srgb(lab, MXI, WH)
            con = gradmag(srgb_to_lab(machado_sim(srgb, Mmac), MX, WH))
            gain = (w * (con - base)).sum()
            natur = torch.linalg.norm(lab - lab0, dim=-1).sum()
            (-gain + lam * natur).backward()
            opt.step()
            with torch.no_grad():
                lab[..., 0].clamp_(0, 100); lab[..., 1:].clamp_(-127, 127)
        with torch.no_grad():
            out = lab_to_srgb(lab, MXI, WH).clamp(0, 1)
    else:
        ab = lab0[..., 1:].clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([ab], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            srgb = lab_to_srgb(torch.cat([L0, ab], dim=-1), MXI, WH)
            con = gradmag(srgb_to_lab(machado_sim(srgb, Mmac), MX, WH))
            gain = (w * (con - base)).sum()
            natur = torch.linalg.norm(ab - lab0[..., 1:], dim=-1).sum()
            (-gain + lam * natur).backward()
            opt.step()
            with torch.no_grad():
                ab.clamp_(-127, 127)
        with torch.no_grad():
            out = lab_to_srgb(torch.cat([L0, ab], dim=-1), MXI, WH).clamp(0, 1)
    return np.nan_to_num(out.cpu().numpy() * 255).astype(np.uint8)


def find(d, s):
    for e in VALID:
        if (d / f"{s}{e}").exists():
            return d / f"{s}{e}"
    return None


def load(p, gray=False):
    im = Image.open(p).convert("L" if gray else "RGB").resize((HW[1], HW[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["d", "p"], required=True)
    ap.add_argument("--methods", nargs="+", default=["guided", "normalguide", "uniform"])
    ap.add_argument("--lams", nargs="+", type=float, default=[0.05, 0.15, 0.4])
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument("--scvd-dir", default=None, help="S_cvd source override (C-1: baseline OOF)")
    ap.add_argument("--snormal-dir", default=None, help="S_normal source override (C-2: full-pred)")
    ap.add_argument("--severity", type=float, default=1.0, choices=[0.6, 0.8, 1.0],
                    help="Machado severity (C-3 #9)")
    ap.add_argument("--out-tag", default=None, help="output dir name (e.g. guided_baseline); if omitted, the method name")
    ap.add_argument("--no-daltonize", action="store_true")
    ap.add_argument("--free-L", action="store_true", help="also optimize L* (C-5: L*-free ablation)")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[recolor_opt] device={device} severity={args.severity} "
          f"scvd={args.scvd_dir or 'default'} snormal={args.snormal_dir or 'default'}")
    t = args.type
    Mmac = _dev(MACHADO_SEV[(t, args.severity)], device); Merr = _dev(ERR2MOD, device)
    MX = _dev(_MXYZ, device); MXI = torch.inverse(MX); WH = _dev(_WHITE, device)

    root = Path(args.dataset_root) if args.dataset_root else Path(DATA_ROOTS[t])
    img_dir = root / "images"
    out_base = REPO / "data" / f"etocvd_{t}_recolor_opt"
    scvd_dir = Path(args.scvd_dir) if args.scvd_dir else SCVD[t]
    snorm_dir = Path(args.snormal_dir) if args.snormal_dir else NORMAL_SM
    stems = [s for s in (p.stem for p in sorted(img_dir.iterdir()) if p.suffix.lower() in VALID)
             if (scvd_dir / f"{s}.png").exists() and (snorm_dir / f"{s}.png").exists()]
    if args.limit:
        stems = stems[: args.limit]


    if not args.no_daltonize:
        dd = out_base / "daltonize"; dd.mkdir(parents=True, exist_ok=True)
        for s in stems:
            im = _dev(load(find(img_dir, s)), device)
            o = daltonize(im, Mmac, Merr)
            Image.fromarray((o.cpu().numpy() * 255).astype(np.uint8)).save(dd / f"{s}.png")
        print(f"  daltonize: {len(stems)}")

    for m in args.methods:
        for lam in args.lams:
            tag = args.out_tag if args.out_tag else m
            outd = out_base / f"{tag}_lam{lam}"; outd.mkdir(parents=True, exist_ok=True)
            for i in range(0, len(stems), args.chunk):
                cs = stems[i:i + args.chunk]
                imgs = torch.stack([_dev(load(find(img_dir, s)), device) for s in cs])
                sc = torch.stack([_dev(load(scvd_dir / f"{s}.png", True), device) for s in cs])
                sn = torch.stack([_dev(load(snorm_dir / f"{s}.png", True), device) for s in cs])
                M = (sn - sc).clamp(min=0)
                M = M / (M.amax(dim=(1, 2), keepdim=True) + 1e-8)
                if m == "guided":
                    W = M
                elif m == "normalguide":
                    W = sn / (sn.amax(dim=(1, 2), keepdim=True) + 1e-8)
                else:
                    W = M.mean(dim=(1, 2), keepdim=True).expand_as(M)
                rec = optimize_batch(imgs, W, Mmac, MX, MXI, WH, lam, args.iters,
                                     free_L=args.free_L)
                for j, s in enumerate(cs):
                    Image.fromarray(rec[j]).save(outd / f"{s}.png")
            print(f"  {m} lam{lam}: {len(stems)}")
    print(f"[recolor_opt] type={t} done -> {out_base}\nNext: scripts/recolor_eval.py --type {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
