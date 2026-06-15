"""Create CVD-simulated SALICON images for pseudo-CVD pretraining."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def _get_simulator():
    """(documentation omitted in the English release)."""
    try:
        from daltonlens import simulate
        return simulate.Simulator_Brettel1997()
    except ImportError:
        raise ImportError(
            "daltonlens is not installed. Install it with `pip install daltonlens`.\n"
            "(for CVD simulation via the Brettel 1997 method)"
        )


def _get_deficiency(name: str):
    """(documentation omitted in the English release)."""
    name = name.upper()
    enum_cls = None
    try:
        from daltonlens import simulate as dl_sim
        if hasattr(dl_sim, "Deficiency"):
            enum_cls = dl_sim.Deficiency
    except Exception:
        pass
    if enum_cls is None:
        try:
            from daltonlens import convert as dl_convert
            if hasattr(dl_convert, "Deficiency"):
                enum_cls = dl_convert.Deficiency
        except Exception:
            pass
    if enum_cls is None:
        raise ImportError(
            "daltonlens.Deficiency enum not found. "
            "Please check the daltonlens API. "
            "`python -c 'import daltonlens; print(dir(daltonlens))'`"
        )

    mapping = {
        "DEUTAN": getattr(enum_cls, "DEUTAN", None),
        "PROTAN": getattr(enum_cls, "PROTAN", None),
        "TRITAN": getattr(enum_cls, "TRITAN", None),
    }
    if mapping.get(name) is None:
        raise ValueError(
            f"deficiency must be one of available enum values "
            f"({[k for k, v in mapping.items() if v is not None]}), got {name!r}"
        )
    return mapping[name]


def convert_image(img_pil: Image.Image, sim, deficiency, severity: float) -> Image.Image:
    """(documentation omitted in the English release)."""
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    rgb = np.asarray(img_pil)
    cvd_rgb = sim.simulate_cvd(rgb, deficiency, severity=severity)
    return Image.fromarray(np.asarray(cvd_rgb, dtype=np.uint8))


def process_split(
    src_root: Path,
    dst_root: Path,
    sim,
    deficiency,
    severity: float,
    verbose: bool,
) -> None:
    """(documentation omitted in the English release)."""

    src_img_dir = src_root / "images"
    dst_img_dir = dst_root / "images"
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    if src_img_dir.exists():
        files = sorted(src_img_dir.iterdir())
        n = 0
        for f in files:
            if not f.is_file() or f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            out = dst_img_dir / f.name
            if out.exists():
                n += 1
                continue
            with Image.open(f) as im:
                cvd = convert_image(im, sim, deficiency, severity)
            cvd.save(out, quality=95)
            n += 1
            if verbose and n % 500 == 0:
                print(f"  ... {n}/{len(files)} images")
        print(f"  images: {n} converted -> {dst_img_dir}")
    else:
        print(f"  SKIP images: {src_img_dir} not found")


    for sub in ("saliency_maps", "fixation_maps"):
        src_sub = src_root / sub
        dst_sub = dst_root / sub
        if not src_sub.exists():
            print(f"  SKIP {sub}: {src_sub} not found")
            continue
        dst_sub.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(src_sub.iterdir()):
            if not f.is_file():
                continue
            out = dst_sub / f.name
            if out.exists():
                n += 1
                continue
            shutil.copy2(f, out)
            n += 1
        print(f"  {sub}: {n} copied -> {dst_sub}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=Path,
                   default=Path("data/salicon_2017/processed/single_split/fold_1"),
                   help="input SALICON processed root; expects fold_1/{train,val}/*")
    p.add_argument("--dst-root", type=Path,
                   default=Path("data/salicon_2017_d_cvd/processed/single_split/fold_1"),
                   help="output root for the CVD-converted version (same directory hierarchy)")
    p.add_argument("--deficiency", default="DEUTAN",
                   help="CVD type: DEUTAN (deutan, default) / PROTAN (protan) / TRITAN (tritan)")
    p.add_argument("--severity", type=float, default=1.0,
                   help="severity 0..1, default 1.0 (full dichromacy)")
    p.add_argument("--verbose", action="store_true", default=True)
    args = p.parse_args()

    sim = _get_simulator()
    deficiency = _get_deficiency(args.deficiency)

    print(f"src_root  : {args.src_root.resolve()}")
    print(f"dst_root  : {args.dst_root.resolve()}")
    print(f"deficiency: {args.deficiency}")
    print(f"severity  : {args.severity}")
    print()

    for split in ("train", "val"):
        src_split = args.src_root / split
        dst_split = args.dst_root / split
        if not src_split.exists():
            print(f"--- {split}: SKIP ({src_split} not found) ---")
            continue
        print(f"--- {split} ---")
        process_split(src_split, dst_split, sim, deficiency, args.severity, args.verbose)
        print()

    print(f">> done. Next: pretrain with configs/msi_salicon_d_cvd_pretrain.yaml")


if __name__ == "__main__":
    main()
