"""Prepare the SALICON 2017 dataset for saliency pretraining."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def _copy_files(src_dir: Path, dst_dir: Path,
                allowed_ext: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
                symlink: bool = False) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src_dir.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in allowed_ext:
            continue
        out = dst_dir / f.name
        if out.exists():
            count += 1
            continue
        if symlink:
            try:
                out.symlink_to(f.resolve())
            except OSError:
                shutil.copy2(f, out)
        else:
            shutil.copy2(f, out)
        count += 1
    return count


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def _mat_to_fixation_png(mat_path: Path, target_size_hw: tuple[int, int],
                        out_png_path: Path) -> bool:
    """(documentation omitted in the English release)."""
    try:
        from scipy.io import loadmat
    except ImportError:
        return False
    try:
        data = loadmat(str(mat_path))
    except Exception:
        return False

    H, W = target_size_hw
    binary = np.zeros((H, W), dtype=np.uint8)


    gaze_data = data.get("gaze")
    if gaze_data is None:
        return False

    try:

        n_subj = gaze_data.shape[1] if gaze_data.ndim == 2 else gaze_data.size
        for i in range(n_subj):
            entry = gaze_data[0, i] if gaze_data.ndim == 2 else gaze_data.flat[i]

            coords = None
            if hasattr(entry, "dtype") and entry.dtype.names is not None:
                if "fixations" in entry.dtype.names:
                    coords = entry["fixations"][0, 0]
            elif isinstance(entry, np.ndarray) and entry.ndim == 2 and entry.shape[1] == 2:
                coords = entry
            if coords is None:
                continue
            for x, y in coords:
                xi = int(np.clip(round(float(x)), 0, W - 1))
                yi = int(np.clip(round(float(y)), 0, H - 1))
                binary[yi, xi] = 255
    except Exception:
        return False

    Image.fromarray(binary, mode="L").save(out_png_path)
    return True


def _fallback_fixation_from_saliency(saliency_path: Path,
                                     out_png_path: Path,
                                     top_percent: float = 5.0) -> bool:
    """(documentation omitted in the English release)."""
    sal = np.asarray(Image.open(saliency_path).convert("L"))
    thr = np.percentile(sal, 100 - top_percent)
    binary = (sal >= thr).astype(np.uint8) * 255
    Image.fromarray(binary, mode="L").save(out_png_path)
    return True


def _prepare_fixation_maps(image_dir: Path, sal_dir: Path,
                          fix_mat_dir: Path | None, out_dir: Path,
                          verbose: bool = False) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        stem = img_path.stem
        out_png = out_dir / f"{stem}.png"
        if out_png.exists():
            count += 1
            continue


        with Image.open(img_path) as im:
            W, H = im.size

        success = False
        if fix_mat_dir is not None:
            mat_path = fix_mat_dir / f"{stem}.mat"
            if mat_path.exists():
                success = _mat_to_fixation_png(mat_path, (H, W), out_png)

        if not success:
            sal_path = sal_dir / f"{stem}.png"
            if not sal_path.exists():
                if verbose:
                    print(f"  WARN: no saliency for {stem}, skipping")
                continue
            _fallback_fixation_from_saliency(sal_path, out_png)

        count += 1
        if verbose and count % 500 == 0:
            print(f"  ... {count} fixation maps generated")
    return count


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="root of the downloaded SALICON raw data")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="output root formatted for this implementation")
    parser.add_argument("--symlink", action="store_true",
                        help="symlink images/saliency instead of copying (saves disk; needs permissions on Windows)")
    parser.add_argument("--no-fixations", action="store_true",
                        help="do not generate fixation_maps (when pretraining with train/saliency_maps only)")
    args = parser.parse_args()

    raw = args.raw_root.resolve()
    out = args.out_root.resolve()

    print(f"raw_root: {raw}")
    print(f"out_root: {out}")
    print()

    for split in ("train", "val"):


        layout_a_img = raw / "images" / split
        layout_a_sal = raw / "fixation_maps" / split
        layout_a_fix = raw / "fixations" / split

        layout_b_img = raw / f"{split}_images"
        layout_b_sal = raw / f"{split}_saliency_maps"
        layout_b_fix = raw / f"{split}_fixation_maps"

        if layout_a_img.exists() and layout_a_sal.exists():
            src_img = layout_a_img
            src_sal = layout_a_sal
            src_fix_mat = layout_a_fix
            print(f"--- {split} (layout A: official zip) ---")
        elif layout_b_img.exists() and layout_b_sal.exists():
            src_img = layout_b_img
            src_sal = layout_b_sal
            src_fix_mat = layout_b_fix
            print(f"--- {split} (layout B: flat) ---")
        else:
            print(f"SKIP {split}: not found in either layout")
            print(f"  expected (A): {layout_a_img}")
            print(f"  expected (B): {layout_b_img}")
            continue

        dst_img = out / split / "images"
        dst_sal = out / split / "saliency_maps"
        dst_fix = out / split / "fixation_maps"

        n_img = _copy_files(src_img, dst_img, (".jpg", ".jpeg"), symlink=args.symlink)
        print(f"  images:        {n_img}")
        n_sal = _copy_files(src_sal, dst_sal, (".png",), symlink=args.symlink)
        print(f"  saliency_maps: {n_sal}")

        if args.no_fixations:
            print("  fixation_maps: skipped (--no-fixations)")
        else:
            fix_dir = src_fix_mat if src_fix_mat is not None and src_fix_mat.exists() else None
            if fix_dir is None:
                print(f"  fixation_maps: .mat not found; generate pseudo-fixations from saliency")
            n_fix = _prepare_fixation_maps(
                dst_img, dst_sal, fix_dir, dst_fix, verbose=True
            )
            print(f"  fixation_maps: {n_fix}")

    print()
    print(f">> done: {out}")
    print(f">> Next: continue MSI-Net pretraining with configs/msi_salicon_pretrain.yaml")


if __name__ == "__main__":
    main()
