"""Build leakage-free cross-validation splits from the original images."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


AUG_SUFFIXES = ("_hflip", "_crop", "_mask", "_vflip")


IMG_EXTS = (".jpeg", ".jpg", ".png", ".bmp")


def is_original(stem: str) -> bool:
    """(documentation omitted in the English release)."""
    return not any(stem.endswith(s) for s in AUG_SUFFIXES)


def read_ids_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def copy_one(src_dir: Path, dst_dir: Path, filename: str) -> bool:
    dst_dir.mkdir(parents=True, exist_ok=True)
    src = src_dir / filename
    if not src.exists():
        print(f"  [warn] missing: {src}")
        return False
    shutil.copy2(src, dst_dir / filename)
    return True


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def build_fold_csv(
    fold_idx: int,
    src_fold_dir: Path,
    dst_fold_dir: Path,
    images_src: Path,
    saliency_src: Path,
    fixation_src: Path,
) -> tuple[int, int]:
    """(documentation omitted in the English release)."""
    train_rows = read_ids_csv(src_fold_dir / "train_ids.csv")
    val_rows = read_ids_csv(src_fold_dir / "val_ids.csv")

    orig_train = [r for r in train_rows if is_original(Path(r["image"]).stem)]
    print(f"fold {fold_idx}: train rows {len(train_rows)} -> originals only {len(orig_train)}, "
          f"val {len(val_rows)}", flush=True)

    n_train = 0
    n_val = 0
    for r in orig_train:
        ok_i = copy_one(images_src, dst_fold_dir / "train" / "images", r["image"])
        ok_s = copy_one(saliency_src, dst_fold_dir / "train" / "saliency_maps", r["map"])
        ok_f = copy_one(fixation_src, dst_fold_dir / "train" / "fixation_maps", r["fixation"])
        if ok_i and ok_s and ok_f:
            n_train += 1

    for r in val_rows:
        ok_i = copy_one(images_src, dst_fold_dir / "val" / "images", r["image"])
        ok_s = copy_one(saliency_src, dst_fold_dir / "val" / "saliency_maps", r["map"])
        ok_f = copy_one(fixation_src, dst_fold_dir / "val" / "fixation_maps", r["fixation"])
        if ok_i and ok_s and ok_f:
            n_val += 1

    return n_train, n_val


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def _list_image_basenames(d: Path) -> list[str]:
    """(documentation omitted in the English release)."""
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _match_by_stem(stem: str, files: list[str]) -> str | None:
    """(documentation omitted in the English release)."""
    for f in files:
        if Path(f).stem == stem:
            return f
    return None


def build_fold_fs(
    fold_idx: int,
    src_fold_dir: Path,
    dst_fold_dir: Path,
) -> tuple[int, int]:
    """(documentation omitted in the English release)."""
    src_train_img = src_fold_dir / "train" / "images"
    src_train_sal = src_fold_dir / "train" / "saliency_maps"
    src_train_fix = src_fold_dir / "train" / "fixation_maps"
    src_val_img = src_fold_dir / "val" / "images"
    src_val_sal = src_fold_dir / "val" / "saliency_maps"
    src_val_fix = src_fold_dir / "val" / "fixation_maps"

    if not src_train_img.exists():
        print(f"[skip] fold {fold_idx}: missing {src_train_img}")
        return 0, 0

    train_imgs_all = _list_image_basenames(src_train_img)
    train_sals_all = _list_image_basenames(src_train_sal)
    train_fixs_all = _list_image_basenames(src_train_fix)
    val_imgs_all = _list_image_basenames(src_val_img)
    val_sals_all = _list_image_basenames(src_val_sal)
    val_fixs_all = _list_image_basenames(src_val_fix)

    orig_train_imgs = [f for f in train_imgs_all if is_original(Path(f).stem)]
    orig_val_imgs = [f for f in val_imgs_all if is_original(Path(f).stem)]
    print(f"fold {fold_idx}: train imgs {len(train_imgs_all)} -> originals only {len(orig_train_imgs)}, "
          f"val imgs {len(val_imgs_all)} -> originals only {len(orig_val_imgs)}", flush=True)

    n_train = 0
    n_val = 0
    for img in orig_train_imgs:
        stem = Path(img).stem
        sal = _match_by_stem(stem, train_sals_all)
        fix = _match_by_stem(stem, train_fixs_all)
        if sal is None:
            print(f"  [warn] no saliency for {stem} (train)")
            continue
        if fix is None:
            print(f"  [warn] no fixation for {stem} (train)")
            continue
        ok_i = copy_one(src_train_img, dst_fold_dir / "train" / "images", img)
        ok_s = copy_one(src_train_sal, dst_fold_dir / "train" / "saliency_maps", sal)
        ok_f = copy_one(src_train_fix, dst_fold_dir / "train" / "fixation_maps", fix)
        if ok_i and ok_s and ok_f:
            n_train += 1

    for img in orig_val_imgs:
        stem = Path(img).stem
        sal = _match_by_stem(stem, val_sals_all)
        fix = _match_by_stem(stem, val_fixs_all)
        if sal is None:
            print(f"  [warn] no saliency for {stem} (val)")
            continue
        if fix is None:
            print(f"  [warn] no fixation for {stem} (val)")
            continue
        ok_i = copy_one(src_val_img, dst_fold_dir / "val" / "images", img)
        ok_s = copy_one(src_val_sal, dst_fold_dir / "val" / "saliency_maps", sal)
        ok_f = copy_one(src_val_fix, dst_fold_dir / "val" / "fixation_maps", fix)
        if ok_i and ok_s and ok_f:
            n_val += 1

    return n_train, n_val


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="e.g. /PATH/TO/EToCVD/D")
    parser.add_argument("--src-splits", type=str, default="splits_30_fricromask",
                        help="name of the source splits folder (augmented version)")
    parser.add_argument("--dst-splits", type=str, default="splits_30_original",
                        help="name of the splits folder to generate (originals only)")
    parser.add_argument("--mode", choices=["csv", "from_split_dir"], default="csv",
                        help="csv: read train_ids.csv (deutan splits_30_fricromask)."
                             "from_split_dir: directly list fold/train/images/ (protan/normal)")
    parser.add_argument("--num-folds", type=int, default=20)

    parser.add_argument("--images-subdir", type=str, default="images")
    parser.add_argument("--saliency-subdir", type=str, default="saliency_maps_30")
    parser.add_argument("--fixation-subdir", type=str, default="fixation_maps")
    args = parser.parse_args()

    root: Path = args.dataset_root.resolve()
    src_root = root / args.src_splits
    dst_root = root / args.dst_splits

    if not src_root.exists():
        raise SystemExit(f"missing: {src_root}")

    if args.mode == "csv":
        images_src = root / args.images_subdir
        saliency_src = root / args.saliency_subdir
        fixation_src = root / args.fixation_subdir
        for p in (images_src, saliency_src, fixation_src):
            if not p.exists():
                raise SystemExit(f"missing (csv mode): {p}")

    dst_root.mkdir(parents=True, exist_ok=True)
    print(f"mode:       {args.mode}")
    print(f"src splits: {src_root}")
    print(f"dst splits: {dst_root}")

    totals_train = 0
    totals_val = 0
    for fold in range(1, args.num_folds + 1):
        src_fold = src_root / f"fold_{fold}"
        dst_fold = dst_root / f"fold_{fold}"
        if not src_fold.exists():
            print(f"[skip] no source fold: {src_fold}")
            continue

        if args.mode == "csv":
            n_t, n_v = build_fold_csv(
                fold, src_fold, dst_fold,
                images_src, saliency_src, fixation_src,
            )
        else:
            n_t, n_v = build_fold_fs(fold, src_fold, dst_fold)
        totals_train += n_t
        totals_val += n_v

    print(f"\n>> done. total train={totals_train}, val={totals_val}")
    print(f">> {dst_root}")


if __name__ == "__main__":
    main()
