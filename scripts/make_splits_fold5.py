"""Build the 5-fold cross-validation splits."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def kfold_indices(n: int, n_splits: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """(documentation omitted in the English release)."""
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1

    out: list[tuple[list[int], list[int]]] = []
    current = 0
    for size in fold_sizes:
        val_idx = indices[current : current + size]
        train_idx = np.concatenate([indices[:current], indices[current + size:]])
        out.append((train_idx.tolist(), val_idx.tolist()))
        current += size
    return out



AUG_SUFFIXES = ("_hflip", "_vflip", "_crop", "_mask")


def is_original(stem: str) -> bool:
    return not any(stem.endswith(s) for s in AUG_SUFFIXES)


def list_image_files(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def collect_originals_from_fold1(src_root: Path) -> dict[str, dict[str, Path]]:
    """(documentation omitted in the English release)."""
    fold1 = src_root / "fold_1"
    if not fold1.exists():
        raise SystemExit(f"missing: {fold1}")

    pool: dict[str, dict[str, Path]] = {}
    for split in ("train", "val"):
        img_dir = fold1 / split / "images"
        sal_dir = fold1 / split / "saliency_maps"
        fix_dir = fold1 / split / "fixation_maps"
        imgs = list_image_files(img_dir)
        sals = list_image_files(sal_dir)
        fixs = list_image_files(fix_dir)

        sal_by_stem = {p.stem: p for p in sals}
        fix_by_stem = {p.stem: p for p in fixs}

        for img in imgs:
            stem = img.stem
            if not is_original(stem):
                continue
            if stem in pool:

                raise AssertionError(
                    f"duplicate stem across train/val in fold_1: {stem}"
                )
            if stem not in sal_by_stem:
                raise AssertionError(
                    f"no saliency for {stem} in {sal_dir}"
                )
            if stem not in fix_by_stem:
                raise AssertionError(
                    f"no fixation for {stem} in {fix_dir}"
                )
            pool[stem] = {
                "image": img,
                "saliency": sal_by_stem[stem],
                "fixation": fix_by_stem[stem],
            }
    return pool


def cross_check_with_all_folds(src_root: Path, expected: set[str], num_src_folds: int) -> None:
    """(documentation omitted in the English release)."""
    for k in range(1, num_src_folds + 1):
        fold = src_root / f"fold_{k}"
        if not fold.exists():
            continue
        stems: set[str] = set()
        for split in ("train", "val"):
            img_dir = fold / split / "images"
            for p in list_image_files(img_dir):
                if is_original(p.stem):
                    stems.add(p.stem)
        if stems != expected:
            extra = stems - expected
            missing = expected - stems
            raise AssertionError(
                f"fold_{k} originals != fold_1 originals. "
                f"extra={sorted(extra)[:5]} missing={sorted(missing)[:5]}"
            )


def copy_triplet(triplet: dict[str, Path], dst_fold_dir: Path, split: str) -> None:
    """(documentation omitted in the English release)."""
    targets = {
        "image": dst_fold_dir / split / "images",
        "saliency": dst_fold_dir / split / "saliency_maps",
        "fixation": dst_fold_dir / split / "fixation_maps",
    }
    for k, src in triplet.items():
        dst_dir = targets[k]
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help='e.g. "/PATH/TO/EToCVD/D"',
    )
    parser.add_argument(
        "--src-splits", type=str, default="splits_30_original",
        help="name of the existing Fold=20 splits folder (originals only).",
    )
    parser.add_argument(
        "--dst-splits", type=str, default="splits_fold5",
        help="name of the new Fold=5 splits folder.",
    )
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--num-src-folds", type=int, default=20,
                        help="number of folds in the source splits (for cross-checking).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="do not copy files; only print the fold assignment.")
    args = parser.parse_args()

    root: Path = args.dataset_root.resolve()
    src_root = root / args.src_splits
    dst_root = root / args.dst_splits

    if not src_root.exists():
        raise SystemExit(f"missing src: {src_root}")
    if dst_root.exists():
        raise SystemExit(
            f"dst already exists: {dst_root}. "
            f"Remove or rename it first (idempotency guard)."
        )

    print(f"src splits: {src_root}")
    print(f"dst splits: {dst_root}")
    print(f"seed:       {args.seed}")
    print(f"num folds:  {args.num_folds}")
    print()


    pool = collect_originals_from_fold1(src_root)
    stems = sorted(pool.keys())
    print(f"[step 1] originals collected from {args.src_splits}/fold_1 : {len(stems)}")
    if len(stems) != 101:
        print(f"  [warn] expected 101 originals, got {len(stems)}. proceeding anyway.")


    cross_check_with_all_folds(src_root, set(stems), args.num_src_folds)
    print(f"[step 2] cross-check passed: all {args.num_src_folds} src folds "
          f"have the same originals set.")


    splits = kfold_indices(len(stems), args.num_folds, args.seed)
    fold_assignments: list[dict] = []
    for k, (train_idx, val_idx) in enumerate(splits, start=1):
        train_stems = [stems[i] for i in train_idx]
        val_stems = [stems[i] for i in val_idx]
        fold_assignments.append({
            "fold": k,
            "train": train_stems,
            "val": val_stems,
        })
        print(f"[fold {k}] train={len(train_stems)} val={len(val_stems)}")


    union_val: set[str] = set()
    for a in fold_assignments:
        overlap = union_val & set(a["val"])
        if overlap:
            raise AssertionError(
                f"val overlap across folds: {sorted(overlap)[:5]}"
            )
        union_val |= set(a["val"])
    if union_val != set(stems):
        missing = set(stems) - union_val
        extra = union_val - set(stems)
        raise AssertionError(
            f"disjoint union of val != all stems. "
            f"missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}"
        )
    print(f"[step 3] disjoint union of val == all {len(stems)} originals. OK.")
    print()

    if args.dry_run:
        print("[dry-run] no files copied. exiting.")
        return


    dst_root.mkdir(parents=True, exist_ok=False)
    for a in fold_assignments:
        k = a["fold"]
        dst_fold = dst_root / f"fold_{k}"
        print(f"[copy] fold_{k}: train -> {dst_fold}/train, val -> {dst_fold}/val")
        for stem in a["train"]:
            copy_triplet(pool[stem], dst_fold, "train")
        for stem in a["val"]:
            copy_triplet(pool[stem], dst_fold, "val")


    manifest = {
        "seed": args.seed,
        "num_folds": args.num_folds,
        "num_originals": len(stems),
        "src_splits": args.src_splits,
        "dst_splits": args.dst_splits,
        "folds": fold_assignments,
    }
    (dst_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print()
    print(f"[done] wrote manifest: {dst_root / 'manifest.json'}")
    print(f"[done] new splits ready under {dst_root}")


if __name__ == "__main__":
    main()
