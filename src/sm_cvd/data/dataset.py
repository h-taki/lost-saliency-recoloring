"""Dataset for CVD saliency: loads images and group-pooled fixation/saliency targets."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import lightning as L
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from sm_cvd.data.augment import AugmentConfig, apply_augmentations
from sm_cvd.data.transforms import (
    image_to_tensor,
    image_to_tensor_imagenet,
    resize_and_pad,
)






PREPROCESS_MODES = ("msinet", "transalnet")


VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp")



_AUG_SUFFIX_TO_VARIANT = {
    "_hflip": "hflip",
    "_vflip": "vflip",
    "_crop":  "crop",
    "_mask":  "mask",
}


def list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(directory)
    files = [p for p in directory.rglob("*") if p.suffix.lower() in VALID_EXTS]
    if not files:
        raise FileNotFoundError(f"No image files found under {directory}")
    return sorted(files)


def _detect_variant(stem: str) -> str:
    """(documentation omitted in the English release)."""
    for suffix, variant in _AUG_SUFFIX_TO_VARIANT.items():
        if stem.endswith(suffix):
            return variant
    return "original"


def _strip_variant_suffix(stem: str) -> str:
    """(documentation omitted in the English release)."""
    for suffix in _AUG_SUFFIX_TO_VARIANT.keys():
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _filter_by_variant(paths: list[Path], variants: Sequence[str]) -> list[Path]:
    """(documentation omitted in the English release)."""
    allowed = set(variants)
    return [p for p in paths if _detect_variant(p.stem) in allowed]


def _basename_key(p: Path) -> str:
    """(documentation omitted in the English release)."""
    name = p.stem
    name = name.replace("_fixMap", "").replace("_fixPts", "")
    return name


def _check_triplet_consistency(xs: list[Path], ys: list[Path],
                               zs: list[Path] | None = None) -> None:
    n = len(xs)
    if len(ys) != n or (zs is not None and len(zs) != n):
        raise AssertionError(
            f"Mismatched counts: image={len(xs)}, sal={len(ys)}"
            + (f", fix={len(zs)}" if zs is not None else "")
        )
    for i in range(n):
        keys = {_basename_key(xs[i]), _basename_key(ys[i])}
        if zs is not None:
            keys.add(_basename_key(zs[i]))
        if len(keys) != 1:
            raise AssertionError(f"Filename mismatch at index {i}: {keys}")


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
class EToCVDDataset(Dataset):
    """(documentation omitted in the English release)."""

    def __init__(
        self,
        split_dir: str | Path,
        image_size: tuple[int, int] = (240, 320),
        augment: AugmentConfig | None = None,
        return_fixation: bool = False,
        offline_variants: Sequence[str] | None = None,
        preprocess_mode: str = "msinet",
        teacher_saliency_dir: str | Path | None = None,
        teacher_channels: int = 1,
    ) -> None:
        super().__init__()
        self.split_dir = Path(split_dir)
        self.image_size = tuple(image_size)
        self.augment = augment
        self.return_fixation = return_fixation

        if teacher_channels not in (1, 3):
            raise ValueError(f"teacher_channels must be 1 or 3, got {teacher_channels}")
        self.teacher_channels = teacher_channels
        self.offline_variants = (
            tuple(offline_variants) if offline_variants is not None else None
        )
        if preprocess_mode not in PREPROCESS_MODES:
            raise ValueError(
                f"preprocess_mode must be one of {PREPROCESS_MODES}, "
                f"got {preprocess_mode!r}"
            )
        self.preprocess_mode = preprocess_mode



        self.teacher_saliency_dir = (
            Path(teacher_saliency_dir) if teacher_saliency_dir is not None else None
        )

        self._img_files = list_images(self.split_dir / "images")
        self._sal_files = list_images(self.split_dir / "saliency_maps")
        if return_fixation:
            self._fix_files = list_images(self.split_dir / "fixation_maps")
        else:
            self._fix_files = None

        if self.offline_variants is not None:
            self._img_files = _filter_by_variant(self._img_files, self.offline_variants)
            self._sal_files = _filter_by_variant(self._sal_files, self.offline_variants)
            if self._fix_files is not None:
                self._fix_files = _filter_by_variant(self._fix_files, self.offline_variants)

        if return_fixation:
            _check_triplet_consistency(self._img_files, self._sal_files, self._fix_files)
        else:
            _check_triplet_consistency(self._img_files, self._sal_files)

    def __len__(self) -> int:
        return len(self._img_files)

    def __getitem__(self, idx: int) -> dict:
        img_path = self._img_files[idx]
        sal_path = self._sal_files[idx]
        fix_path = self._fix_files[idx] if self._fix_files is not None else None

        img = Image.open(img_path).convert("RGB")
        sal = Image.open(sal_path).convert("L")
        if fix_path is not None:
            fix = Image.open(fix_path).convert("L")
        else:

            fix = Image.new("L", img.size, 0)


        teacher: Image.Image | None = None
        if self.teacher_saliency_dir is not None:
            stem = Path(img_path).stem

            base_stem = _strip_variant_suffix(stem)
            t_path = self.teacher_saliency_dir / f"{base_stem}.png"
            if t_path.exists():
                mode = "RGB" if self.teacher_channels == 3 else "L"
                teacher = Image.open(t_path).convert(mode)


                if teacher.size != img.size:
                    teacher = teacher.resize(img.size, resample=Image.BILINEAR)


        if self.augment is not None and self.offline_variants is None:
            img, sal, fix, teacher = apply_augmentations(
                img, sal, fix, self.augment, extra=teacher,
            )


        if self.preprocess_mode == "msinet":

            img = resize_and_pad(img, self.image_size, is_image=True)
            sal = resize_and_pad(sal, self.image_size, is_image=False)
            fix = resize_and_pad(fix, self.image_size, is_image=False)
            if teacher is not None:
                teacher = resize_and_pad(
                    teacher, self.image_size,
                    is_image=(self.teacher_channels == 3),
                )
        else:
            target_h, target_w = self.image_size
            img = img.resize((target_w, target_h), resample=Image.BICUBIC)
            sal = sal.resize((target_w, target_h), resample=Image.BILINEAR)
            fix = fix.resize((target_w, target_h), resample=Image.NEAREST)
            if teacher is not None:
                teacher = teacher.resize((target_w, target_h), resample=Image.BILINEAR)


        if self.preprocess_mode == "msinet":
            img_t = image_to_tensor(img)               # [3, H, W], float 0-255
        else:
            img_t = image_to_tensor_imagenet(img)
        sal_t = image_to_tensor(sal)                   # [1, H, W], float 0-255

        sample = {
            "image": img_t,
            "saliency": sal_t,
            "image_path": str(img_path),
            "saliency_path": str(sal_path),
        }
        if self.return_fixation:
            fix_t = image_to_tensor(fix)               # [1, H, W], float 0-255
            sample["fixation"] = fix_t
            sample["fixation_path"] = str(fix_path) if fix_path is not None else ""
        if teacher is not None:
            sample["teacher_saliency"] = image_to_tensor(teacher)   # [1, H, W], 0-255
        return sample


# ----------------------------------------------------------------------
# LightningDataModule
# ----------------------------------------------------------------------
@dataclass
class EToCVDPathConfig:
    """(documentation omitted in the English release)."""

    dataset_root: str
    splits_folder: str
    fold: int                  # 1..20

    def fold_dir(self) -> Path:
        return Path(self.dataset_root) / self.splits_folder / f"fold_{self.fold}"


class EToCVDDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_root: str,
        splits_folder: str = "splits_30_original",
        fold: int = 1,
        image_size: tuple[int, int] = (240, 320),
        augment: AugmentConfig | None = None,
        batch_size: int = 1,
        num_workers: int = 2,
        offline_variants: Sequence[str] | None = None,
        preprocess_mode: str = "msinet",
        teacher_saliency_dir: str | None = None,
        teacher_channels: int = 1,
        saliencymix_lam: float | None = None,
        saliencymix_prob: float = 0.5,
        train_return_fixation: bool = False,
    ) -> None:
        super().__init__()
        self.path_cfg = EToCVDPathConfig(
            dataset_root=dataset_root, splits_folder=splits_folder, fold=fold,
        )
        self.image_size = tuple(image_size)
        self.augment = augment
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.offline_variants = (
            tuple(offline_variants) if offline_variants is not None else None
        )
        self.preprocess_mode = preprocess_mode

        self.teacher_saliency_dir = teacher_saliency_dir
        self.teacher_channels = teacher_channels

        self.saliencymix_lam = saliencymix_lam
        self.saliencymix_prob = saliencymix_prob

        self.train_return_fixation = train_return_fixation

        self._train_set: EToCVDDataset | None = None
        self._val_set: EToCVDDataset | None = None
        self._test_set: EToCVDDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        fold_dir = self.path_cfg.fold_dir()
        if stage in (None, "fit", "validate"):
            self._train_set = EToCVDDataset(
                fold_dir / "train",
                image_size=self.image_size,
                augment=self.augment,
                return_fixation=self.train_return_fixation,
                offline_variants=self.offline_variants,
                preprocess_mode=self.preprocess_mode,
                teacher_saliency_dir=self.teacher_saliency_dir,
                teacher_channels=self.teacher_channels,
            )

            if self.saliencymix_lam is not None:
                self._train_set = SaliencyMixDataset(
                    self._train_set,
                    lam=self.saliencymix_lam,
                    prob=self.saliencymix_prob,
                )


            self._val_set = EToCVDDataset(
                fold_dir / "val",
                image_size=self.image_size,
                augment=None,
                return_fixation=True,
                offline_variants=self.offline_variants,
                preprocess_mode=self.preprocess_mode,
                teacher_saliency_dir=self.teacher_saliency_dir,
                teacher_channels=self.teacher_channels,
            )
        if stage in (None, "test", "predict"):
            self._test_set = EToCVDDataset(
                fold_dir / "val",
                image_size=self.image_size,
                augment=None,
                return_fixation=True,
                offline_variants=self.offline_variants,
                preprocess_mode=self.preprocess_mode,
                teacher_saliency_dir=self.teacher_saliency_dir,
                teacher_channels=self.teacher_channels,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_set, batch_size=1, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_set, batch_size=1, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )


class SaliencyMixDataset(Dataset):
    """(documentation omitted in the English release)."""

    def __init__(self, base: Dataset, lam: float = 0.5, prob: float = 0.5,
                 seed: int | None = None) -> None:
        super().__init__()
        if not (0.0 < lam < 1.0):
            raise ValueError(f"lam must be in (0,1), got {lam}")
        self.base = base
        self.lam = float(lam)
        self.prob = float(prob)

        self._rng = random.Random(seed) if seed is not None else random

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def _peak_box(saliency, lam: float) -> tuple[int, int, int, int]:
        """(documentation omitted in the English release)."""
        import torch as _torch
        _, H, W = saliency.shape
        flat_idx = int(_torch.argmax(saliency[0]))
        cy, cx = divmod(flat_idx, W)
        bh = max(1, int(round(H * math.sqrt(lam))))
        bw = max(1, int(round(W * math.sqrt(lam))))
        y0 = min(max(cy - bh // 2, 0), H - bh)
        x0 = min(max(cx - bw // 2, 0), W - bw)
        return y0, y0 + bh, x0, x0 + bw

    def __getitem__(self, idx: int) -> dict:
        sample = self.base[idx]
        if self._rng.random() >= self.prob:
            return sample
        j = self._rng.randrange(len(self.base))
        if j == idx:
            return sample
        other = self.base[j]

        y0, y1, x0, x1 = self._peak_box(other["saliency"], self.lam)

        sample["image"] = sample["image"].clone()
        sample["saliency"] = sample["saliency"].clone()
        sample["image"][:, y0:y1, x0:x1] = other["image"][:, y0:y1, x0:x1]
        sample["saliency"][:, y0:y1, x0:x1] = other["saliency"][:, y0:y1, x0:x1]
        if "teacher_saliency" in sample and "teacher_saliency" in other:
            sample["teacher_saliency"] = sample["teacher_saliency"].clone()
            sample["teacher_saliency"][:, y0:y1, x0:x1] = \
                other["teacher_saliency"][:, y0:y1, x0:x1]
        return sample


class PseudoCvdPretrainDataset(Dataset):
    """(documentation omitted in the English release)."""

    def __init__(
        self,
        images_dir: str | Path,
        target_saliency_dir: str | Path,
        normal_sm_dir: str | Path,
        image_size: tuple[int, int] = (240, 320),
        augment: AugmentConfig | None = None,
    ) -> None:
        super().__init__()
        self.images_dir = Path(images_dir)
        self.target_saliency_dir = Path(target_saliency_dir)
        self.normal_sm_dir = Path(normal_sm_dir)
        self.image_size = tuple(image_size)
        self.augment = augment

        self._img_files = list_images(self.images_dir)

        kept = []
        for f in self._img_files:
            s = f.stem
            t_path = self.target_saliency_dir / f"{s}.png"
            n_path = self.normal_sm_dir / f"{s}.png"
            if t_path.exists() and n_path.exists():
                kept.append(f)
        self._img_files = kept
        if not self._img_files:
            raise FileNotFoundError(
                f"No matching triplet found under "
                f"{self.images_dir} / {self.target_saliency_dir} / {self.normal_sm_dir}"
            )

    def __len__(self) -> int:
        return len(self._img_files)

    def __getitem__(self, idx: int) -> dict:
        img_path = self._img_files[idx]
        stem = img_path.stem
        target_path = self.target_saliency_dir / f"{stem}.png"
        normal_path = self.normal_sm_dir / f"{stem}.png"

        img = Image.open(img_path).convert("RGB")
        target = Image.open(target_path).convert("L")
        normal_sm = Image.open(normal_path).convert("L")


        if target.size != img.size:
            target = target.resize(img.size, resample=Image.BILINEAR)
        if normal_sm.size != img.size:
            normal_sm = normal_sm.resize(img.size, resample=Image.BILINEAR)


        fix_dummy = Image.new("L", img.size, 0)
        if self.augment is not None:
            img, target, fix_dummy, normal_sm = apply_augmentations(
                img, target, fix_dummy, self.augment, extra=normal_sm,
            )


        img = resize_and_pad(img, self.image_size, is_image=True)
        target = resize_and_pad(target, self.image_size, is_image=False)
        normal_sm = resize_and_pad(normal_sm, self.image_size, is_image=False)

        img_t = image_to_tensor(img)
        target_t = image_to_tensor(target)
        normal_sm_t = image_to_tensor(normal_sm)


        fix_dummy_t = target_t.clone()

        return {
            "image": img_t,
            "saliency": target_t,            # pseudo CVD GT = training target
            "teacher_saliency": normal_sm_t, # SM_normal = MSINetV2 input
            "fixation": fix_dummy_t,
            "image_path": str(img_path),
        }


class PseudoCvdPretrainDataModule(L.LightningDataModule):
    """(documentation omitted in the English release)."""

    def __init__(
        self,
        cvd_images_root: str,      # data/salicon_2017_d_cvd/processed/single_split/fold_1
        pseudo_gt_root: str,        # data/salicon_2017_d_cvd/pseudo_cvd_gt
        normal_sm_root: str,        # data/salicon_2017/processed/single_split/fold_1
        image_size: tuple[int, int] = (240, 320),
        augment: AugmentConfig | None = None,
        batch_size: int = 4,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.cvd_images_root = Path(cvd_images_root)
        self.pseudo_gt_root = Path(pseudo_gt_root)
        self.normal_sm_root = Path(normal_sm_root)
        self.image_size = tuple(image_size)
        self.augment = augment
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train_set: PseudoCvdPretrainDataset | None = None
        self._val_set: PseudoCvdPretrainDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit", "validate"):
            self._train_set = PseudoCvdPretrainDataset(
                images_dir=self.cvd_images_root / "train" / "images",
                target_saliency_dir=self.pseudo_gt_root / "train",
                normal_sm_dir=self.normal_sm_root / "train" / "saliency_maps",
                image_size=self.image_size,
                augment=self.augment,
            )
            self._val_set = PseudoCvdPretrainDataset(
                images_dir=self.cvd_images_root / "val" / "images",
                target_saliency_dir=self.pseudo_gt_root / "val",
                normal_sm_dir=self.normal_sm_root / "val" / "saliency_maps",
                image_size=self.image_size,
                augment=None,
            )
            print(f"[PseudoCvdPretrainDataModule] train samples = {len(self._train_set)}, "
                  f"val samples = {len(self._val_set)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_set, batch_size=1, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )


class MultiCVDDataModule(L.LightningDataModule):
    """(documentation omitted in the English release)."""

    def __init__(
        self,
        dataset_roots: dict[str, str],
        train_cvd_types: Sequence[str] = ("D", "P"),
        val_cvd_type: str = "D",
        splits_folders: dict[str, str] | str = "splits_30_original",
        train_offline_variants_per_type: dict[str, Sequence[str] | None] | None = None,
        fold: int = 1,
        image_size: tuple[int, int] = (240, 320),
        augment: AugmentConfig | None = None,
        batch_size: int = 1,
        num_workers: int = 2,
        preprocess_mode: str = "msinet",
    ) -> None:
        super().__init__()
        if not dataset_roots:
            raise ValueError("dataset_roots must not be empty")
        unknown = [t for t in train_cvd_types if t not in dataset_roots]
        if unknown:
            raise ValueError(
                f"train_cvd_types contains keys not in dataset_roots: {unknown}"
            )
        if val_cvd_type not in dataset_roots:
            raise ValueError(
                f"val_cvd_type={val_cvd_type!r} not in dataset_roots keys "
                f"{list(dataset_roots)}"
            )

        self.dataset_roots = {k: str(v) for k, v in dataset_roots.items()}
        self.train_cvd_types = tuple(train_cvd_types)
        self.val_cvd_type = val_cvd_type


        if isinstance(splits_folders, str):
            self.splits_folders: dict[str, str] = {t: splits_folders for t in dataset_roots}
        else:
            self.splits_folders = dict(splits_folders)
            for t in dataset_roots:
                self.splits_folders.setdefault(t, "splits_30_original")


        self.train_offline_variants_per_type: dict[str, tuple[str, ...] | None] = {}
        if train_offline_variants_per_type is not None:
            for t, v in train_offline_variants_per_type.items():
                self.train_offline_variants_per_type[t] = tuple(v) if v is not None else None
        for t in dataset_roots:
            self.train_offline_variants_per_type.setdefault(t, None)

        self.fold = fold
        self.image_size = tuple(image_size)
        self.augment = augment
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocess_mode = preprocess_mode

        self._train_set: ConcatDataset | None = None
        self._val_set: EToCVDDataset | None = None
        self._test_set: EToCVDDataset | None = None

    def _fold_dir(self, cvd_type: str) -> Path:
        return (
            Path(self.dataset_roots[cvd_type])
            / self.splits_folders[cvd_type]
            / f"fold_{self.fold}"
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit", "validate"):
            train_subsets: list[EToCVDDataset] = []
            for t in self.train_cvd_types:
                fd = self._fold_dir(t)
                variants = self.train_offline_variants_per_type[t]
                ds = EToCVDDataset(
                    fd / "train",
                    image_size=self.image_size,
                    augment=self.augment,
                    return_fixation=False,
                    offline_variants=variants,
                    preprocess_mode=self.preprocess_mode,
                )
                train_subsets.append(ds)
                print(
                    f"[MultiCVDDataModule] train cvd={t} fold={self.fold} "
                    f"samples={len(ds)} "
                    f"splits={self.splits_folders[t]} "
                    f"variants={variants} "
                    f"root={self.dataset_roots[t]}"
                )
            self._train_set = ConcatDataset(train_subsets)
            print(
                f"[MultiCVDDataModule] train total samples = {len(self._train_set)} "
                f"(types={list(self.train_cvd_types)})"
            )

            val_fd = self._fold_dir(self.val_cvd_type)
            self._val_set = EToCVDDataset(
                val_fd / "val",
                image_size=self.image_size,
                augment=None,
                return_fixation=True,
                offline_variants=None,
                preprocess_mode=self.preprocess_mode,
            )
            print(
                f"[MultiCVDDataModule] val cvd={self.val_cvd_type} fold={self.fold} "
                f"samples={len(self._val_set)}"
            )

        if stage in (None, "test", "predict"):
            test_fd = self._fold_dir(self.val_cvd_type)
            self._test_set = EToCVDDataset(
                test_fd / "val",
                image_size=self.image_size,
                augment=None,
                return_fixation=True,
                offline_variants=None,
                preprocess_mode=self.preprocess_mode,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_set, batch_size=1, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_set, batch_size=1, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )


__all__ = [
    "EToCVDDataset", "EToCVDDataModule", "EToCVDPathConfig",
    "MultiCVDDataModule",
    "PseudoCvdPretrainDataset", "PseudoCvdPretrainDataModule",
    "list_images",
]
