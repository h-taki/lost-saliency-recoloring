"""Online data augmentation (horizontal flip, random-resized-crop, random-erasing)."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from PIL import Image
from torchvision.transforms import functional as TF


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def get_params_for_random_resized_crop(
    img: Image.Image,
    scale_range: Sequence[float],
    ratio_range: Sequence[float],
) -> tuple[int, int, int, int]:
    """(documentation omitted in the English release)."""
    W, H = img.size
    area = W * H

    for _ in range(10):
        target_area = random.uniform(scale_range[0], scale_range[1]) * area
        log_ratio = (math.log(ratio_range[0]), math.log(ratio_range[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= W and 0 < h <= H:
            i = random.randint(0, H - h)
            j = random.randint(0, W - w)
            return i, j, h, w


    in_ratio = W / H
    if in_ratio < min(ratio_range):
        w = W
        h = int(round(w / min(ratio_range)))
    elif in_ratio > max(ratio_range):
        h = H
        w = int(round(h * max(ratio_range)))
    else:
        w = min(W, H)
        h = min(W, H)
    i = (H - h) // 2
    j = (W - w) // 2
    return i, j, h, w


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def hflip_triplet(img: Image.Image, sal: Image.Image, fix: Image.Image
                  ) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Horizontal Flip (HF)."""
    return (
        img.transpose(Image.FLIP_LEFT_RIGHT),
        sal.transpose(Image.FLIP_LEFT_RIGHT),
        fix.transpose(Image.FLIP_LEFT_RIGHT),
    )


def vflip_triplet(img: Image.Image, sal: Image.Image, fix: Image.Image
                  ) -> tuple[Image.Image, Image.Image, Image.Image]:
    """(documentation omitted in the English release)."""
    return (
        img.transpose(Image.FLIP_TOP_BOTTOM),
        sal.transpose(Image.FLIP_TOP_BOTTOM),
        fix.transpose(Image.FLIP_TOP_BOTTOM),
    )


def random_resized_crop_triplet(
    img: Image.Image,
    sal: Image.Image,
    fix: Image.Image,
    scale: Sequence[float] = (0.8, 1.0),
    ratio: Sequence[float] = (3 / 4, 4 / 3),
    target_size: tuple[int, int] | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """(documentation omitted in the English release)."""
    W, H = img.size
    if target_size is None:
        target_size = (W, H)

    i, j, h, w = get_params_for_random_resized_crop(img, scale, ratio)
    box = (j, i, j + w, i + h)
    img_c = img.crop(box).resize(target_size, resample=Image.BICUBIC)
    sal_c = sal.crop(box).resize(target_size, resample=Image.BILINEAR)
    fix_c = fix.crop(box).resize(target_size, resample=Image.NEAREST)
    return img_c, sal_c, fix_c


def color_jitter_triplet(
    img: Image.Image,
    sal: Image.Image,
    fix: Image.Image,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """(documentation omitted in the English release)."""
    if brightness > 0:
        bf = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
        img = TF.adjust_brightness(img, bf)
    if contrast > 0:
        cf = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
        img = TF.adjust_contrast(img, cf)
    if saturation > 0:
        sf = random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)
        img = TF.adjust_saturation(img, sf)
    if hue > 0:
        hf = random.uniform(-hue, hue)
        img = TF.adjust_hue(img, hf)
    return img, sal, fix


def random_erasing_triplet(
    img: Image.Image,
    sal: Image.Image,
    fix: Image.Image,
    scale: Sequence[float] = (0.05, 0.18),
    ratio: Sequence[float] = (0.2, 5.0),
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """(documentation omitted in the English release)."""
    W, H = img.size
    i, j, h, w = get_params_for_random_resized_crop(img, scale, ratio)

    mask = Image.new("L", (W, H), 255)
    mask.paste(Image.new("L", (w, h), 0), (j, i))

    img_m = Image.composite(img, Image.new("RGB", img.size, (0, 0, 0)), mask)
    sal_m = Image.composite(sal, Image.new("L", sal.size, 0), mask)
    fix_m = Image.composite(fix, Image.new("L", fix.size, 0), mask)
    return img_m, sal_m, fix_m


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
@dataclass
class AugmentConfig:
    """(documentation omitted in the English release)."""

    use_hf: bool = True
    use_rrc: bool = False
    use_re: bool = False
    use_color_jitter: bool = False
    rrc_scale: tuple[float, float] = (0.8, 1.0)
    rrc_ratio: tuple[float, float] = (3 / 4, 4 / 3)
    re_scale: tuple[float, float] = (0.05, 0.18)
    re_ratio: tuple[float, float] = (0.2, 5.0)

    cj_brightness: float = 0.0
    cj_contrast: float = 0.0
    cj_saturation: float = 0.0
    cj_hue: float = 0.0
    prob_each: float = 0.5


def apply_augmentations(
    img: Image.Image,
    sal: Image.Image,
    fix: Image.Image,
    cfg: AugmentConfig,
    extra: Image.Image | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image | None]:
    """(documentation omitted in the English release)."""
    if cfg.use_hf and random.random() < cfg.prob_each:
        img, sal, fix = hflip_triplet(img, sal, fix)
        if extra is not None:
            extra = extra.transpose(Image.FLIP_LEFT_RIGHT)
    if cfg.use_rrc and random.random() < cfg.prob_each:

        W, H = img.size
        i, j, h, w = get_params_for_random_resized_crop(img, cfg.rrc_scale, cfg.rrc_ratio)
        box = (j, i, j + w, i + h)
        img = img.crop(box).resize((W, H), resample=Image.BICUBIC)
        sal = sal.crop(box).resize((W, H), resample=Image.BILINEAR)
        fix = fix.crop(box).resize((W, H), resample=Image.NEAREST)
        if extra is not None:
            extra = extra.crop(box).resize((W, H), resample=Image.BILINEAR)
    if cfg.use_re and random.random() < cfg.prob_each:
        W, H = img.size
        i, j, h, w = get_params_for_random_resized_crop(img, cfg.re_scale, cfg.re_ratio)
        mask = Image.new("L", (W, H), 255)
        mask.paste(Image.new("L", (w, h), 0), (j, i))
        img = Image.composite(img, Image.new("RGB", img.size, (0, 0, 0)), mask)
        sal = Image.composite(sal, Image.new("L", sal.size, 0), mask)
        fix = Image.composite(fix, Image.new("L", fix.size, 0), mask)
        if extra is not None:
            extra = Image.composite(extra, Image.new("L", extra.size, 0), mask)
    if cfg.use_color_jitter and random.random() < cfg.prob_each:
        img, sal, fix = color_jitter_triplet(
            img, sal, fix,
            brightness=cfg.cj_brightness,
            contrast=cfg.cj_contrast,
            saturation=cfg.cj_saturation,
            hue=cfg.cj_hue,
        )

    return img, sal, fix, extra


__all__ = [
    "AugmentConfig",
    "apply_augmentations",
    "hflip_triplet",
    "vflip_triplet",
    "random_resized_crop_triplet",
    "random_erasing_triplet",
    "color_jitter_triplet",
    "get_params_for_random_resized_crop",
]
