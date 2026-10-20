"""Image and target transforms."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image


def resize_keep_aspect(
    image: Image.Image,
    target_size: tuple[int, int],
    overfull: bool = False,
    *,
    is_image: bool = True,
) -> Image.Image:
    """(documentation omitted in the English release)."""
    target_h, target_w = target_size
    cur_w, cur_h = image.size
    ratio_h = target_h / cur_h
    ratio_w = target_w / cur_w
    ratio = max(ratio_h, ratio_w) if overfull else min(ratio_h, ratio_w)

    new_w = int(round(cur_w * ratio))
    new_h = int(round(cur_h * ratio))

    if (new_w, new_h) == image.size:
        return image

    resample = Image.BICUBIC if is_image else Image.BILINEAR
    return image.resize((new_w, new_h), resample=resample)


def pad_to_size(image: Image.Image, target_size: tuple[int, int],
                pad_value: int) -> Image.Image:
    """(documentation omitted in the English release)."""
    target_h, target_w = target_size
    cur_w, cur_h = image.size
    if (cur_w, cur_h) == (target_w, target_h):
        return image

    if image.mode == "RGB":
        canvas = Image.new("RGB", (target_w, target_h),
                           (pad_value, pad_value, pad_value))
    else:
        canvas = Image.new(image.mode, (target_w, target_h), pad_value)

    left = (target_w - cur_w) // 2
    top = (target_h - cur_h) // 2
    canvas.paste(image, (left, top))
    return canvas


def resize_and_pad(image: Image.Image, target_size: tuple[int, int],
                   *, is_image: bool) -> Image.Image:
    """(documentation omitted in the English release)."""
    image = resize_keep_aspect(image, target_size, overfull=False, is_image=is_image)
    pad_value = 126 if is_image else 0
    return pad_to_size(image, target_size, pad_value=pad_value)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """(documentation omitted in the English release)."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    # HWC → CHW
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()



IMAGENET_MEAN_BGR: tuple[float, float, float] = (103.939, 116.779, 123.68)


IMAGENET_MEAN_RGB: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD_RGB:  tuple[float, float, float] = (0.229, 0.224, 0.225)


def preprocess_for_vgg(image_chw_rgb_255: torch.Tensor) -> torch.Tensor:
    """(documentation omitted in the English release)."""
    mean = torch.tensor(IMAGENET_MEAN_BGR, dtype=image_chw_rgb_255.dtype,
                        device=image_chw_rgb_255.device).view(3, 1, 1)
    return image_chw_rgb_255 - mean


def image_to_tensor_imagenet(image: Image.Image) -> torch.Tensor:
    """(documentation omitted in the English release)."""
    arr = np.asarray(image, dtype=np.float32) / 255.0   # 0-1 range
    if arr.ndim == 2:
        arr = arr[..., None]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(IMAGENET_MEAN_RGB, dtype=t.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD_RGB,  dtype=t.dtype).view(3, 1, 1)
    return (t - mean) / std


__all__ = [
    "resize_keep_aspect",
    "pad_to_size",
    "resize_and_pad",
    "image_to_tensor",
    "image_to_tensor_imagenet",
    "preprocess_for_vgg",
    "IMAGENET_MEAN_BGR",
    "IMAGENET_MEAN_RGB",
    "IMAGENET_STD_RGB",
]
