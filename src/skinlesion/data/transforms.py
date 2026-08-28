"""Albumentations pipelines for dermoscopy.

Design notes
------------
* Dermoscopic images are colour-critical (pigment network, blue-white veil), so
  we keep colour jitter mild and rely more on geometric augmentation.
* Lesions have no canonical orientation -> full dihedral (flip + 90-degree
  rotation) group is safe and effective.
* ``CoarseDropout`` acts as a lightweight occlusion / cutout regulariser.
* Evaluation pipeline is deterministic; TTA is composed separately in
  :mod:`skinlesion.serve.inference`.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics (backbones are ImageNet-pretrained).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(image_size: int, *, strength: str = "medium") -> A.Compose:
    s = {"light": 0.5, "medium": 1.0, "heavy": 1.5}[strength]
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.75, 1.0),
                ratio=(0.9, 1.1),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(1 - 0.1 * s, 1 + 0.1 * s),
                translate_percent=(0.0, 0.05 * s),
                rotate=(-20 * s, 20 * s),
                shear=(-8 * s, 8 * s),
                p=0.7,
            ),
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.1 * s,
                        contrast=0.1 * s,
                        saturation=0.1 * s,
                        hue=0.02 * s,
                    ),
                    A.RandomBrightnessContrast(brightness_limit=0.1 * s, contrast_limit=0.1 * s),
                ],
                p=0.5,
            ),
            A.GaussNoise(p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                fill=0,
                p=0.3 * min(s, 1.0),
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def eval_transform(image_size: int) -> A.Compose:
    resize = round(image_size * 1.14)  # standard 224 -> 256 style padding
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=resize),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def tta_transforms(image_size: int) -> list[A.Compose]:
    """A small deterministic-ish TTA bank (identity + dihedral flips)."""
    base = [
        A.SmallestMaxSize(max_size=round(image_size * 1.14)),
        A.CenterCrop(height=image_size, width=image_size),
    ]
    tail = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    variants = [
        [],
        [A.HorizontalFlip(p=1.0)],
        [A.VerticalFlip(p=1.0)],
        [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)],
    ]
    return [A.Compose(base + v + tail) for v in variants]
