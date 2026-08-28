"""Torch ``Dataset`` over a HAM10000-style metadata frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from skinlesion import CLASSES

_CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class LesionDataset(Dataset):
    """Yields ``(image_tensor, label, meta)`` where ``meta`` carries ids.

    Parameters
    ----------
    frame:
        Must contain ``image_id`` and ``dx``; ``filepath`` is used if present,
        otherwise ``<image_root>/<image_id>.jpg``.
    image_root:
        Directory containing the images (ignored when ``filepath`` is set).
    transform:
        An albumentations ``Compose`` operating on ``image=np.ndarray``.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path,
        transform,
        *,
        return_meta: bool = True,
    ) -> None:
        if "dx" not in frame.columns or "image_id" not in frame.columns:
            raise KeyError("frame needs 'image_id' and 'dx' columns")
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.return_meta = return_meta
        self.labels = self.frame["dx"].map(_CLASS_TO_IDX).to_numpy()
        if np.isnan(self.labels.astype(float)).any():
            bad = sorted(set(self.frame["dx"]) - set(CLASSES))
            raise ValueError(f"unknown dx values: {bad}")

    def __len__(self) -> int:
        return len(self.frame)

    def _path(self, row: pd.Series) -> Path:
        if "filepath" in row and isinstance(row["filepath"], str) and row["filepath"]:
            return Path(row["filepath"])
        return self.image_root / f"{row['image_id']}.jpg"

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        path = self._path(row)
        with Image.open(path) as im:
            image = np.asarray(im.convert("RGB"))
        image = self.transform(image=image)["image"]
        label = int(self.labels[idx])
        if not self.return_meta:
            return image, label
        meta = {
            "image_id": str(row["image_id"]),
            "lesion_id": str(row.get("lesion_id", row["image_id"])),
            "dx": str(row["dx"]),
        }
        return image, label, meta

    # -- helpers used by the datamodule -----------------------------------
    def class_counts(self) -> torch.Tensor:
        counts = np.bincount(self.labels, minlength=len(CLASSES))
        return torch.as_tensor(counts, dtype=torch.long)

    def sample_weights(self) -> torch.Tensor:
        """Inverse-frequency weight per sample for a WeightedRandomSampler."""
        counts = self.class_counts().clamp(min=1).float()
        per_class_w = 1.0 / counts
        w = per_class_w[self.labels]
        return w * (len(w) / w.sum())
