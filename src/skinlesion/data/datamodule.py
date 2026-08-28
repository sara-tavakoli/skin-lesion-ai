"""LightningDataModule tying together metadata, splits, and loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler

from skinlesion.data.dataset import LesionDataset
from skinlesion.data.splits import SplitConfig, assign_folds, split_frames
from skinlesion.data.transforms import eval_transform, train_transform
from skinlesion.utils.seed import worker_init_fn


class HAM10000DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        metadata_csv: str = "metadata.csv",
        image_subdir: str = "images",
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        aug_strength: str = "medium",
        balanced_sampler: bool = True,
        n_folds: int = 5,
        test_fold: int = 0,
        val_fold: int = 1,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()  # recorded in checkpoints for provenance

        self.data_dir = Path(data_dir)
        self.metadata_csv = metadata_csv
        self.image_subdir = image_subdir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.aug_strength = aug_strength
        self.balanced_sampler = balanced_sampler
        self.split_cfg = SplitConfig(n_folds=n_folds, test_fold=test_fold, val_fold=val_fold, seed=seed)

        self._train_ds: LesionDataset | None = None
        self._val_ds: LesionDataset | None = None
        self._test_ds: LesionDataset | None = None
        self.class_weights: torch.Tensor | None = None

    # -- lifecycle ------------------------------------------------------------
    def prepare_data(self) -> None:
        meta = self.data_dir / self.metadata_csv
        if not meta.exists():
            raise FileNotFoundError(
                f"{meta} not found. Run scripts/download_ham10000.py or scripts/make_synthetic_data.py first."
            )

    def setup(self, stage: str | None = None) -> None:
        df = pd.read_csv(self.data_dir / self.metadata_csv)
        df["lesion_id"] = df.get("lesion_id", df["image_id"]).fillna(df["image_id"])
        df = assign_folds(df, self.split_cfg)
        train_df, val_df, test_df = split_frames(df, self.split_cfg)

        image_root = self.data_dir / self.image_subdir
        tf_train = train_transform(self.image_size, strength=self.aug_strength)
        tf_eval = eval_transform(self.image_size)

        self._train_ds = LesionDataset(train_df, image_root, tf_train)
        self._val_ds = LesionDataset(val_df, image_root, tf_eval)
        self._test_ds = LesionDataset(test_df, image_root, tf_eval)

        counts = self._train_ds.class_counts().float()
        # Effective-number-of-samples class weights (Cui et al., 2019).
        beta = 0.999
        eff_num = 1.0 - torch.pow(torch.tensor(beta), counts.clamp(min=1))
        w = (1.0 - beta) / eff_num
        self.class_weights = (w / w.sum() * len(counts)).float()

    # -- loaders -----------------------------------------------------------
    def _loader(self, ds: LesionDataset, shuffle: bool, sampler=None) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=worker_init_fn,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None
        sampler = None
        if self.balanced_sampler:
            weights = self._train_ds.sample_weights().tolist()
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return self._loader(self._train_ds, shuffle=True, sampler=sampler)

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None
        return self._loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self._test_ds is not None
        return self._loader(self._test_ds, shuffle=False)
