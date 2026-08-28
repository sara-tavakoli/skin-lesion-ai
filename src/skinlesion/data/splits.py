"""Leakage-free dataset splitting for HAM10000.

HAM10000 contains multiple dermoscopic images of the *same physical lesion*
(shared ``lesion_id``).  A naive random split leaks near-duplicate views of a
lesion across train/val/test and massively inflates reported performance.  We
therefore split on ``lesion_id`` with stratification on the lesion's class,
using scikit-learn's ``StratifiedGroupKFold``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class SplitConfig:
    n_folds: int = 5
    test_fold: int = 0
    val_fold: int = 1
    seed: int = 1337


def assign_folds(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """Return ``df`` with an added integer ``fold`` column in ``[0, n_folds)``.

    Rows sharing a ``lesion_id`` always land in the same fold.
    """
    required = {"image_id", "lesion_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"metadata missing columns: {sorted(missing)}")

    df = df.reset_index(drop=True).copy()
    y = df["dx"].to_numpy()
    groups = df["lesion_id"].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold = np.full(len(df), -1, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(df, y, groups)):
        fold[val_idx] = fold_idx
    if (fold < 0).any():  # pragma: no cover - defensive
        raise RuntimeError("some rows were not assigned to a fold")
    df["fold"] = fold
    _assert_no_group_leakage(df)
    return df


def split_frames(df: pd.DataFrame, cfg: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Materialise (train, val, test) frames from a fold-annotated frame."""
    if "fold" not in df.columns:
        df = assign_folds(df, cfg)
    test = df[df["fold"] == cfg.test_fold]
    val = df[df["fold"] == cfg.val_fold]
    train = df[~df["fold"].isin({cfg.test_fold, cfg.val_fold})]
    for name, part in (("train", train), ("val", val), ("test", test)):
        if part.empty:
            raise RuntimeError(f"{name} split is empty; check SplitConfig")
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _assert_no_group_leakage(df: pd.DataFrame) -> None:
    per_group_folds = df.groupby("lesion_id")["fold"].nunique()
    leaked = per_group_folds[per_group_folds > 1]
    if len(leaked):  # pragma: no cover - defensive
        raise RuntimeError(f"{len(leaked)} lesion_id(s) span multiple folds: {list(leaked.index[:5])}")


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per-split class counts + proportions, handy for the dataset card."""
    tab = (
        df.groupby(["fold", "dx"]).size().rename("n").reset_index()
        if "fold" in df.columns
        else df.groupby("dx").size().rename("n").reset_index()
    )
    denom = tab.groupby("fold")["n"].transform("sum") if "fold" in tab else tab["n"].sum()
    tab["frac"] = tab["n"] / denom
    return tab
