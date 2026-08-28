from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from skinlesion.data.dataset import LesionDataset
from skinlesion.data.splits import SplitConfig, assign_folds, split_frames
from skinlesion.data.transforms import eval_transform, train_transform


def test_folds_have_no_lesion_leakage(dummy_metadata):
    cfg = SplitConfig(n_folds=5, seed=0)
    df = assign_folds(dummy_metadata, cfg)
    per_group = df.groupby("lesion_id")["fold"].nunique()
    assert (per_group == 1).all()


def test_split_frames_are_disjoint_and_nonempty(dummy_metadata):
    cfg = SplitConfig(n_folds=5, test_fold=0, val_fold=1, seed=0)
    train, val, test = split_frames(dummy_metadata, cfg)
    assert len(train) and len(val) and len(test)
    ids = [set(p["lesion_id"]) for p in (train, val, test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])


def test_split_is_deterministic(dummy_metadata):
    cfg = SplitConfig(n_folds=5, seed=123)
    a = assign_folds(dummy_metadata, cfg)["fold"].to_numpy()
    b = assign_folds(dummy_metadata, cfg)["fold"].to_numpy()
    np.testing.assert_array_equal(a, b)


def test_missing_columns_raise():
    with pytest.raises(KeyError):
        assign_folds(pd.DataFrame({"image_id": ["a"]}), SplitConfig())


def test_dataset_returns_tensor_and_label(synthetic_data):
    df = pd.read_csv(synthetic_data / "metadata.csv")
    ds = LesionDataset(df, synthetic_data / "images", eval_transform(48))
    img, label, meta = ds[0]
    assert img.shape == (3, 48, 48)
    assert 0 <= label < 7
    assert set(meta) == {"image_id", "lesion_id", "dx"}


def test_sample_weights_balance_classes(synthetic_data):
    df = pd.read_csv(synthetic_data / "metadata.csv")
    ds = LesionDataset(df, synthetic_data / "images", train_transform(48))
    w = ds.sample_weights()
    assert w.shape[0] == len(ds)
    # inverse-frequency: rarer class -> larger mean weight
    by_class = {}
    for i, lab in enumerate(ds.labels):
        by_class.setdefault(int(lab), []).append(float(w[i]))
    counts = ds.class_counts().tolist()
    means = [np.mean(by_class[c]) for c in sorted(by_class)]
    # classes with fewer samples should not have smaller weight
    order = np.argsort(counts)
    assert means[order[0]] >= means[order[-1]] - 1e-6


def test_train_transform_is_stochastic(synthetic_data):
    df = pd.read_csv(synthetic_data / "metadata.csv")
    with Image.open(synthetic_data / "images" / f"{df.iloc[0]['image_id']}.jpg") as im:
        arr = np.asarray(im.convert("RGB"))
    tf = train_transform(48)
    a = tf(image=arr)["image"]
    b = tf(image=arr)["image"]
    assert not np.allclose(a.numpy(), b.numpy())
