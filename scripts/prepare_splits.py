#!/usr/bin/env python
"""Assign leakage-free folds and write a dataset summary.

Outputs (next to ``metadata.csv``):
    * ``splits.csv``            -- image_id, lesion_id, dx, fold
    * ``split_summary.json``    -- per-fold class counts + leakage check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from skinlesion.data.splits import SplitConfig, assign_folds, split_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/ham10000"))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--test-fold", type=int, default=0)
    ap.add_argument("--val-fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    df = pd.read_csv(args.data_dir / "metadata.csv")
    df["lesion_id"] = df.get("lesion_id", df["image_id"]).fillna(df["image_id"])

    cfg = SplitConfig(args.n_folds, args.test_fold, args.val_fold, args.seed)
    df = assign_folds(df, cfg)
    train_df, val_df, test_df = split_frames(df, cfg)

    df[["image_id", "lesion_id", "dx", "fold"]].to_csv(args.data_dir / "splits.csv", index=False)

    summary = {
        "config": cfg.__dict__,
        "n_images": len(df),
        "n_lesions": int(df["lesion_id"].nunique()),
        "splits": {
            name: {
                "n_images": len(part),
                "n_lesions": int(part["lesion_id"].nunique()),
                "class_counts": part["dx"].value_counts().sort_index().to_dict(),
            }
            for name, part in (("train", train_df), ("val", val_df), ("test", test_df))
        },
        "lesion_overlap_between_splits": _overlap(train_df, val_df, test_df),
    }
    (args.data_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["splits"], indent=2))
    print("leakage check (should all be 0):", summary["lesion_overlap_between_splits"])


def _overlap(train, val, test) -> dict[str, int]:
    t, v, s = set(train["lesion_id"]), set(val["lesion_id"]), set(test["lesion_id"])
    return {
        "train_val": len(t & v),
        "train_test": len(t & s),
        "val_test": len(v & s),
    }


if __name__ == "__main__":
    main()
