#!/usr/bin/env python
"""Fetch and normalise the HAM10000 dataset into ``data/ham10000/``.

Two sources are supported:

1. **Kaggle** (default) -- ``kmader/skin-cancer-mnist-ham10000``.  Requires a
   Kaggle API token at ``~/.kaggle/kaggle.json`` (see
   https://www.kaggle.com/docs/api).
2. **Harvard Dataverse** direct download (``--source dataverse``), no auth.

After extraction the script writes a unified ``metadata.csv`` with columns
``image_id, lesion_id, dx, dx_type, age, sex, localization`` and moves every
JPEG into ``images/``.

Usage::

    python scripts/download_ham10000.py --out data/ham10000
    python scripts/download_ham10000.py --source dataverse --out data/ham10000
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

DATAVERSE_FILES = {
    # HAM10000 DOI 10.7910/DVN/DBW86T
    "HAM10000_metadata.tab": "https://dataverse.harvard.edu/api/access/datafile/3172582",
    "HAM10000_images_part_1.zip": "https://dataverse.harvard.edu/api/access/datafile/3172585",
    "HAM10000_images_part_2.zip": "https://dataverse.harvard.edu/api/access/datafile/3172584",
}


def _kaggle_download(raw_dir: Path) -> None:
    try:
        import kaggle  # noqa: F401
    except Exception as exc:
        sys.exit(f"kaggle package not importable ({exc}); `pip install kaggle` and add a token")
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "kmader/skin-cancer-mnist-ham10000",
        "-p",
        str(raw_dir),
        "--unzip",
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _dataverse_download(raw_dir: Path) -> None:
    import requests

    raw_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in DATAVERSE_FILES.items():
        dest = raw_dir / fname
        if dest.exists():
            print(f"skip {fname} (exists)")
            continue
        print(f"downloading {fname} ...")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        if fname.endswith(".zip"):
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(raw_dir)


def _normalise(raw_dir: Path, out_dir: Path) -> None:
    meta_path = next(
        (
            p
            for p in raw_dir.rglob("*")
            if p.name in {"HAM10000_metadata.csv", "HAM10000_metadata.tab", "HAM10000_metadata"}
        ),
        None,
    )
    if meta_path is None:
        sys.exit(f"could not find HAM10000 metadata under {raw_dir}")
    sep = "\t" if meta_path.suffix == ".tab" else ","
    df = pd.read_csv(meta_path, sep=sep)
    df = df.rename(columns={"image_id": "image_id", "lesion_id": "lesion_id", "dx": "dx"})
    keep = ["image_id", "lesion_id", "dx", "dx_type", "age", "sex", "localization"]
    df = df[[c for c in keep if c in df.columns]]

    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    jpegs = {p.stem: p for p in raw_dir.rglob("*.jpg")}
    missing = 0
    for image_id in df["image_id"]:
        src = jpegs.get(image_id)
        if src is None:
            missing += 1
            continue
        dst = img_out / f"{image_id}.jpg"
        if not dst.exists():
            shutil.copy2(src, dst)
    if missing:
        print(f"warning: {missing} images referenced in metadata not found on disk")

    df.to_csv(out_dir / "metadata.csv", index=False)
    print(f"normalised dataset -> {out_dir}")
    print(df["dx"].value_counts().to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/ham10000"))
    ap.add_argument("--source", choices=["kaggle", "dataverse"], default="kaggle")
    ap.add_argument("--raw-dir", type=Path, default=None)
    args = ap.parse_args()

    raw_dir = args.raw_dir or (args.out / "_raw")
    if args.source == "kaggle":
        _kaggle_download(raw_dir)
    else:
        _dataverse_download(raw_dir)
    _normalise(raw_dir, args.out)


if __name__ == "__main__":
    main()
