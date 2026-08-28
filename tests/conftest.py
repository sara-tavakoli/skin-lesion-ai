from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory) -> Path:
    """A tiny synthetic HAM10000-style dataset shared across the test session."""
    out = tmp_path_factory.mktemp("ham10000")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_synthetic_data.py"),
            "--out",
            str(out),
            "--per-class",
            "12",
            "--size",
            "64",
            "--seed",
            "0",
        ],
        check=True,
    )
    assert (out / "metadata.csv").exists()
    return out


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def dummy_metadata() -> pd.DataFrame:
    rows = []
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    for c in classes:
        for lesion in range(6):
            for view in range(2):
                rows.append(
                    {
                        "image_id": f"{c}_{lesion}_{view}",
                        "lesion_id": f"{c}_{lesion}",
                        "dx": c,
                    }
                )
    return pd.DataFrame(rows)
