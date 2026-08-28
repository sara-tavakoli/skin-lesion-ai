#!/usr/bin/env python
"""Generate a small synthetic HAM10000-style dataset.

Produces coloured procedural "lesions" with class-correlated statistics so that
a model can actually learn something above chance in a smoke test.  This is
NOT real data -- it exists purely so the full pipeline (splits -> train ->
evaluate -> serve) runs end-to-end in CI and on machines without the download.

Usage::

    python scripts/make_synthetic_data.py --out data/ham10000 --per-class 60
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
# per-class (hue_center, texture_freq, border_irregularity)
_PROFILE = {
    "akiec": (0.06, 8, 0.5),
    "bcc": (0.95, 5, 0.4),
    "bkl": (0.09, 12, 0.25),
    "df": (0.05, 3, 0.15),
    "mel": (0.75, 16, 0.8),
    "nv": (0.08, 6, 0.1),
    "vasc": (0.99, 2, 0.2),
}


def _hsv_to_rgb(h, s, v):
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]


def render_lesion(cls: str, size: int, rng: np.random.Generator) -> Image.Image:
    hue_c, freq, irr = _PROFILE[cls]
    yy, xx = np.mgrid[0:size, 0:size] / size - 0.5
    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)

    radius = 0.32 + irr * 0.12 * rng.standard_normal()
    wobble = irr * 0.10 * np.sin(theta * rng.integers(3, 7) + rng.uniform(0, 6.28))
    mask = r < (radius + wobble)

    skin = np.array(_hsv_to_rgb((0.06 + 0.02 * rng.random()) % 1.0, 0.35, 0.85))
    lesion_hue = (hue_c + 0.03 * rng.standard_normal()) % 1.0
    lesion = np.array(_hsv_to_rgb(lesion_hue, 0.55 + 0.2 * rng.random(), 0.5 + 0.2 * rng.random()))

    texture = (
        0.12
        * np.sin(freq * np.pi * xx + rng.uniform(0, 6.28))
        * np.sin(freq * np.pi * yy + rng.uniform(0, 6.28))
    )
    img = np.empty((size, size, 3))
    for c in range(3):
        base = np.where(mask, lesion[c] + texture, skin[c])
        img[:, :, c] = base
    img += 0.03 * rng.standard_normal(img.shape)
    return Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/ham10000"))
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--images-per-lesion-max", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    counter = 0
    for cls in CLASSES:
        made = 0
        lesion_no = 0
        while made < args.per_class:
            lesion_id = f"SYN_{cls}_{lesion_no:04d}"
            n_views = int(rng.integers(1, args.images_per_lesion_max + 1))
            for _ in range(min(n_views, args.per_class - made)):
                image_id = f"SYN_{counter:06d}"
                render_lesion(cls, args.size, rng).save(img_dir / f"{image_id}.jpg", quality=92)
                rows.append(
                    {
                        "image_id": image_id,
                        "lesion_id": lesion_id,
                        "dx": cls,
                        "dx_type": "synthetic",
                        "age": int(rng.integers(20, 85)),
                        "sex": rng.choice(["male", "female"]),
                        "localization": rng.choice(["back", "trunk", "face", "lower extremity"]),
                    }
                )
                counter += 1
                made += 1
            lesion_no += 1

    with open(args.out / "metadata.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} synthetic images to {img_dir}")
    print(f"metadata: {args.out / 'metadata.csv'}")


if __name__ == "__main__":
    main()
