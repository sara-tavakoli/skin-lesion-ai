#!/usr/bin/env python
"""Batch Grad-CAM / Grad-CAM++ overlays for qualitative inspection.

    python scripts/explain.py --checkpoint artifacts/best.ckpt \
        --images data/ham10000/images --limit 24 --out artifacts/cams
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skinlesion import CLASSES
from skinlesion.explain.cam import LesionExplainer
from skinlesion.serve.inference import LesionPredictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", type=Path, required=True, help="file or directory")
    ap.add_argument("--out", type=Path, default=Path("artifacts/cams"))
    ap.add_argument("--method", default="gradcam++", choices=["gradcam", "gradcam++", "xgradcam", "scorecam"])
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    predictor = LesionPredictor(args.checkpoint, device=args.device, use_tta=False)
    explainer = LesionExplainer(predictor.model, method=args.method, device=str(predictor.device))

    paths = (
        [args.images]
        if args.images.is_file()
        else sorted(p for p in args.images.rglob("*.jpg"))[: args.limit]
    )
    args.out.mkdir(parents=True, exist_ok=True)

    for path in paths:
        arr = np.asarray(Image.open(path).convert("RGB"))
        x = predictor.eval_tf(image=arr)["image"]
        res = explainer.explain(torch.as_tensor(x))
        pred = predictor.predict(arr)
        stem = path.stem
        Image.fromarray(res.overlay).save(args.out / f"{stem}_{CLASSES[res.class_idx]}_cam.png")
        print(
            f"{stem}: pred={pred.label} p={pred.probabilities[pred.label]:.3f} "
            f"malignant={pred.malignant_probability:.3f}"
        )

    print(f"\nwrote {len(paths)} overlays to {args.out}")


if __name__ == "__main__":
    main()
