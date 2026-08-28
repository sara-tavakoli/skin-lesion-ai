"""FastAPI service exposing prediction + Grad-CAM explanation endpoints.

Run locally::

    uvicorn skinlesion.serve.api:app --reload

Configuration via environment variables:
    SKINLESION_CKPT       path to a checkpoint (default: artifacts/best.ckpt)
    SKINLESION_DEVICE     auto | cpu | cuda | mps
    SKINLESION_TTA        1/0 (default 1)
    SKINLESION_CAM_METHOD gradcam++ | gradcam | xgradcam | scorecam
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from io import BytesIO

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from skinlesion import CLASS_NAMES, CLASSES, __version__

app = FastAPI(
    title="Dermoscopic Skin-Lesion Classifier",
    version=__version__,
    description="Research prototype. NOT a medical device. Not for clinical use.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_BYTES = 12 * 1024 * 1024
DISCLAIMER = (
    "This tool is a research prototype trained on HAM10000 dermoscopy images. "
    "It is not a medical device and must not be used for diagnosis or treatment "
    "decisions. Always consult a qualified clinician."
)


class PredictResponse(BaseModel):
    label: str
    label_name: str
    probabilities: dict[str, float]
    malignant_probability: float = Field(..., ge=0.0, le=1.0)
    entropy: float
    epistemic_uncertainty: float | None = None
    warnings: list[str] = []
    disclaimer: str = DISCLAIMER


class ExplainResponse(PredictResponse):
    cam_method: str
    overlay_png_base64: str


@lru_cache(maxsize=1)
def get_predictor():
    from skinlesion.serve.inference import LesionPredictor

    ckpt = os.environ.get("SKINLESION_CKPT", "artifacts/best.ckpt")
    if not os.path.exists(ckpt):
        raise RuntimeError(f"checkpoint '{ckpt}' not found; set SKINLESION_CKPT or train a model first")
    return LesionPredictor(
        ckpt,
        device=os.environ.get("SKINLESION_DEVICE", "auto"),
        use_tta=os.environ.get("SKINLESION_TTA", "1") == "1",
    )


@lru_cache(maxsize=1)
def get_explainer():
    from skinlesion.explain.cam import LesionExplainer

    predictor = get_predictor()
    method = os.environ.get("SKINLESION_CAM_METHOD", "gradcam++")
    return LesionExplainer(predictor.model, method=method, device=str(predictor.device)), method


async def _read_image(file: UploadFile) -> np.ndarray:
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "image too large (max 12 MB)")
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"could not decode image: {exc}") from exc
    return np.asarray(img)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/model-card")
def model_card() -> dict:
    return {
        "classes": {c: CLASS_NAMES[c] for c in CLASSES},
        "training_data": "HAM10000 (ISIC 2018 Task 3), lesion-level split",
        "intended_use": "research and education only",
        "not_intended_use": "clinical diagnosis, triage, or treatment decisions",
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    mc_dropout_samples: int = 0,
    predictor=Depends(get_predictor),
) -> PredictResponse:
    arr = await _read_image(file)
    pred = predictor.predict(arr, mc_dropout_samples=mc_dropout_samples)
    return PredictResponse(**pred.as_dict())


@app.post("/explain", response_model=ExplainResponse)
async def explain(
    file: UploadFile = File(...),
    predictor=Depends(get_predictor),
) -> ExplainResponse:
    import torch

    arr = await _read_image(file)
    pred = predictor.predict(arr)
    explainer, method = get_explainer()
    x = predictor.eval_tf(image=arr)["image"]
    result = explainer.explain(torch.as_tensor(x))

    buf = BytesIO()
    Image.fromarray(result.overlay).save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return ExplainResponse(
        **pred.as_dict(),
        cam_method=method,
        overlay_png_base64=overlay_b64,
    )
