"""Checkpoint loading and single-image inference with optional TTA + MC-dropout."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from skinlesion import CLASS_NAMES, CLASSES, MALIGNANT
from skinlesion.data.transforms import eval_transform, tta_transforms
from skinlesion.models.backbones import TimmClassifier
from skinlesion.uncertainty.selective import mc_dropout_predict, predictive_entropy


@dataclass
class Prediction:
    label: str
    label_name: str
    probabilities: dict[str, float]
    malignant_probability: float
    entropy: float
    epistemic_uncertainty: float | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "label_name": self.label_name,
            "probabilities": self.probabilities,
            "malignant_probability": self.malignant_probability,
            "entropy": self.entropy,
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "warnings": self.warnings,
        }


def _select_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LesionPredictor:
    """Framework-agnostic predictor.  Accepts either a Lightning checkpoint or a
    plain ``state_dict`` + a small metadata dict (``backbone``, ``image_size``)."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        backbone: str | None = None,
        image_size: int | None = None,
        device: str = "auto",
        use_tta: bool = True,
    ) -> None:
        self.device = _select_device(device)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        hparams = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
        model_cfg = hparams.get("model", {}) if hparams else {}
        self.backbone = backbone or model_cfg.get("name", "effnetv2_s")
        self.image_size = image_size or hparams.get("data", {}).get("image_size", 224)

        self.model = TimmClassifier(self.backbone, num_classes=len(CLASSES), pretrained=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        # prefer EMA weights if the checkpoint carries them
        if isinstance(ckpt, dict) and "ema" in ckpt:
            ema_state = {k.replace("module.", "", 1): v for k, v in ckpt["ema"]["module"].items()}
            self.model.load_state_dict(ema_state, strict=False)
        else:
            clean = {k.replace("model.", "", 1): v for k, v in state.items() if k.startswith("model.")}
            self.model.load_state_dict(clean or state, strict=False)
        self.model.eval().to(self.device)

        self.eval_tf = eval_transform(self.image_size)
        self.tta_tfs = tta_transforms(self.image_size) if use_tta else None

    # -- io -------------------------------------------------------------
    @staticmethod
    def _to_array(image: bytes | str | Path | Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, Image.Image):
            return np.asarray(image.convert("RGB"))
        if isinstance(image, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image))
        else:
            image = Image.open(image)
        return np.asarray(image.convert("RGB"))

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def _forward_probs(self, arr: np.ndarray) -> np.ndarray:
        tfs = self.tta_tfs or [self.eval_tf]
        batch = torch.stack([t(image=arr)["image"] for t in tfs]).to(self.device)
        probs = self.model(batch).softmax(1).mean(0)
        return probs.cpu().numpy()

    def predict(
        self,
        image: bytes | str | Path | Image.Image | np.ndarray,
        *,
        mc_dropout_samples: int = 0,
    ) -> Prediction:
        arr = self._to_array(image)
        probs = self._forward_probs(arr)

        epistemic = None
        if mc_dropout_samples > 0:
            x = self.eval_tf(image=arr)["image"].unsqueeze(0).to(self.device)
            sampled = mc_dropout_predict(self.model, x, n_samples=mc_dropout_samples)
            mean = sampled.mean(0).cpu().numpy()[0]
            probs = 0.5 * (probs + mean)
            epistemic = float(sampled.var(0).sum().cpu())

        idx = int(np.argmax(probs))
        label = CLASSES[idx]
        prob_map = {c: float(probs[i]) for i, c in enumerate(CLASSES)}
        mal = float(sum(prob_map[c] for c in MALIGNANT))
        ent = float(predictive_entropy(probs[None, :])[0])

        warnings: list[str] = []
        if ent > 1.5:
            warnings.append("High predictive entropy - model is uncertain; recommend expert review.")
        if 0.35 <= mal <= 0.65:
            warnings.append("Borderline malignant probability; do not use as a standalone decision.")

        return Prediction(
            label=label,
            label_name=CLASS_NAMES[label],
            probabilities=prob_map,
            malignant_probability=mal,
            entropy=ent,
            epistemic_uncertainty=epistemic,
            warnings=warnings,
        )
