"""Grad-CAM family explanations via the ``grad-cam`` package.

Handles both CNN and ViT backbones (the latter needs a ``reshape_transform``
to fold the token sequence back into a spatial grid).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from skinlesion.data.transforms import IMAGENET_MEAN, IMAGENET_STD

_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "xgradcam": XGradCAM,
    "scorecam": ScoreCAM,
}


@dataclass
class CamResult:
    heatmap: np.ndarray  # HxW float in [0, 1]
    overlay: np.ndarray  # HxWx3 uint8
    class_idx: int
    class_prob: float


def _vit_reshape_transform(tensor, height: int = 14, width: int = 14):
    # drop CLS token, fold to (B, C, H, W)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(-1))
    return result.permute(0, 3, 1, 2)


def denormalize(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (x.detach().cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


class LesionExplainer:
    def __init__(self, model, method: str = "gradcam++", target_layer=None, device="cpu") -> None:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {sorted(_METHODS)}")
        self.model = model.eval().to(device)
        self.device = device
        inner = getattr(model, "model", model)  # unwrap LightningModule
        layer = target_layer or inner.cam_target_layer()
        reshape = _vit_reshape_transform if "vit" in inner.backbone_name.lower() else None
        self.cam = _METHODS[method](
            model=inner,
            target_layers=[layer],
            reshape_transform=reshape,
        )

    @torch.no_grad()
    def _predict(self, x: torch.Tensor) -> tuple[int, float]:
        inner = getattr(self.model, "model", self.model)
        probs = inner(x.to(self.device)).softmax(1)[0]
        idx = int(probs.argmax())
        return idx, float(probs[idx])

    def explain(self, x: torch.Tensor, class_idx: int | None = None) -> CamResult:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        pred_idx, prob = self._predict(x)
        target_idx = pred_idx if class_idx is None else class_idx
        grayscale = self.cam(
            input_tensor=x.to(self.device),
            targets=[ClassifierOutputTarget(target_idx)],
        )[0]
        rgb = denormalize(x[0])
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
        return CamResult(
            heatmap=grayscale.astype(np.float32),
            overlay=overlay.astype(np.uint8),
            class_idx=target_idx,
            class_prob=prob if class_idx is None else float("nan"),
        )
