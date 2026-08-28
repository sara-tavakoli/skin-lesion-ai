"""timm backbone factory with a consistent feature/CAM interface."""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import nn

# Friendly aliases -> timm model names.
BACKBONES: dict[str, str] = {
    "effnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "effnetv2_m": "tf_efficientnetv2_m.in21k_ft_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "convnext_small": "convnext_small.fb_in22k_ft_in1k",
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "resnet50": "resnet50.a1_in1k",
}


@dataclass
class BackboneOutput:
    logits: torch.Tensor
    features: torch.Tensor  # pre-classifier pooled features


class TimmClassifier(nn.Module):
    """Wraps a timm backbone + a fresh linear head with dropout.

    Exposes :meth:`cam_target_layer` so explainability code does not need to
    know per-architecture layer names.
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool = True,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        timm_name = BACKBONES.get(backbone, backbone)
        self.backbone_name = timm_name
        self.encoder = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,  # remove head, keep pooled features
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        self.feature_dim: int = self.encoder.num_features  # type: ignore[assignment]
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.feature_dim, num_classes)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.head(self.dropout(feats))

    def forward_with_features(self, x: torch.Tensor) -> BackboneOutput:
        feats = self.encoder(x)
        logits = self.head(self.dropout(feats))
        return BackboneOutput(logits=logits, features=feats)

    # -- explainability support -----------------------------------------
    def cam_target_layer(self) -> nn.Module:
        """Best-effort last spatial layer for Grad-CAM."""
        candidates = []
        for _, module in self.encoder.named_modules():
            if isinstance(module, (nn.Conv2d, nn.LayerNorm, nn.BatchNorm2d)):
                candidates.append(module)
        if not candidates:  # pragma: no cover - ViT path handled by reshape_transform
            raise RuntimeError("no CAM-compatible layer found; pass one explicitly")
        return candidates[-1]

    def group_parameters(self, head_lr_mult: float = 10.0):
        """Param groups so the fresh head can learn faster than the encoder."""
        return [
            {"params": self.encoder.parameters(), "lr_mult": 1.0},
            {
                "params": list(self.dropout.parameters()) + list(self.head.parameters()),
                "lr_mult": head_lr_mult,
            },
        ]


def create_model(cfg) -> TimmClassifier:
    return TimmClassifier(
        backbone=cfg.name,
        num_classes=cfg.num_classes,
        pretrained=cfg.get("pretrained", True),
        drop_rate=cfg.get("drop_rate", 0.2),
        drop_path_rate=cfg.get("drop_path_rate", 0.1),
    )
