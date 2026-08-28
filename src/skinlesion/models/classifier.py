"""The LightningModule: training / validation / test logic and metrics."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassCohenKappa,
    MulticlassF1Score,
)

from skinlesion import CLASSES
from skinlesion.losses import SoftTargetCrossEntropy, build_loss
from skinlesion.models.backbones import create_model
from skinlesion.models.ema import ModelEMA
from skinlesion.models.mixup import MixupCutmix


class LesionClassifier(LightningModule):
    """Config-driven classifier with focal loss, MixUp, EMA, and a rich metric
    suite (balanced accuracy, macro AUROC/AP, quadratic-weighted kappa)."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.num_classes = len(CLASSES)

        self.model = create_model(cfg.model)
        self.criterion = build_loss(
            cfg.train.loss,
            gamma=cfg.train.get("focal_gamma", 2.0),
            label_smoothing=cfg.train.get("label_smoothing", 0.05),
        )
        # Register a real (ones) weight buffer up front so the checkpoint key is
        # always present; ``setup`` overwrites it with the data-driven weights.
        if getattr(self.criterion, "weight", None) is None:
            self.criterion.register_buffer("weight", torch.ones(self.num_classes))
        self.soft_criterion = SoftTargetCrossEntropy()

        self.mixup = None
        if cfg.train.get("mixup", {}).get("enabled", False):
            m = cfg.train.mixup
            self.mixup = MixupCutmix(
                num_classes=self.num_classes,
                mixup_alpha=m.get("mixup_alpha", 0.2),
                cutmix_alpha=m.get("cutmix_alpha", 1.0),
                prob=m.get("prob", 0.5),
                switch_prob=m.get("switch_prob", 0.5),
                label_smoothing=cfg.train.get("label_smoothing", 0.05),
                seed=cfg.get("seed", 0),
            )

        self._ema: ModelEMA | None = None
        self.class_weights: torch.Tensor | None = None

        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(self.num_classes, average="micro"),
                "bacc": MulticlassAccuracy(self.num_classes, average="macro"),
                "f1": MulticlassF1Score(self.num_classes, average="macro"),
                "auroc": MulticlassAUROC(self.num_classes, average="macro"),
                "ap": MulticlassAveragePrecision(self.num_classes, average="macro"),
                "kappa": MulticlassCohenKappa(self.num_classes, weights="quadratic"),
            }
        )
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    # -- setup ---------------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and getattr(dm, "class_weights", None) is not None:
            self.class_weights = dm.class_weights.to(self.device)
            if hasattr(self.criterion, "weight"):
                self.criterion.weight = self.class_weights
        if self.cfg.train.get("ema", {}).get("enabled", False) and self._ema is None:
            self._ema = ModelEMA(
                self.model,
                decay=self.cfg.train.ema.get("decay", 0.9998),
                warmup_steps=self.cfg.train.ema.get("warmup_steps", 2000),
            )

    # -- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # -- train -----------------------------------------------------------
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch[0], batch[1]
        if self.mixup is not None:
            x, soft = self.mixup(x, y)
            logits = self.model(x)
            loss = self.soft_criterion(logits, soft)
        else:
            logits = self.model(x)
            loss = self.criterion(logits, y)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        opt = self.optimizers()
        if not isinstance(opt, list):
            self.log("train/lr", opt.param_groups[0]["lr"], prog_bar=False)
        return loss

    def on_before_zero_grad(self, optimizer) -> None:
        if self._ema is not None:
            self._ema.update(self.model)

    # -- eval ------------------------------------------------------------
    def _eval_model(self):
        """EMA weights for evaluation, kept on the LightningModule's device.

        ``ModelEMA`` is a plain container, not a registered submodule, so
        Lightning's device move does not reach ``self._ema.module``.
        """
        if self._ema is None:
            return self.model
        if next(self._ema.module.parameters()).device != self.device:
            self._ema.module.to(self.device)
        return self._ema.module

    def _shared_eval(self, batch, metrics) -> torch.Tensor:
        x, y = batch[0], batch[1]
        model = self._eval_model()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        probs = logits.softmax(dim=1)
        metrics.update(probs, y)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        loss = self._shared_eval(batch, self.val_metrics)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx: int) -> None:
        self._shared_eval(batch, self.test_metrics)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x = batch[0]
        return self._eval_model()(x).softmax(dim=1)

    # -- optim ---------------------------------------------------------------
    def configure_optimizers(self):
        t = self.cfg.train
        base_lr = t.lr
        wd = t.get("weight_decay", 0.05)
        head_mult = t.get("head_lr_mult", 10.0)

        decay, no_decay, head = [], [], []
        head_ids = {id(p) for p in list(self.model.head.parameters())}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in head_ids:
                head.append(p)
            elif p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay, "weight_decay": wd, "lr": base_lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": base_lr},
            {"params": head, "weight_decay": wd, "lr": base_lr * head_mult},
        ]
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))

        sched_name = t.get("scheduler", "cosine")
        if sched_name == "none":
            return opt
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup = int(t.get("warmup_frac", 0.05) * total_steps)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            import math

            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

    # -- checkpoint plumbing for EMA --------------------------------------
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self._ema is not None:
            checkpoint["ema"] = self._ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "ema" in checkpoint:
            if self._ema is None:
                self._ema = ModelEMA(self.model)
            self._ema.load_state_dict(checkpoint["ema"])
