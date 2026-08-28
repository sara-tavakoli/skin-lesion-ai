"""End-to-end smoke: config -> datamodule -> Lightning fit -> predict -> serve."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_train_one_epoch_and_predict(synthetic_data, tmp_path):
    import pytorch_lightning as pl

    from skinlesion.data.datamodule import HAM10000DataModule
    from skinlesion.models.classifier import LesionClassifier
    from skinlesion.utils.config import load_config

    cfg = load_config(
        [
            f"data.data_dir={synthetic_data}",
            "data.image_size=64",
            "data.batch_size=8",
            "data.num_workers=0",
            "data.n_folds=3",
            "data.balanced_sampler=false",
            "model.pretrained=false",
            "model.drop_path_rate=0.0",
            "train.max_epochs=1",
            "train.precision=32-true",
            "train.ema.enabled=false",
            "train.mixup.enabled=false",
        ],
    )
    dm = HAM10000DataModule(
        data_dir=str(synthetic_data),
        image_size=64,
        batch_size=8,
        num_workers=0,
        n_folds=3,
        balanced_sampler=False,
    )
    model = LesionClassifier(cfg)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_train_batches=4,
        limit_val_batches=2,
        limit_test_batches=2,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(model, datamodule=dm)
    metrics = trainer.test(model, datamodule=dm)
    assert "test/auroc" in metrics[0]

    ckpt = tmp_path / "m.ckpt"
    trainer.save_checkpoint(ckpt)

    from skinlesion.serve.inference import LesionPredictor

    predictor = LesionPredictor(ckpt, backbone=cfg.model.name, image_size=64, device="cpu", use_tta=False)
    img = next((synthetic_data / "images").glob("*.jpg"))
    pred = predictor.predict(img.read_bytes())
    assert pred.label in {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-4


@pytest.mark.slow
def test_gradcam_runs_on_trained_stub(synthetic_data):
    from skinlesion.explain.cam import LesionExplainer
    from skinlesion.models.backbones import TimmClassifier

    model = TimmClassifier("resnet50", num_classes=7, pretrained=False).eval()
    explainer = LesionExplainer(model, method="gradcam++", device="cpu")
    res = explainer.explain(torch.randn(3, 64, 64))
    assert res.overlay.shape == (64, 64, 3)
    assert res.heatmap.min() >= 0.0 and res.heatmap.max() <= 1.0
