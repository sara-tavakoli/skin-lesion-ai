from __future__ import annotations

import torch

from skinlesion.models.backbones import TimmClassifier
from skinlesion.models.ema import ModelEMA
from skinlesion.models.mixup import MixupCutmix


def test_backbone_forward_shape():
    model = TimmClassifier("resnet50", num_classes=7, pretrained=False)
    out = model(torch.randn(2, 3, 96, 96))
    assert out.shape == (2, 7)


def test_forward_with_features():
    model = TimmClassifier("resnet50", num_classes=7, pretrained=False)
    res = model.forward_with_features(torch.randn(2, 3, 96, 96))
    assert res.logits.shape == (2, 7)
    assert res.features.shape[0] == 2
    assert res.features.shape[1] == model.feature_dim


def test_cam_target_layer_exists():
    model = TimmClassifier("resnet50", num_classes=7, pretrained=False)
    assert model.cam_target_layer() is not None


def test_ema_tracks_and_differs_from_model():
    model = TimmClassifier("resnet50", num_classes=7, pretrained=False)
    ema = ModelEMA(model, decay=0.9, warmup_steps=0)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(4, 3, 96, 96)
    for _ in range(3):
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()
        ema.update(model)
    p_model = next(model.parameters()).detach()
    p_ema = next(ema.module.parameters()).detach()
    assert not torch.allclose(p_model, p_ema)


def test_mixup_produces_soft_targets_summing_to_one():
    mix = MixupCutmix(num_classes=7, prob=1.0, seed=0)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 7, (8,))
    x2, soft = mix(x, y)
    assert x2.shape == x.shape
    assert soft.shape == (8, 7)
    torch.testing.assert_close(soft.sum(1), torch.ones(8), rtol=1e-4, atol=1e-4)


def test_mixup_disabled_path_is_one_hot_smoothed():
    mix = MixupCutmix(num_classes=7, prob=0.0, label_smoothing=0.1, seed=0)
    y = torch.tensor([0, 3])
    _, soft = mix(torch.randn(2, 3, 16, 16), y)
    assert soft.argmax(1).tolist() == [0, 3]
    assert soft.max() < 1.0  # smoothing applied
