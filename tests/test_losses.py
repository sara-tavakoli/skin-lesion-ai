from __future__ import annotations

import torch

from skinlesion.losses import FocalLoss, SoftTargetCrossEntropy, build_loss


def test_focal_reduces_to_ce_when_gamma_zero():
    logits = torch.randn(16, 7)
    target = torch.randint(0, 7, (16,))
    focal = FocalLoss(gamma=0.0)
    ce = torch.nn.functional.cross_entropy(logits, target)
    torch.testing.assert_close(focal(logits, target), ce, rtol=1e-4, atol=1e-4)


def test_focal_downweights_easy_examples():
    # one very easy example, one hard example
    easy = torch.tensor([[10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]])
    hard = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([0])
    fl = FocalLoss(gamma=2.0, reduction="none")
    loss_easy = fl(easy, target)
    loss_hard = fl(hard, target)
    assert loss_easy.item() < 0.01 * loss_hard.item()


def test_focal_with_class_weights_runs():
    w = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0])
    fl = FocalLoss(gamma=1.5, weight=w)
    loss = fl(torch.randn(8, 7), torch.randint(0, 7, (8,)))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_soft_target_ce_matches_hard_ce_on_one_hot():
    logits = torch.randn(8, 7)
    target = torch.randint(0, 7, (8,))
    one_hot = torch.nn.functional.one_hot(target, 7).float()
    soft = SoftTargetCrossEntropy()(logits, one_hot)
    hard = torch.nn.functional.cross_entropy(logits, target)
    torch.testing.assert_close(soft, hard, rtol=1e-4, atol=1e-4)


def test_build_loss_dispatch():
    assert isinstance(build_loss("focal"), FocalLoss)
    assert build_loss("ce").__class__.__name__ == "CrossEntropyLoss"
    try:
        build_loss("nope")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
