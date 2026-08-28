from __future__ import annotations

import numpy as np

from skinlesion import CLASSES
from skinlesion.metrics.classification import (
    compute_report,
    delong_roc_test,
    per_class_sensitivity_specificity,
)


def _synth_probs(y_true, n_classes, noise, rng):
    probs = rng.dirichlet(np.ones(n_classes) * 0.5, size=y_true.size)
    probs[np.arange(y_true.size), y_true] += noise
    return probs / probs.sum(1, keepdims=True)


def test_perfect_predictions_give_unit_metrics():
    y = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
    probs = np.eye(7)[y]
    rep = compute_report(y, probs, list(CLASSES), n_bootstrap=0)
    assert rep.accuracy == 1.0
    assert abs(rep.balanced_accuracy - 1.0) < 1e-9
    assert abs(rep.quadratic_kappa - 1.0) < 1e-9


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, size=400)
    probs = _synth_probs(y, 7, noise=1.5, rng=rng)
    rep = compute_report(y, probs, list(CLASSES), n_bootstrap=300, seed=0)
    lo, hi = rep.ci["accuracy"]
    assert lo <= rep.accuracy <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_sensitivity_specificity_shapes():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 7, size=200)
    y_pred = rng.integers(0, 7, size=200)
    out = per_class_sensitivity_specificity(y_true, y_pred, list(CLASSES))
    assert set(out) == set(CLASSES)
    for m in out.values():
        assert 0.0 <= m["sensitivity"] <= 1.0
        assert 0.0 <= m["specificity"] <= 1.0


def test_delong_identical_scores_gives_high_pvalue():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=300)
    score = rng.random(300) + y * 0.3
    p = delong_roc_test(y, score, score.copy())
    assert p > 0.99


def test_delong_detects_clear_difference():
    rng = np.random.default_rng(3)
    y = np.r_[np.ones(150), np.zeros(150)].astype(int)
    good = rng.normal(y * 2.0, 1.0)
    bad = rng.normal(y * 0.1, 1.0)
    p = delong_roc_test(y, good, bad)
    assert p < 0.05
