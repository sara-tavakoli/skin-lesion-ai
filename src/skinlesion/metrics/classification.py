"""Evaluation metrics with uncertainty quantification.

Everything here operates on plain numpy arrays so it can be reused outside the
training loop (notebooks, the ``evaluate`` script, unit tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)


@dataclass
class ClassificationReport:
    n: int
    accuracy: float
    balanced_accuracy: float
    macro_auroc: float
    macro_ap: float
    quadratic_kappa: float
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_auroc": self.macro_auroc,
            "macro_ap": self.macro_ap,
            "quadratic_kappa": self.quadratic_kappa,
            "per_class": self.per_class,
            "confusion": self.confusion,
            "ci95": {k: list(v) for k, v in self.ci.items()},
        }


def _one_hot(y: np.ndarray, k: int) -> np.ndarray:
    oh = np.zeros((y.size, k), dtype=np.float64)
    oh[np.arange(y.size), y] = 1.0
    return oh


def per_class_sensitivity_specificity(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict[str, dict[str, float]]:
    k = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    out: dict[str, dict[str, float]] = {}
    total = cm.sum()
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        ppv = tp / (tp + fp) if (tp + fp) else float("nan")
        out[name] = {
            "support": int(tp + fn),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "precision": float(ppv),
            "f1": float(2 * ppv * sens / (ppv + sens)) if (ppv + sens) else float("nan"),
        }
    return out


def compute_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> ClassificationReport:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    k = len(class_names)
    y_pred = y_prob.argmax(1)
    y_oh = _one_hot(y_true, k)

    present = y_oh.sum(0) > 0  # AUROC/AP undefined for absent classes

    def _macro_auroc(t_oh, p):
        cols = present & (t_oh.sum(0) > 0)
        return roc_auc_score(t_oh[:, cols], p[:, cols], average="macro") if cols.sum() > 1 else float("nan")

    def _macro_ap(t_oh, p):
        cols = present & (t_oh.sum(0) > 0)
        if not cols.any():
            return float("nan")
        return average_precision_score(t_oh[:, cols], p[:, cols], average="macro")

    report = ClassificationReport(
        n=int(y_true.size),
        accuracy=float((y_pred == y_true).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_auroc=_macro_auroc(y_oh, y_prob),
        macro_ap=_macro_ap(y_oh, y_prob),
        quadratic_kappa=float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        per_class=per_class_sensitivity_specificity(y_true, y_pred, class_names),
        confusion=confusion_matrix(y_true, y_pred, labels=list(range(k))).tolist(),
    )

    if n_bootstrap:
        rng = np.random.default_rng(seed)
        acc, bacc, auroc, ap, kappa = [], [], [], [], []
        idx_all = np.arange(y_true.size)
        for _ in range(n_bootstrap):
            idx = rng.choice(idx_all, size=idx_all.size, replace=True)
            yt, yp, pr = y_true[idx], y_pred[idx], y_prob[idx]
            if np.unique(yt).size < 2:
                continue
            acc.append((yp == yt).mean())
            bacc.append(balanced_accuracy_score(yt, yp))
            auroc.append(_macro_auroc(_one_hot(yt, k), pr))
            ap.append(_macro_ap(_one_hot(yt, k), pr))
            kappa.append(cohen_kappa_score(yt, yp, weights="quadratic"))

        def _ci(vals: list[float]) -> tuple[float, float]:
            arr = np.asarray([v for v in vals if np.isfinite(v)])
            if arr.size == 0:
                return (float("nan"), float("nan"))
            return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

        report.ci = {
            "accuracy": _ci(acc),
            "balanced_accuracy": _ci(bacc),
            "macro_auroc": _ci(auroc),
            "macro_ap": _ci(ap),
            "quadratic_kappa": _ci(kappa),
        }
    return report


def delong_roc_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> float:
    """Two-sided p-value for AUC(A) == AUC(B) on paired binary scores (DeLong).

    Implementation follows Sun & Xu (2014).  Returns the p-value.
    """
    from scipy import stats

    y_true = np.asarray(y_true).astype(int)
    pos = y_true == 1
    neg = ~pos
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        return float("nan")

    def _structural_components(scores):
        sp, sn = scores[pos], scores[neg]
        # midrank-based placements
        tx = _midrank(sp)
        ty = _midrank(sn)
        tz = _midrank(np.concatenate([sp, sn]))
        auc = (tz[:m].sum() / m - (m + 1) / 2) / n
        v01 = (tz[:m] - tx) / n
        v10 = 1.0 - (tz[m:] - ty) / m
        return auc, v01, v10

    a_auc, a01, a10 = _structural_components(prob_a)
    b_auc, b01, b10 = _structural_components(prob_b)

    def _cov(x, y):
        return np.cov(np.stack([x, y]), ddof=1)

    s01 = _cov(a01, b01)
    s10 = _cov(a10, b10)
    var = s01 / m + s10 / n
    diff_var = var[0, 0] + var[1, 1] - 2 * var[0, 1]
    if diff_var <= 0:
        return 1.0
    z = (a_auc - b_auc) / np.sqrt(diff_var)
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranked = np.empty_like(order, dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j < len(xs) and xs[j] == xs[i]:
            j += 1
        ranked[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty_like(ranked)
    out[order] = ranked
    return out
