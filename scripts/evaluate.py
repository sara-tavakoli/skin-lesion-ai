#!/usr/bin/env python
"""Full test-set evaluation with uncertainty quantification.

Produces, in ``<output>/eval/``:
    * ``report.json``            -- accuracy, balanced acc, macro AUROC/AP,
                                    quadratic kappa, per-class sens/spec, all
                                    with bootstrap 95% CIs
    * ``calibration.json``       -- ECE/MCE before and after temperature scaling
    * ``selective.json``         -- AURC / excess-AURC, risk at fixed coverage
    * ``confusion_matrix.png``, ``reliability.png``, ``risk_coverage.png``
    * ``RESULTS.md``             -- a ready-to-paste results table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skinlesion import CLASSES, MALIGNANT
from skinlesion.data.datamodule import HAM10000DataModule
from skinlesion.metrics.calibration import TemperatureScaler, reliability_bins
from skinlesion.metrics.classification import compute_report
from skinlesion.models.classifier import LesionClassifier
from skinlesion.uncertainty.selective import predictive_entropy, risk_coverage_curve
from skinlesion.utils.config import load_config
from skinlesion.utils.seed import seed_everything


@torch.no_grad()
def _collect(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    logits, targets = [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        logits.append(model(x).cpu())
        targets.append(y)
    return torch.cat(logits).numpy(), torch.cat(targets).numpy()


def _plots(y_true, y_prob, out_dir: Path, temp: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_prob.argmax(1),
        display_labels=CLASSES,
        normalize="true",
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format=".2f",
    )
    ax.set_title("Row-normalised confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # reliability diagram
    rb = reliability_bins(y_true, y_prob, n_bins=15)
    centres = 0.5 * (rb.bin_edges[:-1] + rb.bin_edges[1:])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.bar(centres, rb.bin_acc, width=1 / 15, alpha=0.7, edgecolor="k", label="accuracy")
    ax.plot(centres, rb.bin_conf, "o-", color="crimson", label="confidence")
    ax.set(xlabel="confidence", ylabel="accuracy", title=f"Reliability (ECE={rb.ece:.3f}, T={temp:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reliability.png", dpi=150)
    plt.close(fig)

    # risk-coverage
    correct = (y_prob.argmax(1) == y_true).astype(float)
    rc = risk_coverage_curve(correct, y_prob.max(1))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rc.coverage, rc.risk)
    ax.set(
        xlabel="coverage",
        ylabel="selective risk",
        title=f"Risk-coverage (AURC={rc.aurc:.4f}, E-AURC={rc.eaurc:.4f})",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "risk_coverage.png", dpi=150)
    plt.close(fig)


def _results_md(report, calib, selective, out_path: Path) -> None:
    r = report.to_dict()
    lines = [
        "# Test-set results",
        "",
        "| Metric | Value | 95% CI |",
        "|---|---|---|",
        f"| Accuracy | {r['accuracy']:.4f} | {_ci(r, 'accuracy')} |",
        f"| Balanced accuracy | {r['balanced_accuracy']:.4f} | {_ci(r, 'balanced_accuracy')} |",
        f"| Macro AUROC | {r['macro_auroc']:.4f} | {_ci(r, 'macro_auroc')} |",
        f"| Macro AP | {r['macro_ap']:.4f} | {_ci(r, 'macro_ap')} |",
        f"| Quadratic-weighted kappa | {r['quadratic_kappa']:.4f} | {_ci(r, 'quadratic_kappa')} |",
        (
            f"| ECE (pre / post temp.) | {calib['ece_pre']:.4f} / "
            f"{calib['ece_post']:.4f} | T={calib['temperature']:.3f} |"
        ),
        f"| AURC / excess-AURC | {selective['aurc']:.4f} / {selective['eaurc']:.4f} | - |",
        "",
        "## Per-class",
        "",
        "| Class | Support | Sensitivity | Specificity | Precision | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for cls, m in r["per_class"].items():
        lines.append(
            f"| {cls} | {m['support']} | {m['sensitivity']:.3f} | {m['specificity']:.3f} "
            f"| {m['precision']:.3f} | {m['f1']:.3f} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def _ci(r: dict, key: str) -> str:
    lo, hi = r["ci95"].get(key, [float("nan"), float("nan")])
    return f"[{lo:.4f}, {hi:.4f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--output-dir", default="artifacts")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.overrides, experiment=args.experiment)
    seed_everything(cfg.seed, deterministic=False)
    device = (
        torch.device("cuda")
        if (args.device in ("auto", "cuda") and torch.cuda.is_available())
        else torch.device("mps")
        if (args.device in ("auto", "mps") and torch.backends.mps.is_available())
        else torch.device("cpu")
    )

    dm = HAM10000DataModule(
        data_dir=cfg.data.data_dir,
        metadata_csv=cfg.data.metadata_csv,
        image_subdir=cfg.data.image_subdir,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        aug_strength="light",
        balanced_sampler=False,
        n_folds=cfg.data.n_folds,
        test_fold=cfg.data.test_fold,
        val_fold=cfg.data.val_fold,
        seed=cfg.seed,
    )
    dm.setup("test")

    model = LesionClassifier.load_from_checkpoint(args.checkpoint, cfg=cfg, map_location="cpu", strict=False)
    core = model._ema.module if model._ema is not None else model.model

    val_logits, val_targets = _collect(core, dm.val_dataloader(), device)
    test_logits, test_targets = _collect(core, dm.test_dataloader(), device)

    # -- calibration on val, applied to test -----------------------------
    scaler = TemperatureScaler().fit(torch.tensor(val_logits), torch.tensor(val_targets))
    temp = scaler.temperature
    test_prob_pre = torch.tensor(test_logits).softmax(1).numpy()
    test_prob_post = (torch.tensor(test_logits) / temp).softmax(1).numpy()

    ece_pre = reliability_bins(test_targets, test_prob_pre).ece
    ece_post = reliability_bins(test_targets, test_prob_post).ece

    # -- classification report (on calibrated probs) --------------------
    report = compute_report(
        test_targets,
        test_prob_post,
        list(CLASSES),
        n_bootstrap=args.n_bootstrap,
        seed=cfg.seed,
    )

    # -- malignant referral operating point ---------------------------
    mal_idx = [CLASSES.index(c) for c in MALIGNANT]
    mal_score = test_prob_post[:, mal_idx].sum(1)
    mal_true = np.isin(test_targets, mal_idx).astype(int)
    from sklearn.metrics import roc_curve

    from skinlesion.utils.numeric import trapezoid

    fpr, tpr, thr = roc_curve(mal_true, mal_score)
    # threshold achieving >= 0.95 sensitivity
    idx = np.where(tpr >= 0.95)[0]
    op = {
        "target_sensitivity": 0.95,
        "threshold": float(thr[idx[0]]) if len(idx) else None,
        "specificity_at_target": float(1 - fpr[idx[0]]) if len(idx) else None,
        "auroc_malignant_vs_benign": trapezoid(tpr, fpr),
    }

    # -- selective prediction ----------------------------------------
    correct = (test_prob_post.argmax(1) == test_targets).astype(float)
    rc = risk_coverage_curve(correct, test_prob_post.max(1))
    ent = predictive_entropy(test_prob_post)
    cov80 = int(0.8 * len(correct))
    order = np.argsort(-test_prob_post.max(1))
    selective = {
        "aurc": rc.aurc,
        "eaurc": rc.eaurc,
        "risk_at_coverage_0.8": float(1 - correct[order[:cov80]].mean()),
        "mean_entropy": float(ent.mean()),
    }

    out_dir = Path(args.output_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report.to_dict(), indent=2))
    (out_dir / "calibration.json").write_text(
        json.dumps({"temperature": temp, "ece_pre": ece_pre, "ece_post": ece_post}, indent=2)
    )
    (out_dir / "selective.json").write_text(json.dumps(selective, indent=2))
    (out_dir / "operating_point.json").write_text(json.dumps(op, indent=2))
    _plots(test_targets, test_prob_post, out_dir, temp)
    _results_md(
        report,
        {"ece_pre": ece_pre, "ece_post": ece_post, "temperature": temp},
        selective,
        out_dir / "RESULTS.md",
    )

    print((out_dir / "RESULTS.md").read_text())
    print("malignant referral operating point:", json.dumps(op, indent=2))


if __name__ == "__main__":
    main()
