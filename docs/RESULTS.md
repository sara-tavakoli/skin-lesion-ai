# Results

> Populate this file from `artifacts/eval/RESULTS.md` after a real training run.
> Always report the **git SHA**, **config snapshot**, and **seed** alongside the
> numbers (they are in `artifacts/env.json` and `artifacts/config.snapshot.yaml`).

## Protocol
- Test = lesion-disjoint fold 0; val = fold 1; train = folds 2–4.
- Metrics computed on **temperature-scaled** probabilities (T fit on val).
- 95 % CIs = 2 000-resample bootstrap over the test set.
- Backbone / recipe: _fill in_ (e.g. `effnetv2_s`, `--experiment focal_balanced`, 40 epochs).

## Headline (template)

| Metric | Value | 95 % CI |
|---|---|---|
| Balanced accuracy | – | [–, –] |
| Macro AUROC | – | [–, –] |
| Macro AP | – | [–, –] |
| Quadratic-weighted κ | – | [–, –] |
| ECE (pre / post temperature) | – / – | T = – |
| AURC / excess-AURC | – / – | – |
| Melanoma sensitivity @ referral threshold | – | – |
| Specificity @ 0.95 malignant sensitivity | – | – |

## Per-class (template)

| Class | Support | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|
| akiec | – | – | – | – | – |
| bcc | – | – | – | – | – |
| bkl | – | – | – | – | – |
| df | – | – | – | – | – |
| mel | – | – | – | – | – |
| nv | – | – | – | – | – |
| vasc | – | – | – | – | – |

## Ablations to report
| Run | Balanced acc | Macro AUROC | κ | Notes |
|---|---|---|---|---|
| `baseline_ce` | – | – | – | plain weighted CE, no MixUp/EMA |
| `focal_balanced` | – | – | – | class-balanced focal + sampler + MixUp + EMA |
| `model=convnext_tiny` | – | – | – | |
| `model=vit_small` | – | – | – | |
| + TTA | – | – | – | dihedral flips at inference |
| + deep ensemble (×3) | – | – | – | |

## Figures
- `artifacts/eval/confusion_matrix.png`
- `artifacts/eval/reliability.png`
- `artifacts/eval/risk_coverage.png`
- Grad-CAM montage from `scripts/explain.py`

## Comparison points from the literature
HAM10000 / ISIC-2018 Task 3 leaderboard solutions report balanced multi-class
accuracy roughly in the 0.85–0.90 range with heavy ensembling and external data.
Single-model, single-dataset, lesion-disjoint numbers are typically lower — report
yours honestly and note the protocol differences.
