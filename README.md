# Skin-Lesion AI — dermoscopic classification with calibration, uncertainty & explainability

[![CI](https://github.com/OWNER/skin-lesion-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/skin-lesion-ai/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A reproducible research pipeline for classifying pigmented skin lesions from
dermoscopy images (HAM10000 / ISIC-2018 Task 3, 7 diagnostic classes). The
emphasis is on the parts that matter for a *trustworthy* medical-imaging model,
not just top-1 accuracy:

| Concern | What this repo does |
|---|---|
| **Data leakage** | Splits on `lesion_id` with `StratifiedGroupKFold` — multiple views of one lesion never cross train/val/test. A leakage assertion runs on every split. |
| **Class imbalance** | Effective-number class weights (Cui et al. 2019), class-balanced focal loss, inverse-frequency `WeightedRandomSampler`, MixUp/CutMix. |
| **Honest metrics** | Balanced accuracy, macro AUROC / AP, quadratic-weighted κ, per-class sensitivity/specificity — each with **bootstrap 95 % CIs**; DeLong test for AUROC comparisons. |
| **Calibration** | Post-hoc temperature scaling fit on validation; ECE/MCE and reliability diagrams reported pre/post. |
| **Uncertainty** | MC-Dropout & deep-ensemble predictive entropy + mutual information (BALD); selective-prediction risk–coverage curves (AURC / excess-AURC); energy-based OOD score. |
| **Explainability** | Grad-CAM / Grad-CAM++ / XGrad-CAM / Score-CAM overlays for CNN *and* ViT backbones. |
| **Operating point** | Malignant-vs-benign referral threshold chosen at a target sensitivity (default 95 %), with the achieved specificity reported. |
| **Reproducibility** | Seeded everything, deterministic kernels, composable OmegaConf configs, per-run environment + git-SHA capture, MLflow logging, pinned deps, CI smoke-training. |

> ⚠️ **Not a medical device.** This is a research and educational prototype
> trained on a public research dataset. It must not be used for diagnosis,
> triage, or treatment decisions. See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

---

## Architecture

```
                     configs/ (OmegaConf, composable)
                            │
        ┌───────────────────┼─────────────────────────────┐
        ▼                   ▼                             ▼
  data/ (splits,      models/ (timm backbone +      metrics/  uncertainty/  explain/
  albumentations,     linear head, EMA, MixUp,      (bootstrap CIs,   (MC-dropout,   (Grad-CAM
  balanced sampler)   focal loss, LightningModule)   calibration)      risk-coverage) family)
        └───────────────────┴──────────────┬──────────────┘
                                           ▼
                        scripts/  train · evaluate · explain · export
                                           ▼
                        serve/  FastAPI  (/predict, /explain, /model-card)
```

Backbones (swap with `model=<name>`): `effnetv2_s` (default), `convnext_tiny`,
`vit_small`, `resnet50` — anything in `timm` works by passing its full name.

## Quickstart

```bash
git clone https://github.com/OWNER/skin-lesion-ai && cd skin-lesion-ai
make venv install                 # creates .venv, installs -e ".[serve,dev]"

# ---- Option A: run the whole thing on synthetic data (no download) ----
make smoke                        # synthetic data → 1-epoch train → full eval

# ---- Option B: the real dataset ----
python scripts/download_ham10000.py --out data/ham10000          # Kaggle token, or
python scripts/download_ham10000.py --source dataverse --out data/ham10000
python scripts/prepare_splits.py  --data-dir data/ham10000       # leakage-checked folds
python scripts/train.py --experiment focal_balanced              # full training
python scripts/evaluate.py --checkpoint artifacts/best.ckpt      # CIs, calibration, selective pred.
python scripts/explain.py  --checkpoint artifacts/best.ckpt --images data/ham10000/images
python scripts/export_model.py --checkpoint artifacts/best.ckpt --onnx
```

### Configuration

Configs compose like Hydra but without the runtime magic:

```bash
python scripts/train.py model=convnext_tiny data.image_size=384 train.lr=1e-4
python scripts/train.py --experiment baseline_ce          # ablation: plain weighted CE
python scripts/train.py --experiment fast_dev             # CI smoke recipe
```

Every run writes `artifacts/config.snapshot.yaml`, `env.json` (python/torch/git
SHA), `train.log`, MLflow metrics, and the best checkpoint.

### Train on a free GPU

[`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) is a self-contained
Kaggle notebook: it clones this repo, installs it, mounts the hosted HAM10000
dataset (no download), runs `--experiment focal_balanced` at 256&nbsp;px for 40
epochs on a P100, then evaluates and produces the Grad-CAM montage. Enable
**GPU + Internet** in the notebook settings and hit *Run All*; results land in
`artifacts/` and can be pushed straight back with a `GITHUB_TOKEN` Kaggle Secret.

## Serving

```bash
SKINLESION_CKPT=artifacts/best.ckpt make serve            # uvicorn on :8000
# or
docker compose up --build
```

| Endpoint | Purpose |
|---|---|
| `POST /predict` | class probabilities, aggregated malignant probability, predictive entropy, optional MC-dropout epistemic term, guard-rail warnings |
| `POST /explain` | everything in `/predict` plus a base64 Grad-CAM++ overlay PNG |
| `GET /model-card` | machine-readable intended-use / limitations |
| `GET /health` | liveness |

A minimal browser client lives in [`frontend/index.html`](frontend/index.html).

## Evaluation output

`scripts/evaluate.py` writes to `artifacts/eval/`:

- `report.json` — all metrics + bootstrap CIs + per-class sensitivity/specificity + confusion matrix
- `calibration.json` — temperature, ECE pre/post
- `selective.json` — AURC, excess-AURC, risk at 80 % coverage, mean entropy
- `operating_point.json` — malignant-referral threshold at target sensitivity
- `confusion_matrix.png`, `reliability.png`, `risk_coverage.png`
- `RESULTS.md` — a ready-to-paste table

See [`docs/RESULTS.md`](docs/RESULTS.md) for the template and how to report.

## Repository layout

```
src/skinlesion/
  data/         splits (group K-fold), Dataset, albumentations, LightningDataModule
  models/       timm backbones, LightningModule, EMA, MixUp/CutMix
  losses/       class-balanced focal, soft-target CE
  metrics/      bootstrap CIs, per-class sens/spec, DeLong; ECE + temperature scaling
  uncertainty/  MC-dropout, entropy/BALD, energy OOD, risk-coverage
  explain/      Grad-CAM family (CNN + ViT)
  serve/        inference (TTA, MC-dropout) + FastAPI
  utils/        seeding, config, logging
scripts/        download · make_synthetic_data · prepare_splits · train · evaluate · explain · export
configs/        config.yaml + data/ model/ experiment/ groups
tests/          data, models, losses, metrics, calibration, config, api, end-to-end pipeline
```

## Testing

```bash
make test                 # full suite
pytest -m "not slow"      # skip the Lightning end-to-end tests
make lint type            # ruff + mypy (both gate CI)
```

CI (`.github/workflows/ci.yml`) runs ruff + `ruff format --check` + mypy, the unit
suite, and a **full synthetic end-to-end train→evaluate→export** on every push.

## Design notes & limitations

- **HAM10000 is small and biased** (single-site, mostly fair skin, `nv`-dominated).
  Reported numbers do not transfer to clinical populations or to non-dermoscopic
  photography. The dataset card ([`docs/DATASET.md`](docs/DATASET.md)) spells this out.
- **No bounding boxes** in HAM10000, so Grad-CAM is evaluated qualitatively only.
- Temperature scaling calibrates *confidence*, not the decision threshold — the
  operating-point analysis is separate and deliberately conservative.
- The synthetic generator exists purely to make the pipeline runnable in CI; it
  is not a data-augmentation or pre-training scheme.

## Citation

If this code is useful in academic work, please cite it via
[`CITATION.cff`](CITATION.cff), and cite the underlying dataset:

> Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset*, Sci. Data 5, 180161 (2018).

## License

MIT — see [`LICENSE`](LICENSE).
