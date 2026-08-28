# Dataset Card — HAM10000

## Overview
- **Name:** HAM10000 ("Human Against Machine with 10000 training images").
- **Source:** Tschandl, P., Rosendahl, C., Kittler, H. (2018). *The HAM10000
  dataset, a large collection of multi-source dermatoscopic images of common
  pigmented skin lesions.* Scientific Data 5, 180161.
- **DOI:** 10.7910/DVN/DBW86T (Harvard Dataverse). Also distributed as ISIC-2018
  Task 3 and on Kaggle (`kmader/skin-cancer-mnist-ham10000`).
- **License:** CC BY-NC 4.0 (non-commercial, attribution).
- **Size:** 10,015 RGB dermoscopic images, 600×450 px, JPEG.

## Labels
Seven diagnostic categories (`dx`): `akiec, bcc, bkl, df, mel, nv, vasc`.
Ground-truth basis (`dx_type`): `histo` (histopathology, ~53 %), `follow_up`,
`consensus` (expert), `confocal`. Metadata also includes `age`, `sex`,
`localization`, and crucially `lesion_id`.

## Known biases and hazards
| Issue | Consequence for modelling |
|---|---|
| **Multiple images per lesion** (shared `lesion_id`) | Random splits leak; always split on `lesion_id` (this repo enforces it). |
| **Severe class imbalance** (`nv` ≈ 67 %, `df`/`vasc` ≈ 1 %) | Accuracy is misleading; use balanced accuracy, per-class sensitivity, class-balanced losses. |
| **Two acquisition sites**, limited hardware diversity | Domain shift to other clinics/dermatoscopes is unmodelled. |
| **Skin-tone distribution** skews to Fitzpatrick I–III | Expect degraded, unmeasured performance on darker skin. |
| **Selection bias** — lesions were imaged because they were clinically notable | Prevalence and difficulty do not match a screening population. |
| **Artefacts** — hair, ink, rulers, gel bubbles, vignetting | Models can shortcut on artefacts; inspect Grad-CAM. |

## Splits produced by this repo
`scripts/prepare_splits.py` runs `StratifiedGroupKFold(n_splits=5)` on
`lesion_id`, stratified by `dx`. Default: fold 0 = test, fold 1 = val, folds 2–4
= train. `split_summary.json` records per-fold class counts and asserts zero
lesion overlap between splits.

## Preprocessing
- Decode to RGB, resize shortest side to `round(1.14 · image_size)`, centre-crop
  to `image_size` for eval; `RandomResizedCrop` + dihedral flips + mild
  colour/affine + coarse dropout for training.
- Normalise with ImageNet mean/std (pretrained backbones).

## Obtaining the data
```bash
# Kaggle (needs ~/.kaggle/kaggle.json)
python scripts/download_ham10000.py --out data/ham10000
# or auth-free Harvard Dataverse
python scripts/download_ham10000.py --source dataverse --out data/ham10000
```
The script normalises everything into `data/ham10000/{metadata.csv, images/}`.

## Ethical use
De-identified, research-only, non-commercial. Do not attempt re-identification.
Do not upload identifiable patient photos to any demo built on this repo.
