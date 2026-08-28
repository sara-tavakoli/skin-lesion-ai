# Model Card — Skin-Lesion AI

Following Mitchell et al., *Model Cards for Model Reporting* (2019).

## Model details
- **Task:** 7-way classification of pigmented skin lesions from dermoscopy images.
- **Classes:** `akiec` (actinic keratosis / intraepithelial carcinoma), `bcc`
  (basal cell carcinoma), `bkl` (benign keratosis), `df` (dermatofibroma),
  `mel` (melanoma), `nv` (melanocytic nevus), `vasc` (vascular lesion).
- **Architecture:** ImageNet-pretrained `timm` backbone (default EfficientNetV2-S)
  with a single linear head, dropout, stochastic depth; optional weight EMA.
- **Training:** AdamW, cosine schedule with warmup, class-balanced focal loss,
  effective-number class weights, balanced sampler, MixUp/CutMix, mixed precision.
- **Post-hoc:** temperature scaling fit on the validation fold.
- **Version:** see `CITATION.cff` / git tag. Each checkpoint embeds its config
  snapshot and environment (`artifacts/env.json`).

## Intended use
- **Intended:** methodological research on medical-image classification,
  calibration, uncertainty, and explainability; teaching; benchmarking.
- **Users:** ML researchers and students.

## Out-of-scope / prohibited use
- **Any clinical use** — diagnosis, screening, triage, treatment selection, or
  decision support for patients or clinicians.
- Use on image types outside the training distribution (clinical/macro
  photography, dermatopathology, non-skin images).
- Use as evidence of safety or efficacy for a medical device submission.

This model is **not FDA/CE cleared** and has **not** been validated on a
prospective clinical cohort.

## Factors
Performance is expected to vary by: skin phototype (HAM10000 skews light),
lesion size and location, dermatoscope hardware and magnification, image quality
(focus, lighting, hair, ink markings, bubbles), and class prevalence (the dataset
is ~67 % `nv`).

## Metrics
Reported on a held-out lesion-disjoint test fold:
- Balanced accuracy, macro AUROC, macro average precision, quadratic-weighted κ.
- Per-class sensitivity, specificity, precision, F1.
- All with bootstrap 95 % confidence intervals (2 000 resamples).
- Calibration: ECE / MCE and reliability diagram, before and after temperature scaling.
- Selective prediction: risk–coverage curve, AURC, excess-AURC.
- Malignant-vs-benign referral operating point at ≥ 0.95 sensitivity.

Fill in `docs/RESULTS.md` from `artifacts/eval/RESULTS.md` after training.

## Training data
HAM10000 (Tschandl et al., 2018) — 10,015 dermoscopic images, ~7,470 unique
lesions, collected at two sites (Austria, Australia). Ground truth by histopathology
(~53 %), follow-up, expert consensus, or confocal microscopy. See `docs/DATASET.md`.

## Ethical considerations
- **Automation bias:** a confident wrong prediction on a melanoma could delay
  care. The service returns entropy and explicit warning strings and refuses to
  present a single "diagnosis"; the operating-point analysis is tuned for
  sensitivity, accepting low specificity.
- **Representation:** under-representation of darker skin tones means error rates
  are almost certainly worse for those patients; this is a known, unmitigated
  limitation of the training data.
- **Privacy:** HAM10000 is de-identified and released for research; do not upload
  identifiable patient images to the demo service.

## Caveats and recommendations
- Recalibrate and re-establish operating points on any new data source.
- Report subgroup metrics whenever phototype/hardware metadata is available.
- Treat Grad-CAM as a qualitative sanity check, not a localisation guarantee.
