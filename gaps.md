# Leafy Remaining Gap List

Last reviewed: 2026-05-01

This file now tracks only the work that is still open after making `data_split/`
the primary dataset source.

## Current Dataset State

| Item | Value |
|---|---:|
| Primary dataset directory | `data_split/` |
| Classes | 90 |
| Total images | 137,186 |
| Train images | 109,736 |
| Validation images | 13,759 |
| Test images | 13,691 |
| Cross-split exact duplicate hash groups | 0 |

The dataset is usable for training as-is. The deleted source folders are no
longer part of the active workflow.

## Completed And Removed From Active Gap Tracking

- Primary dataset metadata now lives with `data_split/`.
- Dataset docs are generated from the split dataset.
- Class naming is standardized as `Species___condition`.
- The stale root `class_names.json` is removed.
- The old custom CNN trainer is removed from the active workflow.
- Training code is separated under `model_training/`.
- The root `train_efficientnet.py` is a compatibility wrapper.
- Training outputs go to `model_outputs/`.
- The trainer supports explicit imbalance strategies.
- Checkpoint selection supports `val_macro_f1`, `val_balanced_accuracy`, and `val_acc`.
- Per-epoch validation logs include macro F1, weighted F1, and balanced accuracy.
- Validation reports and worst-class CSVs can be saved during training.
- TTA is selected from validation performance before final test evaluation.
- Model exports include class names, class count, image size, normalization stats, and dataset fingerprint.
- ONNX export now gives a clear dependency message and `--export-only` can retry export.
- Environment metadata is saved in each training output directory.
- Data-prep smoke tests cover split ratios and split validation.

## Remaining Gaps

### Gap 1: Full training has not been rerun on the current primary dataset

The previous saved model was trained on an older dataset state. A fresh run is
needed before making final model-performance claims for the current 90-class
`data_split/` dataset.

Command:

```bash
python train_efficientnet.py \
  --data-dir ./data_split \
  --architecture efficientnet_v2_s \
  --selection-metric val_macro_f1 \
  --imbalance-strategy ens_loss
```

Acceptance criteria:

- `model_outputs/EfficientNetV2S/` contains fresh checkpoints and logs.
- `logs/classification_report.txt` reports the current 90 classes.
- `models/inference_config.json` has `num_classes = 90`.
- Training summary reports accuracy, macro F1, weighted F1, and balanced accuracy.

### Gap 2: Compare imbalance strategies before choosing the final model

The code now supports controlled imbalance experiments, but the experiments have
not been run yet.

Recommended order:

```bash
python train_efficientnet.py --data-dir ./data_split --imbalance-strategy none
python train_efficientnet.py --data-dir ./data_split --imbalance-strategy ens_loss
python train_efficientnet.py --data-dir ./data_split --imbalance-strategy sampler
python train_efficientnet.py --data-dir ./data_split --imbalance-strategy focal
python train_efficientnet.py --data-dir ./data_split --imbalance-strategy sampler_focal
```

Acceptance criteria:

- Pick the final model by validation macro F1 and low-support class behavior,
  not only by overall accuracy.
- Keep the selected run's dataset fingerprint and environment snapshot.

### Gap 3: Model-error review still depends on a fresh trained model

The trainer can now export mistake-review files, but those files require a fresh
evaluation run.

Expected files after training:

- `model_outputs/<run>/logs/error_review_top_mistakes.csv`
- `reports/cassava_error_review.csv`
- `reports/pepper_error_review.csv`
- `reports/low_support_error_review.csv`

Acceptance criteria:

- Review high-confidence mistakes manually.
- Correct or remove confirmed mislabeled images in the dataset before publishing
  final model claims.

### Gap 4: Optional perceptual duplicate review

Exact duplicate leakage across train/validation/test is fixed. Perceptual
near-duplicate review is still optional and should be run if the final reported
test score looks unusually high.

Command:

```bash
python data_preparation/deduplicate_dataset.py \
  --data-dir ./data_split \
  --include-perceptual \
  --report-json ./reports/duplicate_report_perceptual.json
```

Acceptance criteria:

- Any suspicious near-duplicate groups are inspected.
- If needed, rebuild or manually adjust affected split groups.

### Gap 5: Final model card is still pending

After the fresh training run, add a model card that records:

- Dataset fingerprint.
- Architecture and image size.
- Imbalance strategy.
- Selection metric.
- Accuracy, macro F1, weighted F1, and balanced accuracy.
- Worst classes by F1.
- Low-support metric caution.
- Known limitations and intended use.
