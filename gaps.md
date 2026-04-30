# Leafy Remaining Gap List

Last reviewed: 2026-05-01

This file tracks work that is still open after making `data_split/` the primary dataset source.

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

## Completed

- Primary dataset metadata lives with `data_split/`
- Dataset docs generated from the split dataset
- Class naming standardized as `Species___condition`
- Old custom CNN trainer removed
- Training code under `model_training/`
- Root `train_efficientnet.py` wrapper removed — call `python model_training/train_efficientnet.py` directly
- Training outputs go to `model_outputs/`
- 7 explicit imbalance strategies supported
- Checkpoint selection supports `val_macro_f1`, `val_balanced_accuracy`, `val_acc`
- Per-epoch validation logs include macro F1, weighted F1, balanced accuracy
- Validation reports and worst-class CSVs saved during training
- TTA gated from validation performance before final test evaluation
- Model exports include class names, count, image size, normalization stats, fingerprint
- ONNX export with dependency message and `--export-only` retry
- Environment metadata saved in each training output directory
- Data-prep smoke tests cover split ratios and validation
- README with architecture diagram and full pipeline documentation
- GUIDE.md for research paper writing

## Remaining Gaps

### Gap 1: Full training has not been run on the current primary dataset

No model has been trained on the current `data_split/` dataset. 7-strategy and 4-architecture comparative experiments are pending.

**8GB RTX 4070 Laptop command:**

```bash
python model_training/train_efficientnet.py \
    --data-dir ./data_split \
    --architecture efficientnet_v2_s \
    --selection-metric val_macro_f1 \
    --imbalance-strategy ens_loss \
    --s1-batch 96 \
    --s2-batch 18 \
    --accum-steps-s2 3 \
    --s2-img-size 224 \
    --num-workers 4
```

**Deskstop/cloud 12GB+ GPU command:**

```bash
python model_training/train_efficientnet.py \
    --data-dir ./data_split \
    --architecture efficientnet_v2_s \
    --selection-metric val_macro_f1 \
    --imbalance-strategy ens_loss \
    --s2-img-size 300
```

Estimated training time: 3-6 hours on RTX 4070, 1-3 hours on 24GB+ GPUs.

Acceptance criteria:
- `model_outputs/EfficientNetV2S/` contains fresh checkpoints and logs
- `models/inference_config.json` has `num_classes = 90`
- Training summary reports accuracy, macro F1, weighted F1, balanced accuracy

### Gap 2: Compare all 7 imbalance strategies

```bash
for strategy in none ens_loss sampler focal sampler_ens sampler_focal sampler_ens_focal; do
    python model_training/train_efficientnet.py \
        --data-dir ./data_split \
        --imbalance-strategy $strategy \
        --output-dir model_outputs/strategy_${strategy} \
        --no-wandb
done
```

Acceptance criteria:
- Pick the final model by validation macro F1 and low-support class behavior
- Keep each run's dataset fingerprint and environment snapshot
- Generate comparison table for the paper

### Gap 3: Compare all 4 architectures

```bash
for arch in efficientnet_b0 mobilenet_v3_large convnext_tiny; do
    python model_training/train_efficientnet.py \
        --data-dir ./data_split \
        --architecture $arch \
        --imbalance-strategy ens_loss \
        --output-dir model_outputs/${arch} \
        --no-wandb
done
```

### Gap 4: Model-error review

The trainer can export mistake-review files after training:
- `model_outputs/<run>/logs/error_review_top_mistakes.csv`
- `reports/cassava_error_review.csv`
- `reports/pepper_error_review.csv`
- `reports/low_support_error_review.csv`

Acceptance criteria:
- Review high-confidence mistakes manually
- Correct or remove confirmed mislabeled images before publishing final claims

### Gap 5: Optional perceptual duplicate review

Exact duplicate leakage across splits is fixed (0 cross-split groups). Perceptual near-duplicate review to verify no visual-identical images exist across splits:

```bash
python data_preparation/deduplicate_dataset.py \
    --data-dir ./data_split \
    --include-perceptual \
    --report-json ./reports/duplicate_report_perceptual.json
```

### Gap 6: Final model card

After fresh training, record:
- Dataset fingerprint
- Architecture and image size
- Imbalance strategy
- Selection metric
- Accuracy, macro F1, weighted F1, balanced accuracy
- Worst classes by F1
- Low-support metric caution
- Known limitations and intended use

### Gap 7: Ablation studies

- Label smoothing sweep: 0.0, 0.05, 0.1, 0.15, 0.2
- Focal γ sweep: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5
- ENS β sweep: 0.9, 0.99, 0.999, 0.9999
- Sampler cap sweep: 1.0, 2.0, 3.0, 5.0

(Require modifying hyperparameters in `train_efficientnet.py` or adding CLI flags)
