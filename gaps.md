# Leafy Remaining Gap List

Last reviewed: 2026-05-01 (post code-review & bug-fix pass)

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

### Code review & bug fixes (2026-05-01)

- **Critical bug fixed**: CutMix never applied in `train_one_epoch()` (called `mixup_data` instead of `cutmix_data` at `train_efficientnet.py:1346`). Stage 1 now correctly alternates Mixup/CutMix as documented.
- **Bug fixed**: Color mapping in `plot_class_distribution` was wrong — `sorted_colors` used positional indices instead of sorted indices. Now the gradient correctly follows sorted counts.
- **Bug fixed**: `analyze_dataset()` counted non-image files (`.DS_Store`, `.json`, `.txt`) as training images. Now filters by image extensions only.
- **Bug fixed**: `create_train_eval_loader()` used `np.random.default_rng(42)` bypassing the global seed. Now uses `random.sample()` from the seeded global RNG for deterministic subset selection.
- **Bug fixed**: `train_stage()` CSV header logic used `not log_path.exists()` which was fragile across interrupted runs. Replaced with a `csv_needs_header` flag.
- **Bug fixed**: MobileNet-V3-Large BN/gamma parameters were not excluded from weight decay — `get_param_groups_no_decay` and `get_param_groups_discriminative_no_decay` used string heuristics (`"bn" in name`) that missed MobileNet-V3's `block.N.1.weight` naming. Replaced with proper module-type inspection (`isinstance(module, nn.BatchNorm2d)`). All 4 architectures now correctly exclude BN/norm weights and all biases from weight decay.
- **Cli flags added** for ablation studies (Gap 7): `--focal-gamma`, `--focal-gamma-s2`, `--ens-beta`, `--s1-label-smoothing`, `--s2-label-smoothing`. Defaults match previous hardcoded values (1.5, 0.5, 0.999, 0.05, 0.02).
- **Tests added**: 32 new tests in `tests/test_model_training.py` covering FocalLoss (6), build_criterion (7), Mixup (4), CutMix (3), ENS weights (3), WeightedSampler (4). All 34 tests pass (2 data-prep + 32 training).

### Previously completed

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

CLI flags added — no code changes needed. Run directly:

```bash
# Label smoothing sweep
for ls in 0.0 0.05 0.1 0.15 0.2; do
    python model_training/train_efficientnet.py \
        --data-dir ./data_split \
        --imbalance-strategy ens_loss \
        --s1-label-smoothing $ls --s2-label-smoothing $ls \
        --output-dir model_outputs/ablation_ls_${ls} --no-wandb
done

# Focal gamma sweep
for gamma in 0.0 0.5 1.0 1.5 2.0 2.5; do
    python model_training/train_efficientnet.py \
        --data-dir ./data_split \
        --imbalance-strategy focal \
        --focal-gamma $gamma --focal-gamma-s2 $gamma \
        --output-dir model_outputs/ablation_focal_${gamma} --no-wandb
done

# ENS beta sweep
for beta in 0.9 0.99 0.999 0.9999; do
    python model_training/train_efficientnet.py \
        --data-dir ./data_split \
        --imbalance-strategy ens_loss \
        --ens-beta $beta \
        --output-dir model_outputs/ablation_ens_${beta} --no-wandb
done

# Sampler cap sweep
for cap in 1.0 2.0 3.0 5.0; do
    python model_training/train_efficientnet.py \
        --data-dir ./data_split \
        --imbalance-strategy sampler \
        --sampler-max-multiplier $cap \
        --output-dir model_outputs/ablation_sampler_${cap} --no-wandb
done
```

### Gap 8: Known architectural gaps (future work, not blocking)

- No multi-GPU support (`DistributedDataParallel`)
- No mid-stage checkpoint resumption (Stage 2 can't resume from epoch 45/60)
- No standalone inference/prediction script
- No YAML/JSON configuration file support
- LR Finder re-runs on `--resume` for Stage 2
- Patience is fixed, no adaptive early stopping
