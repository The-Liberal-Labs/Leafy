# Model Training

Maintained training code lives in this package. The training script supports
7 imbalance strategies, 4 backbone architectures, and automatic hardware detection.

## Quick Start

```bash
# Default strong run
python model_training/train_efficientnet.py \
    --data-dir ./data_split \
    --architecture efficientnet_v2_s \
    --selection-metric val_macro_f1 \
    --imbalance-strategy ens_loss
```

## Supported Backbones

| Architecture | Best for |
|-------------|---------|
| `efficientnet_v2_s` | Primary accuracy baseline |
| `efficientnet_b0` | Quick EfficientNet baseline |
| `mobilenet_v3_large` | Faster mobile/deployment candidate |
| `convnext_tiny` | Strong non-EfficientNet comparison |

## Imbalance Strategies

Pass with `--imbalance-strategy`:

| Strategy | Sampler | ENS Loss | Focal Loss |
|----------|:-------:|:--------:|:----------:|
| `none` | | | |
| `ens_loss` | | ✓ | |
| `sampler` | ✓ | | |
| `focal` | | | ✓ |
| `sampler_ens` | ✓ | ✓ | |
| `sampler_focal` | ✓ | | ✓ |
| `sampler_ens_focal` | ✓ | ✓ | ✓ |

Default: `ens_loss` — favors macro-F1 without oversampling tiny classes.

## Outputs

Training outputs are written to `model_outputs/<architecture>/`:
- `images/` — LR finder plots, training curves, confusion matrices, class distribution
- `models/` — best_model.pth, model.onnx, class_names.json, inference_config.json
- `logs/` — full_training_log.txt, epoch CSV logs, error review CSVs

## Common Commands

```bash
# Dry run (1 epoch, ~5 min)
python model_training/train_efficientnet.py --data-dir ./data_split --dry-run

# Without W&B
python model_training/train_efficientnet.py --data-dir ./data_split --no-wandb

# Custom batch sizes & image sizes
python model_training/train_efficientnet.py --data-dir ./data_split \
    --s1-batch 64 --s2-batch 12 --accum-steps-s2 4 --s2-img-size 260

# Resume from Stage 2 checkpoint
python model_training/train_efficientnet.py --data-dir ./data_split --resume

# Export only
python model_training/train_efficientnet.py --data-dir ./data_split \
    --export-only model_outputs/EfficientNetV2S/models/best_model.pth
```

See [GUIDE.md](../GUIDE.md) for the full research paper guide.
