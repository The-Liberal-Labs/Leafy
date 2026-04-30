# Leafy

**Multi-class plant leaf disease classification with systematic imbalance handling — a research-grade training and evaluation pipeline for 90-class fine-grained plant pathology.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)

---

## Architecture & Pipeline Flow

```
                          ┌───────────────────────────────────────┐
                          │            data_split/                │
                          │  ┌─────────┐ ┌───────┐ ┌───────┐     │
                          │  │  train  │ │  val  │ │ test  │     │
                          │  │ 109,736 │ │13,759 │ │13,691 │     │
                          │  └────┬────┘ └───┬───┘ └───┬───┘     │
                          └───────┼──────────┼─────────┼─────────┘
                                  │          │         │
                    ┌─────────────┘          │         └──────────────┐
                    ▼                        ▼                        ▼
          ┌─────────────────┐    ┌───────────────────┐    ┌──────────────────┐
          │  Data Prep      │    │  TRAINING PIPELINE│    │  Final Test      │
          │  • validate     │    │                   │    │  Evaluation      │
          │  • audit        │    │  Stage 1: Feature │    │  • TTA selection │
          │  • clean        │    │     Extraction    │    │  • classification│
          │  • deduplicate  │    │     (frozen BB)   │    │    report        │
          └─────────────────┘    │  8 epochs, 224px  │    │  • confusion mat │
                                 │  Mixup + CutMix   │    │  • error review  │
                                 │  CosineWarmRestart│    └──────────────────┘
                                 │         │         │
                                 │         ▼         │
                                 │  Stage 2: Fine-   │
                                 │     Tuning        │
                                 │  (unfrozen all)   │
                                 │  60 epochs, 224px+│
                                 │  Discrim LR, warmup│
                                 │  Mixup-only       │
                                 │  CosineAnnealingLR│
                                 └────────┬──────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │    Imbalance Engine   │
                              │                       │
                              │  ┌───────────────────┐│
                              │  │   7 Strategies    ││
                              │  │                   ││
                              │  │  none             ││
                              │  │  ens_loss         ││
                              │  │  sampler          ││
                              │  │  focal            ││
                              │  │  sampler_ens      ││
                              │  │  sampler_focal    ││
                              │  │  sampler_ens_focal││
                              │  └───────────────────┘│
                              │                       │
                              │  ┌───────────────────┐│
                              │  │  Loss Functions   ││
                              │  │  • CrossEntropy   ││
                              │  │  • Focal Loss     ││
                              │  │  • ENS weights    ││
                              │  │  • Label Smoothing││
                              │  └───────────────────┘│
                              └───────────────────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │     Model Export      │
                              │  • PTH checkpoint     │
                              │  • ONNX runtime       │
                              │  • class_names.json   │
                              │  • inference_config   │
                              └───────────────────────┘
```

The pipeline implements **two-stage transfer learning** with an explicit imbalance-strategy system. Stage 1 freezes the backbone for feature extraction (Mixup + CutMix augmentation). Stage 2 unfreezes all layers with discriminative learning rates, linear warmup, and Mixup-only regularization. Between stages, an LR Finder (Leslie Smith method) determines the optimal learning rate. A validation TTA gate selects whether test-time augmentation helps, and only enables it if validation metrics improve.

---

## Dataset

| Metric | Value |
|--------|------:|
| Classes | 90 |
| Total images | 137,186 |
| Train images | 109,736 |
| Validation images | 13,759 |
| Test images | 13,691 |
| Naming convention | `Species___condition` |
| Cross-split exact duplicates | 0 |
| Imbalance ratio (max/min) | 64× |
| Min class | Watermelon\_\_\_healthy (205 images) |
| Max class | Cassava\_\_\_mosaic\_disease (13,158 images) |

Class names follow `Species___condition` (e.g., `Tomato___early_blight`, `Potato___healthy`). The dataset covers 27 plant species across fungal, bacterial, viral, and pest-related conditions. See [dataset_README.md](dataset_README.md) for the full class table and per-plant coverage statistics.

---

## Imbalance Strategies (Comparative Analysis Framework)

The project implements **7 controlled strategies** for handling class imbalance, enabling rigorous comparative analysis:

| Strategy | Sampler | Loss | Key Mechanism |
|----------|:-------:|:----:|---------------|
| `none` | None | CrossEntropyLoss | Baseline — no balancing |
| `ens_loss` | None | CrossEntropyLoss + ENS weights | Effective Number of Samples (Cui et al.) class weights; β=0.999 |
| `sampler` | WeightedRandomSampler | CrossEntropyLoss | Inverse-frequency sampling, capped at 3× max multiplier |
| `focal` | None | Focal Loss | Focal Loss with γ=1.5 (S1) / γ=0.5 (S2), per-class α weights |
| `sampler_ens` | WeightedRandomSampler | CrossEntropyLoss + ENS weights | Sampler + ENS loss weights combined |
| `sampler_focal` | WeightedRandomSampler | Focal Loss | Sampler + Focal Loss combined |
| `sampler_ens_focal` | WeightedRandomSampler | Focal Loss + ENS weights | Full stack: all three mechanisms |

**ENS (Effective Number of Samples)** — Cui et al., CVPR 2019. Each sample contributes `(1-βⁿ)/(1-β)` to its class's effective count, where n is the observed count. This prevents tiny classes from receiving extreme loss weights. β=0.999 means a class needs ~693 samples to reach half of its asymptotic effective count.

**WeightedRandomSampler** — Inverse of class frequency as sampling weight, capped at `max_multiplier=3.0` to avoid pathological over-sampling of tiny classes.

**Focal Loss** — Lin et al., ICCV 2017. Down-weights well-classified examples via `(1-p_t)^γ`, focusing gradient on hard and minority-class samples. Stage 1 uses γ=1.5 (stronger focus), Stage 2 uses γ=0.5 (gentler).

**Label Smoothing** — 0.05 for Stage 1, 0.02 for Stage 2. Prevents overconfidence on imbalanced classes.

**Default recommendation**: `ens_loss` — favors macro-F1 without oversampling tiny classes.

---

## Model Architectures

Four ImageNet-pretrained backbones with custom classification heads:

| Architecture | Key | Parameters | Best For |
|-------------|:---:|:---:|---|
| EfficientNet-V2-S | `efficientnet_v2_s` | ~21M | Primary accuracy baseline |
| EfficientNet-B0 | `efficientnet_b0` | ~5.3M | Quick experiments |
| MobileNet-V3-Large | `mobilenet_v3_large` | ~5.5M | Mobile deployment |
| ConvNeXt-Tiny | `convnext_tiny` | ~29M | Non-EfficientNet comparison |

**Custom head architectures** per backbone:
- **EfficientNet variants**: Dropout(0.3) → Linear(1280→512) → BatchNorm → SiLU → Dropout(0.1) → Linear(512→90)
- **MobileNet**: Linear(960→1280) → Hardswish → Dropout(0.2) → Linear(1280→90)
- **ConvNeXt**: Replace final Linear(768→90), no extra head layers

---

## Hardware Profiles (Auto-Detect)

The trainer auto-detects your GPU and selects batch/image-size profiles. For **RTX 4070 Laptop 8GB**:

| Stage | Batch × Accum | Effective | Image Size | Epochs |
|-------|:-----------:|:---------:|:----------:|:------:|
| Stage 1 | 128 × 1 | 128 | 224 | 8 |
| Stage 2 | 24 × 2 | 48 | 224 | 60 |

VRAM usage: ~6-7 GB peak (mixed precision enabled). You can manually tune with `--s1-batch`, `--s2-batch`, `--accum-steps-s1`, `--accum-steps-s2`, `--s2-img-size`.

Available profiles: `auto` (default), `mobile_8gb`, `rtx_12gb`, `t4_16gb`, `generic_16gb`, `generic_24gb`, `cpu`.

---

## Setup

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: lock file pins exact tested versions (reproducibility)
# pip install -r requirements.lock.txt

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Training Commands

### Default baseline (auto-configured for your hardware)

```bash
python model_training/train_efficientnet.py \
    --data-dir ./data_split \
    --architecture efficientnet_v2_s \
    --selection-metric val_macro_f1 \
    --imbalance-strategy ens_loss
```

### For 8GB RTX 4070 Laptop — optimized

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

### Dry run (1-epoch sanity check, ~5 min)

```bash
python model_training/train_efficientnet.py --data-dir ./data_split --dry-run
```

### Imbalance strategy comparison suite

```bash
# Run these sequentially for your comparative analysis:
python model_training/train_efficientnet.py --data-dir ./data_split --imbalance-strategy none
python model_training/train_efficientnet.py --data-dir ./data_split --imbalance-strategy ens_loss
python model_training/train_efficientnet.py --data-dir ./data_split --imbalance-strategy sampler
python model_training/train_efficientnet.py --data-dir ./data_split --imbalance-strategy focal
python model_training/train_efficientnet.py --data-dir ./data_split --imbalance-strategy sampler_focal
# Advanced combinations:
python model_training/train_efficientnet.py --data-dir ./data_split --imbalance-strategy sampler_ens
python model_training/train_efficientnet.py --data-dir ./data_split --imbalance-strategy sampler_ens_focal
```

### Architecture comparison

```bash
python model_training/train_efficientnet.py --data-dir ./data_split --architecture efficientnet_b0 --imbalance-strategy ens_loss
python model_training/train_efficientnet.py --data-dir ./data_split --architecture mobilenet_v3_large --imbalance-strategy ens_loss
python model_training/train_efficientnet.py --data-dir ./data_split --architecture convnext_tiny --imbalance-strategy ens_loss
```

### No W&B logging

```bash
python model_training/train_efficientnet.py --data-dir ./data_split --no-wandb
```

### Resume from Stage 2 checkpoint

```bash
python model_training/train_efficientnet.py --data-dir ./data_split --resume
```

### Export trained model without retraining

```bash
python model_training/train_efficientnet.py \
    --data-dir ./data_split \
    --export-only model_outputs/EfficientNetV2S/models/best_model.pth
```

Outputs go to `model_outputs/<architecture>/`:
```
model_outputs/EfficientNetV2S/
├── images/          # LR finder plots, training curves, confusion matrices
├── models/          # best_model.pth, model.onnx, class_names.json, inference_config.json
└── logs/            # full_training_log.txt, CSV per-epoch logs, error review CSVs
```

---

## Evaluation Metrics

The trainer computes and logs multiple metrics to evaluate imbalanced-class performance:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall top-1 accuracy (biased toward majority classes) |
| Macro F1 | Per-class F1 averaged equally across all 90 classes — **primary selection metric** |
| Weighted F1 | Per-class F1 weighted by support |
| Balanced Accuracy | Average recall across all classes |
| Per-class Precision/Recall/F1 | Full classification report per epoch |
| Confusion Matrix | 90×90 matrix with plant-species grouping |

Checkpoint selection uses `--selection-metric` (default: `val_macro_f1`). Macro F1 is chosen over accuracy because it treats every class equally — critical for a dataset with 64× imbalance.

---

## Validation & Data Quality

```bash
# Validate the split dataset and refresh metadata
python data_preparation/validate_split_dataset.py --data-dir ./data_split --write-summary

# Audit class counts and imbalance statistics
python data_preparation/dataset_audit.py --data-dir ./data_split --top-k 25

# Verify image integrity (dry-run by default)
python data_preparation/clean_dataset.py --data-dir ./data_split

# Optional: perceptual duplicate check
python data_preparation/deduplicate_dataset.py \
    --data-dir ./data_split \
    --include-perceptual \
    --report-json ./reports/duplicate_report_perceptual.json
```

The dataset has been validated with 0 cross-split exact duplicate hash groups. Perceptual near-duplicate review is optional but recommended before publishing final results.

---

## Project Structure

```
├── data_split/                      # Primary dataset (gitignored)
│   ├── train/                       # 109,736 images in 90 class folders
│   ├── val/                         # 13,759 images
│   ├── test/                        # 13,691 images
│   ├── class_names.json             # Sorted 90-class name list
│   ├── split_summary.json           # Per-class split counts
│   └── dataset_fingerprint.json     # SHA-256 fingerprint for reproducibility
├── data_preparation/                # Dataset validation & utilities
│   ├── validate_split_dataset.py    # Split validator + metadata generation
│   ├── dataset_audit.py             # Class count & imbalance audit
│   ├── clean_dataset.py             # PIL image integrity check
│   ├── deduplicate_dataset.py       # Exact + perceptual duplicate detection
│   └── prepare_dataset.py           # Create splits from unsplit data (one-time)
├── model_training/                  # Training implementation
│   └── train_efficientnet.py        # Full 2-stage pipeline (3K+ lines)
├── model_outputs/                   # Training outputs (gitignored)
├── tests/                           # Test suite
├── GUIDE.md                         # Research paper writing guide
├── dataset_README.md                # Dataset card with full class table
├── gaps.md                          # Remaining work tracker
├── requirements.txt                 # Dependencies (unpinned)
├── requirements.lock.txt            # Dependencies (pinned, tested versions)
└── README.md
```

---

## Key Design Decisions

1. **No hardcoded class count** — `num_classes` is auto-detected from `data_split/` at runtime
2. **Dataset-specific normalization** — Mean/std computed from 5000 training images, not ImageNet defaults
3. **Dataset fingerprinting** — SHA-256 of class counts stored alongside model for provenance
4. **TTA gating** — Test-time augmentation is only used if it improves validation metrics (prevents degradation)
5. **Per-epoch classification report** — Full sklearn `classification_report` every `--val-report-every` epochs
6. **Error review exports** — Top mistakes, per-plant error breakdowns, low-support class errors saved as CSVs
7. **Environment snapshot** — Every training run saves hardware info, git revision, and dataset fingerprint

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
