# Leafy

Leafy is a plant pathology workspace for training multi-class plant disease classifiers on a clean, ready-to-use split dataset.

## Dataset

The primary dataset is `data_split/`. It is already organized for PyTorch `ImageFolder` training.

| Metric | Value |
|--------|------:|
| Classes | 90 |
| Total images | 137,186 |
| Train images | 109,736 |
| Validation images | 13,759 |
| Test images | 13,691 |
| Cross-split exact duplicate hash groups | 0 |

Class names follow `Species___condition`, for example `Tomato___early_blight` and `Potato___healthy`.

See [dataset_README.md](dataset_README.md) and [docs/DATASET.md](docs/DATASET.md) for the full class table, class descriptions, split counts, and Kaggle/GitHub dataset notes.

## Expected Structure

```text
data_split/
├── train/
│   └── Species___condition/
├── val/
│   └── Species___condition/
├── test/
│   └── Species___condition/
├── split_summary.json
└── dataset_fingerprint.json
```

## Setup

```bash
pip install -r requirements.txt

# Optional: use the tested local package versions
pip install -r requirements.lock.txt
```

## Validate The Dataset

```bash
python data_preparation/validate_split_dataset.py \
    --data-dir ./data_split \
    --write-summary

python data_preparation/dataset_audit.py \
    --data-dir ./data_split \
    --top-k 25

python data_preparation/clean_dataset.py \
    --data-dir ./data_split

python data_preparation/deduplicate_dataset.py \
    --data-dir ./data_split \
    --report-json ./reports/duplicate_report.json
```

## Train

Default strong run:

```bash
python train_efficientnet.py \
    --data-dir ./data_split \
    --architecture efficientnet_v2_s \
    --selection-metric val_macro_f1 \
    --imbalance-strategy ens_loss
```

Supported architectures:

| Architecture | Best for |
|-------------|---------|
| `efficientnet_v2_s` | Primary accuracy baseline |
| `efficientnet_b0` | Smaller EfficientNet experiment |
| `mobilenet_v3_large` | Faster/mobile candidate |
| `convnext_tiny` | Strong non-EfficientNet comparison |

Useful imbalance experiments:

```bash
python train_efficientnet.py --data-dir ./data_split --imbalance-strategy none
python train_efficientnet.py --data-dir ./data_split --imbalance-strategy ens_loss
python train_efficientnet.py --data-dir ./data_split --imbalance-strategy sampler
python train_efficientnet.py --data-dir ./data_split --imbalance-strategy focal
python train_efficientnet.py --data-dir ./data_split --imbalance-strategy sampler_focal
```

Outputs are written to `model_outputs/<architecture>/` by default.

## Data Preparation Scripts

| Script | Purpose |
|--------|---------|
| `validate_split_dataset.py` | Validates an existing `train/val/test` dataset and writes summary/fingerprint metadata |
| `dataset_audit.py` | Counts images per class, reports imbalance, and flags duplicate label groups |
| `clean_dataset.py` | Validates images with PIL; use `--delete` only when you want to remove invalid files |
| `deduplicate_dataset.py` | Reports exact duplicate hashes and optional perceptual average hashes |
| `prepare_dataset.py` | Optional utility for creating a split from a future unsplit class-folder dataset |

## Project Structure

```text
├── data_split/               # Primary train/val/test dataset
├── data_preparation/         # Dataset validation and utility scripts
├── model_training/           # Maintained training implementation
├── model_outputs/            # New training outputs
├── EfficientNetV2S/          # Previous run artifacts
├── train_efficientnet.py     # Compatibility wrapper
├── dataset_README.md         # Dataset card for Kaggle/GitHub
└── README.md
```
