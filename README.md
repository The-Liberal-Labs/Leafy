# Leafy

Leafy is a plant pathology workspace for training large multi-class leaf disease classification models.

## The Dataset

A curated collection of leaf images organized into **class folders** — each subdirectory represents one plant species paired with a disease or healthy condition.

### Current state

| Metric | Value |
|--------|-------|
| Classes | 101 |
| Total images | ~138,580 |
| Train/Val/Test | 80/10/10 splits |
| Sources | PlantVillage, Kaggle, and supplemental collections |

### Plant species covered

Apple, Blueberry, Cassava, Cherry, Chili, Coffee, Corn, Cucumber, Grape, Guava, Jamun, Lemon, Mango, Orange, Peach, Pepper (bell), Pomegranate, Potato, Raspberry, Rice, Rose, Soybean, Squash, Strawberry, Sugarcane, Tea, Tomato, Watermelon, Wheat

### Diseases and conditions

Each class follows the naming convention `Species___condition` (e.g., `Tomato___early_blight`, `Potato___healthy`). Conditions include bacterial blights, fungal diseases (rust, powdery mildew, rot, blight), viral infections (mosaic virus, leaf curl), pest damage, and healthy controls.

### Classes dropped during curation

13 classes with insufficient images were permanently removed:

`Coffee__cercospora_leaf_spot`, `Lemon__diseased`, `Pepper_bell__bacterial_spot`, `Pepper_bell__healthy`, `Potato___nematode`, `Soybean__bacterial_blight`, `Soybean__downy_mildew`, `Soybean__mosaic_virus`, `Soybean__powdery_mildew`, `Soybean__rust`, `Soybean__southern_blight`, `Sugarcane__red_stripe`, `Wheat__septoria`

### Duplicate labels merged

| Merged from | Merged into |
|-------------|-------------|
| `Bell_pepper___bacterial_spot` | `Pepper_bell___bacterial_spot` |
| `Bell_pepper___healthy` | `Pepper_bell___healthy` |
| `Tomato__yellow_leaf_curl_virus` | `Tomato___leaf_curl` |

### Expected structure

```
data/                    # raw class-folder dataset (101 classes, ~138,580 images)
├── Species___condition1/
│   ├── image001.jpg
│   └── image002.png
└── Species___condition2/
    └── image003.jpg

data_split/              # output of prepare_dataset.py
├── train/               # 80% of images
│   ├── Species___condition1/
│   └── Species___condition2/
├── val/                 # 10% of images
│   ├── Species___condition1/
│   └── Species___condition2/
└── test/                # 10% of images
    ├── Species___condition1/
    └── Species___condition2/
```

## Data Preparation

```bash
# 1. Audit — see image counts per class and check for imbalance
python data_preparation/dataset_audit.py --data-dir ./data

# 2. Clean — remove corrupted or unreadable images
python data_preparation/clean_dataset.py --data-dir ./data --delete

# 3. Split — create train/val/test splits
python data_preparation/prepare_dataset.py \
    --source-data-dir ./data \
    --output-dir ./data_split \
    --min-images-per-class 80 \
    --overwrite
```

**Then train:**

```bash
python train_efficientnet.py --data-dir ./data_split
```

## Training

```bash
python train_efficientnet.py --data-dir ./data_split
```

Supported architectures:

| Architecture | Best for |
|-------------|---------|
| `efficientnet_v2_s` | Best overall accuracy and robustness |
| `efficientnet_b0` | Lighter baseline for quick experiments |
| `mobilenet_v3_large` | Fastest training and deployment |
| `convnext_tiny` | Strong non-EfficientNet alternative |

```bash
python train_efficientnet.py --architecture efficientnet_v2_s --data-dir ./data_split
```

## Data Preparation Scripts

| Script | Purpose |
|--------|---------|
| `dataset_audit.py` | Count images per class, show imbalance ratio, flag duplicate label groups |
| `clean_dataset.py` | Validate images with PIL; use `--delete` to remove invalid files in-place |
| `prepare_dataset.py` | Split class-folder dataset into `train/`, `val/`, `test/` subdirectories |

See `data_preparation/README.md` for detailed arguments.

## Project Structure

```
├── data/                    # Raw class-folder dataset (101 classes)
├── data_preparation/        # Audit, clean, and split scripts
│   ├── dataset_audit.py
│   ├── clean_dataset.py
│   └── prepare_dataset.py
├── configs/                 # Configuration files
├── EfficientNetV2S/         # Saved models and training logs
├── train_efficientnet.py    # Main training entrypoint
└── README.md
```
