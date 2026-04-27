# Data Preparation

Tools for auditing, cleaning, curating, and splitting the Leafy dataset.

## Workflow

```bash
# 1. Audit — get a class-by-class image count overview
python data_preparation/dataset_audit.py --data-dir ./data

# 2. Clean — remove corrupted/unreadable images
python data_preparation/clean_dataset.py --data-dir ./data --delete

# 3. Curate — merge duplicate labels and drop weak classes (produces ./data_curated)
python data_preparation/curate_dataset.py \
    --source-data-dir ./data \
    --output-dir ./data_curated \
    --overwrite

# 4. Split — create train/val/test splits (80/10/10 by default)
python data_preparation/prepare_dataset.py \
    --source-data-dir ./data_curated \
    --output-dir ./data_split \
    --min-images-per-class 80 \
    --overwrite
```

**Then train with:**

```bash
python train_efficientnet.py --data-dir ./data_split
```

## Scripts

| Script | What it does |
|--------|-------------|
| `dataset_audit.py` | Counts images per class, shows imbalance ratio, identifies likely duplicate label groups |
| `clean_dataset.py` | Validates every image with PIL; use `--delete` to remove corrupted files in-place |
| `curate_dataset.py` | Merges duplicate class labels and drops weak classes per `configs/dataset_curation/default_leafy_curation.json` |
| `prepare_dataset.py` | Splits a class-folder dataset into `train/`, `val/`, `test/` subdirectories |

## Arguments

### dataset_audit.py
```
--data-dir         Directory with one subdirectory per class  (default: ./data)
--top-k           How many highest/lowest classes to print  (default: 15)
--report-json     Optional path to save the audit report as JSON
```

### clean_dataset.py
```
--data-dir    Dataset root directory                     (default: ./data)
--workers     Override parallel worker count             (default: None = auto)
--delete      Actually delete invalid files            (default: dry-run only)
```

### curate_dataset.py
```
--source-data-dir    Folder with one subdirectory per class   (default: ./data)
--output-dir        Curated output directory                (default: ./data_curated)
--config           JSON file with merge/drop rules         (default: ./configs/dataset_curation/default_leafy_curation.json)
--overwrite        Remove output directory before writing
--dry-run          Print the plan without copying files
```

The curation config (`configs/dataset_curation/default_leafy_curation.json`) defines:
- **`merge_into`**: source class → target class mappings (duplicates that should be merged)
- **`drop_classes`**: class names to permanently remove

### prepare_dataset.py
```
--source-data-dir       Folder with one subdirectory per class   (default: ./data)
--output-dir            Destination for split dataset           (default: ./data_split)
--train-ratio           Training split ratio                    (default: 0.80)
--val-ratio             Validation split ratio                  (default: 0.10)
--seed                  Random seed for splitting                (default: 42)
--min-images-per-class  Skip classes below this threshold        (default: 80)
--overwrite             Remove output directory before writing
--report-json           Optional path to save split summary as JSON
```

## Dropping additional classes

After auditing, if you see classes with very few images, you can permanently delete them from the raw data before splitting:

```bash
rm -rf ./data/ClassNameToDelete
```
