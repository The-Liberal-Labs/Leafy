# Data Preparation

The primary dataset is `data_split/`. It already contains `train/`, `val/`,
and `test/` folders in PyTorch `ImageFolder` format. Normal workflow validates
the existing split rather than rebuilding it.

## Primary Workflow

```bash
# 1. Validate the existing split and refresh metadata
python data_preparation/validate_split_dataset.py \
    --data-dir ./data_split \
    --write-summary

# 2. Audit class counts and imbalance
python data_preparation/dataset_audit.py \
    --data-dir ./data_split \
    --top-k 25

# 3. Check image integrity (dry-run by default)
python data_preparation/clean_dataset.py \
    --data-dir ./data_split

# 4. Optional: perceptual duplicate check
python data_preparation/deduplicate_dataset.py \
    --data-dir ./data_split \
    --include-perceptual \
    --report-json ./reports/duplicate_report_perceptual.json
```

## Train After Validation

```bash
python model_training/train_efficientnet.py --data-dir ./data_split
```

## Scripts

| Script | What it does |
|--------|-------------|
| `validate_split_dataset.py` | Validates train/val/test class consistency, empty classes, split totals, and cross-split exact duplicate hashes |
| `dataset_audit.py` | Counts images per class from either a split dataset or a single class-folder dataset |
| `clean_dataset.py` | Validates every image with PIL; add `--delete` only if you want invalid files removed |
| `deduplicate_dataset.py` | Reports exact duplicate hashes; add `--include-perceptual` for average-hash grouping |
| `prepare_dataset.py` | Optional helper for creating `data_split/` from a future unsplit class-folder dataset |

## Commands

### validate_split_dataset.py

```text
--data-dir        Split dataset root (default: ./data_split)
--write-summary  Write split_summary.json, dataset_fingerprint.json, and class_names.json
```

### dataset_audit.py

```text
--data-dir      Dataset root (default: ./data_split)
--top-k         Number of highest/lowest classes to print (default: 15)
--report-json   Optional path to save the audit report
```

### clean_dataset.py

```text
--data-dir   Dataset root (default: ./data_split)
--workers    Override parallel worker count
--delete     Delete invalid files instead of dry-run
```

### deduplicate_dataset.py

```text
--data-dir              Dataset root (default: ./data_split)
--report-json           Output report path
--include-perceptual    Also compute average perceptual hashes
```

### prepare_dataset.py

Use this only if you later receive a new unsplit class-folder dataset.

```text
--source-data-dir       Folder with one subdirectory per class (default: ./class_folder_data)
--output-dir            Destination split dataset (default: ./data_split)
--train-ratio           Training split ratio (default: 0.80)
--val-ratio             Validation split ratio (default: 0.10)
--seed                  Random seed (default: 42)
--min-images-per-class  Skip classes below this count (default: 80)
--overwrite             Remove output directory before writing
--report-json           Optional extra path to save split summary
--no-hash-groups        Disable exact duplicate SHA-256 grouping
```

`prepare_dataset.py` writes `split_summary.json` and `dataset_fingerprint.json`.
The validator can refresh those files and add `class_names.json` afterward.
