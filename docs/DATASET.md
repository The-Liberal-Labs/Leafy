# Dataset Notes

## Current raw dataset

- Raw classes: `115`
- Raw images: `141,867`
- Raw min/max class count: `22 / 13,158`
- Raw imbalance ratio: about `598x`

## Confirmed duplicate or conflicting labels

- `Bell_pepper___bacterial_spot` vs `Pepper_bell__bacterial_spot`
- `Bell_pepper___healthy` vs `Pepper_bell__healthy`
- `Tomato___leaf_curl` vs `Tomato__yellow_leaf_curl_virus`

These are not harmless naming differences. They fragment supervision across separate folders for the same semantic disease label and create avoidable confusion in training and evaluation.

## Weak classes from the last EfficientNetV2S report

Lowest-impact problem groups from `EfficientNetV2S/logs/classification_report.txt`:

- `Pepper_bell__bacterial_spot`: precision `0.2667`, recall `0.0370`
- `Pepper_bell__healthy`: precision `0.3478`, recall `0.1032`
- `Pepper_bell___bacterial_spot`: precision `0.4718`, recall `0.9109`
- `Pepper_bell___healthy`: precision `0.4724`, recall `0.8054`
- `Cassava___bacterial_blight`: precision `0.4800`, recall `0.8727`
- `Cassava___brown_streak_disease`: precision `0.5916`, recall `0.7045`
- `Cassava___healthy`: precision `0.5897`, recall `0.6602`
- `Coffee___red_spider_mite`: precision `0.6111`, recall `0.6111`

The pepper classes are the clearest taxonomy issue. Cassava is a harder biological separation problem and likely needs better data quality, better class cleanup, or hierarchical training.

## Recommended curated profile

The default curation profile in `configs/dataset_curation/default_leafy_curation.json` does this:

- Merges bell-pepper aliases into canonical `Pepper_bell___...` labels
- Merges `Tomato__yellow_leaf_curl_virus` into `Tomato___leaf_curl`
- Drops very small or low-value classes that have already been unstable in prior runs
- Drops the `Pepper_bell__...` duplicate pair that acts as dead-weight supervision

This gives you a reproducible curated taxonomy without touching the original `data/` folder.

## Recommended workflow

1. Audit raw data with `python data_preparation/dataset_audit.py --data-dir ./data`
2. Clean corrupt files with `python data_preparation/clean_dataset.py --data-dir ./data --delete`
3. Build curated raw data with `python data_preparation/curate_dataset.py --source-data-dir ./data --output-dir ./data_curated --overwrite`
4. Create splits with `python data_preparation/prepare_dataset.py --source-data-dir ./data_curated --output-dir ./new_data_curated --overwrite`

This is safer than deleting folders in-place and makes the training dataset state reproducible.
