# Model Training

Maintained training code lives in this package. The root `train_efficientnet.py`
file is only a compatibility wrapper.

Default workflow:

```bash
python data_preparation/validate_split_dataset.py --data-dir ./data_split --write-summary
python train_efficientnet.py --data-dir ./data_split --architecture efficientnet_v2_s --imbalance-strategy ens_loss
```

Supported backbones are intentionally limited to models that make sense for
leaf-image transfer learning:

- `efficientnet_v2_s`: primary accuracy baseline.
- `efficientnet_b0`: quick EfficientNet baseline.
- `mobilenet_v3_large`: faster mobile/deployment candidate.
- `convnext_tiny`: strong non-EfficientNet comparison.

Outputs are written under `model_outputs/<architecture>/` by default.
