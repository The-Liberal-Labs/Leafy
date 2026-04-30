# Leafy Research Paper Guide

This document is the **technical reference for writing the research paper**. It covers the problem formulation, system architecture, imbalance strategies, dataset characteristics, evaluation framework, and experimental design — everything you need to write each section of the paper.

---

## 1. Problem Statement

**Task**: Fine-grained multi-class classification of plant leaf images into 90 classes spanning 27 plant species and their disease/pest/healthy conditions.

**Core challenge**: Severe natural class imbalance (64× ratio between max and min class). Standard cross-entropy training favors majority classes, producing degenerate models that achieve high overall accuracy by ignoring minority classes.

**Research question**: What combination of loss functions, sampling strategies, and augmentation techniques best handles extreme class imbalance in fine-grained leaf pathology classification? How do these strategies trade off between overall accuracy and per-class (macro-averaged) performance?

---

## 2. Dataset

### 2.1 Composition

| Property | Value |
|----------|-------|
| Total images | 137,186 |
| Classes | 90 |
| Plant species | 27 |
| Condition types | fungal, bacterial, viral, pest, healthy |
| Split | 80/10/10 (train/val/test) |
| Imbalance ratio | 64:1 |
| Cross-split leakage | 0 exact duplicate groups |
| Naming | `Species___condition` |

### 2.2 Class Distribution Characteristics

- **Heavy-tailed**: Cassava mosaic (13,158 images) vs Watermelon healthy (205 images)
- **Inter-species imbalance**: Rose (14,910 images) vs Mango (265 images)
- **Intra-species imbalance**: Tomato leaf curl (8,571) vs Tomato mosaic virus (373) within the same species
- **Per-plant class counts**: Range from 1 class (Blueberry, Guava, etc.) to 10 (Tomato)

### 2.3 Data Sources

The dataset is curated from publicly available plant disease repositories, primarily Kaggle's plant pathology datasets. Images were deduplicated, validated for integrity, and split with cross-split duplicate-hash grouping to ensure zero train/val/test leakage.

---

## 3. System Architecture

### 3.1 Training Pipeline (Two-Stage Transfer Learning)

```
Input: 90-class ImageFolder dataset (data_split/)
    │
    ├─[Stage 1: Feature Extraction]── 8 epochs ──┐
    │   Backbone: FROZEN                         │
    │   Head: trained from scratch                │
    │   Image size: 224×224                       │
    │   Augmentation: Mixup(α=0.4) + CutMix(α=1.0)│
    │   Scheduler: CosineAnnealingWarmRestarts     │
    │   LR: determined by LR Finder               │
    │                                             │
    ├─[LR Finder]── re-run with unfrozen backbone ┤
    │                                             │
    └─[Stage 2: Fine-Tuning]── 60 epochs ─────────┘
        Backbone: UNFROZEN (discriminative LR)
        Head LR: 20× backbone LR
        Image size: 224-384× (hardware-dependent)
        Warmup: 5 epochs linear
        Augmentation: Mixup(α=0.2) only
        Scheduler: CosineAnnealingLR
        │
        ├─[Validation TTA Gate]─── compare std vs TTA ──┐
        │                                                │
        └─[Test Evaluation]── with/without TTA ──────────┘
            │
            └─[Export]── PTH + ONNX + configs
```

### 3.2 Backbone Architectures

Four ImageNet-pretrained CNNs with custom classification heads:

**EfficientNet-V2-S** (~21M params): Primary architecture. Compound-scaled EfficientNet with Fused-MBConv blocks and SiLU activation. Chosen for its strong accuracy-efficiency tradeoff on fine-grained tasks.

**EfficientNet-B0** (~5.3M): Ablation baseline — tests whether V2 improvements matter.

**MobileNet-V3-Large** (~5.5M): Mobile deployment candidate using hard-swish activation and squeeze-excitation.

**ConvNeXt-Tiny** (~29M): Non-EfficientNet comparison — modernized ResNet with depthwise convolutions, LayerNorm, and GELU.

### 3.3 Classification Head Design

**EfficientNet heads**: Global pooling → Dropout(0.3) → 1280→512 Linear → BatchNorm → SiLU → Dropout(0.1) → 512→90 Linear. The intermediate bottleneck (512-dim) provides regularization; SiLU gives smooth gradients for the unbalanced head.

**MobileNet head**: 960→1280 Linear → Hardswish → Dropout(0.2) → 1280→90 Linear. Expands features before compression — follows MobileNet-V3's head pattern.

**ConvNeXt head**: Direct replacement of the final Linear(768→90). No extra layers — ConvNeXt's layer norm already provides sufficient regularization.

---

## 4. Imbalance Handling Strategies

This is the **core comparative analysis** of the paper. Seven strategies are implemented as orthogonal combinations of three mechanisms:

### 4.1 Mechanism 1: WeightedRandomSampler

Inverse-frequency sampling: class *i* is sampled with probability proportional to `1 / count(i)`, capped at `max_multiplier × uniform_probability`. The cap (default 3×) prevents pathological over-sampling of tiny classes (which would otherwise be seen 60× more than large classes).

**Formulation**: For a class with count *n_i*:
```
weight(i) = min(1/n_i, max_multiplier × 1/num_classes)
probability(i) = weight(i) / Σ weight(j)
```

### 4.2 Mechanism 2: Effective Number of Samples (ENS)

From Cui et al. (CVPR 2019). Models diminishing marginal benefit of additional samples:

**Formulation**:
```
E_n = (1 - β^n) / (1 - β)
```
where β = 0.999, and *n* is the observed sample count. The class weight is:
```
w_i = 1 / E_n(i)
```

A class with n=205 samples has E_n ≈ 185.5 (90.5% of asymptotic). A class with n=13,158 has E_n ≈ 998.9. The weight ratio is only 998.9/185.5 ≈ 5.4× compared to the raw count ratio of 64× — effectively compressing the imbalance.

### 4.3 Mechanism 3: Focal Loss

From Lin et al. (ICCV 2017). Reshapes the cross-entropy loss to focus on hard examples:

**Formulation**:
```
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
```
where p_t is the model's estimated probability for the true class, γ controls focusing strength, and α_t is a per-class balancing factor (derived from ENS weights when combined).

**Hyperparameters**: γ=1.5 (Stage 1, stronger focus), γ=0.5 (Stage 2, gentler), α derived from ENS.

### 4.4 Seven Strategies

| Strategy | Sampler | ENS Loss | Focal Loss | Label Smoothing |
|----------|:-------:|:--------:|:----------:|:---------------:|
| `none` | | | | 0.05 → 0.02 |
| `ens_loss` | | ✓ | | 0.05 → 0.02 |
| `sampler` | ✓ | | | 0.05 → 0.02 |
| `focal` | | | ✓ | 0.05 → 0.02 |
| `sampler_ens` | ✓ | ✓ | | 0.05 → 0.02 |
| `sampler_focal` | ✓ | | ✓ | 0.05 → 0.02 |
| `sampler_ens_focal` | ✓ | ✓ | ✓ | 0.05 → 0.02 |

### 4.5 Additional Regularization

**Mixup** (Zhang et al., ICLR 2018): Convex combination of pairs. α=0.4 (S1), α=0.2 (S2). Probability decays linearly from initial to final value within each stage.

**CutMix** (Yun et al., ICCV 2019): Region-level mixing. α=1.0, used only in Stage 1.

**Label Smoothing** (Szegedy et al., CVPR 2016): 0.05 (S1), 0.02 (S2). Prevents 1.0 probability assignments — critical for imbalanced classes where confident-but-wrong predictions on minority classes are masked by correct majority predictions.

---

## 5. Training Details

### 5.1 Optimization

- **Optimizer**: AdamW (weight_decay=0.01, β₁=0.9, β₂=0.999)
- **Gradient clipping**: norm=1.0
- **Gradient accumulation**: enabled when effective batch sizes exceed hardware limits
- **Mixed precision**: AMP with gradient scaling (reduces VRAM ~40%)
- **Memory format**: channels_last for tensor-core acceleration

### 5.2 Learning Rate Schedule

**LR Finder** (Leslie Smith, 2018): Exponential LR sweep from 1e-7 to 10. The suggested LR is chosen by the steepest-gradient heuristic.

**Stage 1**: CosineAnnealingWarmRestarts with T_0 = steps_per_epoch, T_mult = 1. Restarts every epoch — gives the newly-initialized head periodic escape from local minima.

**Stage 2**: CosineAnnealingLR with 5-epoch linear warmup. Peak LR from LR Finder, final LR = peak_LR / 100.

**Discriminative LR**: Head parameters receive 20× the backbone LR in Stage 2.

### 5.3 Augmentation Pipeline

**Training (Stage 1)**:
1. RandomResizedCrop(224, scale=(0.5, 1.0))
2. RandAugment(N=2, M=9)
3. HorizontalFlip(p=0.5)
4. VerticalFlip(p=0.2) — plant leaves can be photographed from any angle
5. RandomRotation(degrees=30)
6. RandomAffine(degrees=0, translate=(0.1, 0.1))
7. ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
8. RandomErasing(p=0.2)
9. Normalize(dataset_mean, dataset_std)

**Training (Stage 2)**: Same as Stage 1 but without CutMix, with gentler Mixup.

**Validation/Test**: Resize(256) → CenterCrop(image_size) → Normalize. Deterministic for reproducibility.

---

## 6. Evaluation Protocol

### 6.1 Primary Metrics

| Metric | Formula | Why |
|--------|---------|-----|
| **Macro F1** | mean(F1_i) over i=1..90 | Treats every class equally — the primary selection metric |
| Balanced Accuracy | mean(recall_i) over i=1..90 | Same spirit as macro F1 but simpler interpretation |
| Weighted F1 | Σ(w_i × F1_i) where w_i = support_i/Σ(support) | Reflects real-world deployment but favors majority classes |
| Accuracy | correct / total | Reported for completeness; inadequate for imbalanced tasks |

### 6.2 Secondary Analysis

- **Per-class precision/recall/F1**: Full sklearn classification_report (90 rows)
- **Confusion matrix**: 90×90 with plant-species grouping heatmaps
- **Worst-class analysis**: Top-k lowest-F1 classes with confusion patterns
- **Low-support behavior**: Classes with <50 validation samples analyzed separately
- **Per-plant error patterns**: Breakdown by plant species to identify cross-species confusion vs within-species confusion

### 6.3 Model Selection

Checkpoint selection is by `--selection-metric` (default: `val_macro_f1`). The best model on the validation set is used for final test evaluation — no test-set tuning.

### 6.4 TTA (Test-Time Augmentation)

TTA is gated: the script first evaluates standard vs TTA predictions on the validation set. TTA is only applied to the final test set if it improves the selection metric on validation. This prevents TTA from being blindly applied when it would degrade performance.

TTA strategy: 5-crop (4 corners + center) × horizontal flip = 10 predictions per image, averaged.

---

## 7. Experimental Design (Comparative Analysis)

### 7.1 Imbalance Strategy Comparison

**Primary experiment**: Train all 7 strategies under identical conditions (same seed, data, hyperparameters except the strategy-specific ones). Compare:

1. **Macro F1** — the primary metric for imbalanced classification
2. **Balanced accuracy** — confirms macro F1 trends
3. **Worst-class F1** — how well does each strategy protect the smallest classes?
4. **Accuracy** — included to quantify the accuracy-fairness tradeoff
5. **Per-plant breakdown** — do strategies help uniformly across species or only for certain plants?

### 7.2 Architecture Comparison

Train all 4 architectures with the best imbalance strategy from 7.1. Compare macro F1, inference speed, and parameter efficiency.

### 7.3 Ablation Studies

- **Label smoothing sweep**: 0.0, 0.05, 0.1, 0.15, 0.2 — how much smoothing before accuracy degrades?
- **Focal γ sweep**: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5 — optimal focusing strength
- **ENS β sweep**: 0.9, 0.99, 0.999, 0.9999 — sensitivity to effective sample assumption
- **Sampler cap sweep**: 1.0, 2.0, 3.0, 5.0, ∞ — how much oversampling before overfitting?

### 7.4 Expected Results (Hypothesis)

- `ens_loss` should give the best macro F1 by softly reweighting without oversampling noise
- `focal` should improve hard-class performance but may hurt easy classes
- `sampler`-based strategies risk overfitting tiny classes unless capped
- `sampler_ens_focal` may over-regularize and converge slower
- Larger architectures (EfficientNet-V2-S) should benefit more from imbalance strategies since they have capacity to model minority-class features

---

## 8. Limitations

1. **Single-image classification**: No temporal or multi-view reasoning. Field diagnosis often uses multiple images over time.
2. **Lab/field gap**: Images come from controlled/public datasets — may not generalize to in-field smartphone photos with varying lighting, backgrounds, and angles.
3. **Class granularity**: "Diseased" categories (Cucumber, Jamun, Mango, Pomegranate) are underspecified — lump multiple conditions without distinction.
4. **Geographic bias**: Plant diseases vary by region. The dataset's geographic coverage is not documented.
5. **No severity estimation**: Only presence/absence of condition, not infection severity.
6. **Batch-dependent metrics**: Macro F1 computed per-batch during training is an approximation; full-dataset metrics are logged per epoch.

---

## 9. Reproducibility

- **Seed**: All random generators seeded (Python, NumPy, PyTorch, CUDA)
- **Dataset fingerprint**: SHA-256 of class counts stored in `dataset_fingerprint.json` and saved alongside model checkpoints
- **Environment snapshot**: Every training run records PyTorch version, CUDA version, GPU model, and Python version
- **Deterministic validation**: No random augmentations during evaluation
- **cudnn.benchmark = True**: Non-deterministic for speed, but seeded to control the initial state

---

## 10. Related Work Context

### 10.1 Imbalance Handling

- **Class-balanced loss** (Cui et al., CVPR 2019): Effective Number of Samples reweighting — our `ens_loss` strategy
- **Focal Loss** (Lin et al., ICCV 2017): Hard-example focusing — our `focal` strategy
- **LDAM Loss** (Cao et al., NeurIPS 2019): Label-distribution-aware margin — not implemented, could be future work
- **Balanced Softmax** (Ren et al., NeurIPS 2020): Adjusts logits by class prior — not implemented

### 10.2 Plant Disease Classification

- **PlantVillage dataset** (Hughes & Salathe, 2015): Original single-leaf lab images
- **Cassava Leaf Disease** (Kaggle competition, 2020): 5-class cassava disease classification
- **PlantDoc** (Singh et al., 2020): In-field plant disease dataset — higher domain gap

### 10.3 Transfer Learning for Fine-Grained Tasks

- **EfficientNet** (Tan & Le, ICML 2019): Compound scaling for accuracy-efficiency
- **ConvNeXt** (Liu et al., CVPR 2022): Modernized CNN design
- **Two-stage fine-tuning** is the de facto standard for transfer learning on small-medium datasets

---

## 11. Key Code References for Paper Writing

| Paper Section | Code Reference |
|---------------|---------------|
| Dataset statistics | `data_preparation/dataset_audit.py` |
| Split validation | `data_preparation/validate_split_dataset.py` |
| Training pipeline entry | `model_training/train_efficientnet.py` |
| ENS weights computation | `model_training/train_efficientnet.py:compute_ens_class_weights()` |
| Focal Loss implementation | `model_training/train_efficientnet.py:FocalLoss` class |
| WeightedRandomSampler | `model_training/train_efficientnet.py:build_sampler()` |
| Mixup/CutMix | `model_training/train_efficientnet.py:mixup_cutmix_batch()` |
| Architecture building | `model_training/train_efficientnet.py:build_model()` |
| Hardware auto-config | `model_training/train_efficientnet.py:get_hardware_profile()` |
| LR Finder integration | `model_training/train_efficientnet.py:run_lr_finder()` |
| Evaluation & metrics | `model_training/train_efficientnet.py:evaluate()` |
| TTA gating | `model_training/train_efficientnet.py:select_tta()` |
| Error review | `model_training/train_efficientnet.py:export_error_review()` |

---

## 12. Writing the Paper — Section-by-Section Notes

### Abstract
- 90-class plant disease classification, 27 species, 137K images
- 64× class imbalance as the core challenge
- 7-strategy comparative framework: ENS weights, Focal Loss, WeightedRandomSampler, and combinations
- Key finding placeholder: [best strategy] achieves X% macro F1, [Y]% balanced accuracy
- Two-stage transfer learning from ImageNet-pretrained EfficientNet-V2-S

### Introduction
- Plant disease detection as a food security challenge
- Prior work (PlantVillage, Cassava) limited to few classes or single species
- The challenge of real-world multi-species classification with extreme imbalance
- Our contributions: (1) 90-class multi-species dataset, (2) systematic 7-strategy imbalance comparison, (3) open-source reproducible pipeline

### Related Work
- Imbalance handling in deep learning (class reweighting, resampling, loss functions)
- Plant disease classification literature
- Transfer learning for fine-grained visual recognition

### Methods
- Use the architecture diagrams and strategy formulations from Sections 3-5 above
- Include the mathematical formulations for ENS, Focal Loss, and WeightedRandomSampler
- Describe the two-stage training and LR scheduling

### Experiments
- Use the experimental design from Section 7
- Report all 7 strategies with macro F1, balanced accuracy, accuracy
- Include per-plant breakdown and worst-class analysis
- Architecture comparison table
- Ablation studies

### Results
- Tables comparing all strategies (use `model_outputs/<run>/logs/` CSV files)
- Confusion matrix heatmaps (use `model_outputs/<run>/images/` plots)
- Per-class F1 bar charts highlighting worst-performing classes

### Discussion
- Why certain strategies outperform others
- Tradeoffs between accuracy and fairness
- Practical recommendations for imbalanced plant disease classification
- Limitations and future work

---

## 13. Commands to Generate Paper Tables & Figures

```bash
# Train all 7 strategies (run sequentially, ideally with screen/tmux)
for strategy in none ens_loss sampler focal sampler_ens sampler_focal sampler_ens_focal; do
    python model_training/train_efficientnet.py \
        --data-dir ./data_split \
        --imbalance-strategy $strategy \
        --output-dir model_outputs/strategy_${strategy} \
        --no-wandb
done

# Train all architectures
for arch in efficientnet_b0 mobilenet_v3_large convnext_tiny; do
    python model_training/train_efficientnet.py \
        --data-dir ./data_split \
        --architecture $arch \
        --imbalance-strategy ens_loss \
        --no-wandb
done
```

Training logs in `model_outputs/<run>/logs/full_training_log.txt` and per-epoch CSVs provide all metrics needed for tables and figures.
