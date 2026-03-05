#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  🌿 LEAFY — Plant Disease Classification Training Pipeline             ║
║                                                                        ║
║  Architecture: EfficientNet-V2-S (Transfer Learning)                   ║
║  Strategy: 2-Stage (Feature Extraction → Fine-Tuning)                  ║
║  Features:                                                             ║
║    • Auto-detect NUM_CLASSES from data (zero hardcoding)               ║
║    • LR Finder with saved plots                                        ║
║    • Compute dataset mean/std (no ImageNet defaults if you want)       ║
║    • WeightedRandomSampler + ENS class weights                         ║
║    • Focal Loss (focuses on hard/minority samples)                     ║
║    • Mixup + CutMix (regularization for imbalanced data)               ║
║    • CosineAnnealingWarmRestarts with LR Finder best LR                ║
║    • Mixed precision + Gradient clipping + Gradient accumulation       ║
║    • WANDB monitoring                                                  ║
║    • All plots saved to EfficientNetV2S/images/                        ║
║    • Model export: PTH + ONNX                                         ║
║                                                                        ║
║  Usage:                                                                ║
║    python train_efficientnet.py                   # Full training       ║
║    python train_efficientnet.py --dry-run         # 1-epoch test        ║
║    python train_efficientnet.py --colab           # Colab T4 profile    ║
║    python train_efficientnet.py --no-wandb        # Skip W&B            ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import copy
import gc
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.transforms import v2 as T_v2
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("[WARN] wandb not installed. Run: pip install wandb")
    WANDB_AVAILABLE = False

try:
    from torch_lr_finder import LRFinder
    LR_FINDER_AVAILABLE = True
except ImportError:
    print("[WARN] torch-lr-finder not installed. Run: pip install torch-lr-finder")
    LR_FINDER_AVAILABLE = False


# ======================================================================
#                        CUDA DIAGNOSTICS
# ======================================================================

def print_cuda_diagnostics():
    """Print comprehensive CUDA runtime diagnostics."""
    print("\n" + "═" * 60)
    print("  🔧 SYSTEM & CUDA DIAGNOSTICS")
    print("═" * 60)

    print(f"  Python:        {sys.version.split()[0]}")
    print(f"  Platform:      {platform.system()} {platform.release()}")
    print(f"  PyTorch:       {torch.__version__}")
    print(f"  CUDA built:    {torch.version.cuda or 'N/A'}")
    print(f"  cuDNN:         {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print(f"  cuDNN enabled: {torch.backends.cudnn.enabled}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\n  GPU:           {props.name}")
        print(f"  VRAM:          {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute Cap:   {props.major}.{props.minor}")
        print(f"  SM Count:      {props.multi_processor_count}")
        print(f"  CUDA Runtime:  {torch.version.cuda}")

        # Quick benchmark
        x = torch.randn(1024, 1024, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            _ = x @ x
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        gflops = (100 * 2 * 1024**3) / elapsed / 1e12
        print(f"  Benchmark:     {gflops:.1f} TFLOPS (FP32 matmul)")
        del x
        torch.cuda.empty_cache()
    else:
        print("\n  ⚠️  No CUDA GPU detected — training will be SLOW on CPU")

    print("═" * 60)


# ======================================================================
#                    REPRODUCIBILITY
# ======================================================================

def seed_everything(seed=42):
    """Seed all random generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"  Seed set to {seed} (cudnn.benchmark=True for speed)")


# ======================================================================
#                      LOGGING (tee to file)
# ======================================================================

class TeeLogger:
    """
    Duplicates stdout to both console and a log file.
    All print() output is captured automatically.
    """
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")
        print(f"  📝 All output logged to: {log_path}")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


# ======================================================================
#                    FOLDER STRUCTURE
# ======================================================================

def create_output_dirs(base_name="EfficientNetV2S"):
    """
    Create a clean output folder structure:
        EfficientNetV2S/
        ├── images/      ← all plots (LR finder, training curves, etc.)
        ├── models/      ← checkpoints, PTH, ONNX
        └── logs/        ← CSV training logs
    Returns dict of paths.
    """
    base = Path(f"./{base_name}")
    dirs = {
        "base": base,
        "images": base / "images",
        "models": base / "models",
        "logs":   base / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n  📁 Output directory: {base}/")
    print(f"     images/  — plots & visualizations")
    print(f"     models/  — checkpoints, PTH, ONNX, class_names.json")
    print(f"     logs/    — CSV training logs")
    return dirs


# ======================================================================
#           DATASET ANALYSIS & PREPROCESSING
# ======================================================================

def analyze_dataset(data_dir):
    """
    Auto-detect NUM_CLASSES, print class distribution,
    and return class names and per-class counts.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            f"Run `python prepare_data.py` first to create train/val/test splits."
        )

    class_names = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')
    ])
    num_classes = len(class_names)

    # Count images per class
    train_counts = {}
    for cls in class_names:
        cls_path = os.path.join(train_dir, cls)
        count = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
        train_counts[cls] = count

    total_train = sum(train_counts.values())
    total_val = sum(
        len(os.listdir(os.path.join(val_dir, c)))
        for c in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, c))
    ) if os.path.isdir(val_dir) else 0
    total_test = sum(
        len(os.listdir(os.path.join(test_dir, c)))
        for c in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, c))
    ) if os.path.isdir(test_dir) else 0

    print(f"\n  📊 DATASET ANALYSIS")
    print(f"  {'─' * 40}")
    print(f"  Classes:      {num_classes} (auto-detected)")
    print(f"  Train images: {total_train:,}")
    print(f"  Val images:   {total_val:,}")
    print(f"  Test images:  {total_test:,}")
    print(f"  Total:        {total_train + total_val + total_test:,}")

    counts = list(train_counts.values())
    print(f"\n  Class size distribution (train):")
    print(f"    Min:    {min(counts):>5}  ({class_names[np.argmin(counts)]})")
    print(f"    Max:    {max(counts):>5}  ({class_names[np.argmax(counts)]})")
    print(f"    Mean:   {np.mean(counts):>7.0f}")
    print(f"    Median: {np.median(counts):>7.0f}")
    print(f"    Ratio (max/min): {max(counts)/max(min(counts),1):.0f}x imbalance")

    # Show bottom 10 and top 10 classes
    sorted_pairs = sorted(train_counts.items(), key=lambda x: x[1])
    print(f"\n  ⚠️  Bottom 10 classes (most at risk):")
    for name, cnt in sorted_pairs[:10]:
        print(f"    {cnt:>5} images — {name}")
    print(f"\n  ✅ Top 10 classes:")
    for name, cnt in sorted_pairs[-10:]:
        print(f"    {cnt:>5} images — {name}")

    return class_names, num_classes, train_counts


def plot_class_distribution(train_counts, class_names, save_path):
    """Plot and save class distribution bar chart."""
    counts = [train_counts[c] for c in class_names]

    fig, ax = plt.subplots(figsize=(24, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(counts)))
    sorted_indices = np.argsort(counts)
    sorted_counts = [counts[i] for i in sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in range(len(sorted_indices))]

    bars = ax.barh(range(len(sorted_counts)), sorted_counts, color=sorted_colors, edgecolor="none")

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=6)
    ax.set_xlabel("Number of Training Images", fontsize=12)
    ax.set_title("Class Distribution (sorted)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Mark median line
    median_val = np.median(sorted_counts)
    ax.axvline(x=median_val, color="red", linestyle="--", alpha=0.7, label=f"Median: {median_val:.0f}")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Class distribution plot saved: {save_path}")


def compute_dataset_stats(data_dir, sample_size=5000):
    """
    Compute the per-channel mean and std of the training set.
    Uses a random sample for efficiency.
    Returns (mean, std) as lists of 3 floats.
    """
    print("  Computing dataset mean & std (sampling images)...")
    train_dir = os.path.join(data_dir, "train")

    all_files = []
    for cls in os.listdir(train_dir):
        cls_dir = os.path.join(train_dir, cls)
        if os.path.isdir(cls_dir):
            for f in os.listdir(cls_dir):
                fp = os.path.join(cls_dir, f)
                if os.path.isfile(fp):
                    all_files.append(fp)

    if len(all_files) > sample_size:
        sampled = random.sample(all_files, sample_size)
    else:
        sampled = all_files

    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    for fpath in tqdm(sampled, desc="  Stats", leave=False):
        try:
            img = Image.open(fpath).convert("RGB")
            t = transform(img).numpy()  # (3, H, W)
            pixel_sum += t.sum(axis=(1, 2))
            pixel_sq_sum += (t ** 2).sum(axis=(1, 2))
            pixel_count += t.shape[1] * t.shape[2]
        except Exception:
            continue

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)

    print(f"  Dataset mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"  Dataset std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    return mean.tolist(), std.tolist()


# ======================================================================
#                DATA TRANSFORMS & LOADERS
# ======================================================================

def get_train_transforms(image_size, mean, std):
    """
    Best-practice training augmentation pipeline for plant pathology.
    Includes geometric, color, and erasing augmentations.
    """
    return transforms.Compose([
        # Geometric
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),

        # Color / Auto-augment
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),

        # To tensor + normalize with DATASET-SPECIFIC stats
        transforms.ToTensor(),
        transforms.Normalize(mean, std),

        # Regularization
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])


def get_eval_transforms(image_size, mean, std):
    """Deterministic eval/test transforms."""
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def plot_augmented_samples(dataset, class_names, save_path, n=16):
    """Show augmented training samples to verify preprocessing quality."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle("Augmented Training Samples", fontsize=16, fontweight="bold")

    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for idx, ax_idx in zip(indices, range(n)):
        img, label = dataset[idx]
        # Denormalize
        img_dn = img * std + mean
        img_np = np.clip(img_dn.permute(1, 2, 0).numpy(), 0, 1)

        ax = axes[ax_idx // 4][ax_idx % 4]
        ax.imshow(img_np)
        ax.set_title(class_names[label], fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  🖼️  Augmented samples saved: {save_path}")


def create_weighted_sampler(dataset):
    """WeightedRandomSampler for class imbalance."""
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets, minlength=len(dataset.classes)).astype(np.float64)

    # Inverse frequency
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[t] for t in targets]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(dataset),
        replacement=True,
    )
    return sampler


def compute_ens_class_weights(dataset, beta, num_classes, device):
    """Effective Number of Samples (ENS) class weights for the loss."""
    targets = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets, minlength=num_classes).astype(np.float64)

    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / weights.sum() * num_classes

    return torch.FloatTensor(weights).to(device)


def create_dataloaders(data_dir, image_size, batch_size, mean, std, num_workers):
    """Create train/val dataloaders with WeightedRandomSampler."""
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        get_train_transforms(image_size, mean, std),
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        get_eval_transforms(image_size, mean, std),
    )

    sampler = create_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )

    sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    return {"train": train_loader, "val": val_loader}, sizes, train_dataset


# ======================================================================
#                          MODEL
# ======================================================================

# ======================================================================
#                        FOCAL LOSS
# ======================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss — focuses training on hard, misclassified samples.
    Excellent for severely imbalanced datasets.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.register_buffer("weight", weight)
        self.ce = nn.CrossEntropyLoss(
            weight=weight, reduction="none", label_smoothing=label_smoothing,
        )

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.weight is not None:
            # Weight already applied by CE, just apply focal modulation
            pass

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ======================================================================
#                        MIXUP / CUTMIX
# ======================================================================

def mixup_data(x, y, alpha=0.4):
    """Mixup: blend two random samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix: cut a patch from one image and paste to another."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cy = np.random.randint(H)
    cx = np.random.randint(W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)  # Adjust lambda by actual area
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixed loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ======================================================================
#                          MODEL
# ======================================================================

def build_model(num_classes, device):
    """EfficientNet-V2-S with a stronger custom head."""
    print("\n  Loading EfficientNet-V2-S (ImageNet pretrained)...")
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # Stronger classification head (wider layers to handle 116 classes)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 768),
        nn.BatchNorm1d(768),
        nn.SiLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(768, 384),
        nn.BatchNorm1d(384),
        nn.SiLU(inplace=True),
        nn.Dropout(p=0.15),
        nn.Linear(384, num_classes),
    )

    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    return model


def unfreeze_backbone(model):
    """Unfreeze all backbone parameters."""
    for param in model.features.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  🔓 Backbone unfrozen: {trainable:,} / {total:,} params trainable ({100*trainable/total:.1f}%)")


def get_param_groups(model, lr_head, lr_backbone):
    """Discriminative learning rate groups."""
    return [
        {"params": model.features.parameters(), "lr": lr_backbone, "name": "backbone"},
        {"params": model.classifier.parameters(), "lr": lr_head, "name": "head"},
    ]


def get_param_groups_no_decay(model, lr, weight_decay):
    """
    Create param groups that EXCLUDE BatchNorm and bias from weight decay.
    Applying weight decay to BN/bias hurts convergence (known best practice).
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude BN weights, BN biases, and all biases from weight decay
        if "bn" in name or "batch" in name.lower() or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    print(f"  Weight decay groups: {len(decay_params)} with decay, {len(no_decay_params)} without")
    return [
        {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
    ]


def get_param_groups_discriminative_no_decay(model, lr_head, lr_backbone, weight_decay):
    """
    Discriminative LR + exclude BN/bias from weight decay.
    4 groups: backbone+decay, backbone+no_decay, head+decay, head+no_decay.
    """
    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = ("bn" in name or "batch" in name.lower() or "bias" in name)
        if name.startswith("features") or name.startswith("_orig_mod.features"):
            if is_no_decay:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
        else:
            if is_no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)

    print(f"  Param groups: backbone({len(backbone_decay)}+{len(backbone_no_decay)}), head({len(head_decay)}+{len(head_no_decay)})")
    return [
        {"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": head_decay, "lr": lr_head, "weight_decay": weight_decay},
        {"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0},
    ]


def try_compile(model):
    """torch.compile on Linux/Mac for speed."""
    if int(torch.__version__.split(".")[0]) >= 2 and platform.system() != "Windows":
        try:
            model = torch.compile(model)
            print("  ⚡ Model compiled with torch.compile()")
        except Exception as e:
            print(f"  torch.compile failed: {e}")
    return model


# ======================================================================
#                        LR FINDER
# ======================================================================

def run_lr_finder(model, train_loader, criterion, device, save_path,
                  start_lr=1e-7, end_lr=10, num_iter=200):
    """
    Run the LR range test, save the plot, and return the suggested LR.
    Uses the torch-lr-finder library for a proper implementation.
    """
    if not LR_FINDER_AVAILABLE:
        print("  ⚠️  torch-lr-finder not available. Using default LR.")
        return 1e-3

    print(f"\n  🔍 Running LR Finder ({num_iter} iterations, LR: {start_lr:.0e} → {end_lr:.0e})...")

    # Move original model to CPU to free GPU memory for the finder copy
    model.cpu()
    torch.cuda.empty_cache()

    # Create a fresh copy of the model for LR finding (so we don't corrupt weights)
    finder_model = copy.deepcopy(model)
    # Unwrap compiled model if needed
    if hasattr(finder_model, "_orig_mod"):
        finder_model = finder_model._orig_mod

    finder_model = finder_model.to(device)

    # Optimizer for the finder (only trainable params)
    trainable_params = [p for p in finder_model.parameters() if p.requires_grad]
    temp_optimizer = optim.AdamW(trainable_params, lr=start_lr, weight_decay=0.01)

    lr_finder = LRFinder(finder_model, temp_optimizer, criterion, device=device)

    lr_finder.range_test(
        train_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode="exp",
    )

    # Extract data
    losses = np.array(lr_finder.history["loss"])
    lrs = np.array(lr_finder.history["lr"])

    # Find the LR with steepest negative gradient (best practice)
    # Smooth the loss curve first
    from scipy.ndimage import uniform_filter1d
    try:
        smoothed = uniform_filter1d(losses, size=5)
        gradients = np.gradient(smoothed)
        min_grad_idx = np.argmin(gradients)
        suggested_lr = lrs[min_grad_idx]
    except:
        # Fallback: 1/10th of the LR at minimum loss
        min_loss_idx = np.argmin(losses)
        suggested_lr = lrs[min_loss_idx] / 10

    # Clip to reasonable range
    suggested_lr = np.clip(suggested_lr, 1e-6, 1e-1)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Learning Rate Finder", fontsize=14, fontweight="bold")

    # Loss vs LR
    ax1.plot(lrs, losses, color="steelblue", linewidth=1.5)
    ax1.axvline(x=suggested_lr, color="red", linestyle="--", linewidth=2,
                label=f"Suggested LR: {suggested_lr:.2e}")
    ax1.set_xscale("log")
    ax1.set_xlabel("Learning Rate (log scale)", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Loss vs Learning Rate", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Loss gradient
    try:
        ax2.plot(lrs[1:], np.gradient(losses)[1:], color="darkorange", linewidth=1.5)
        ax2.axvline(x=suggested_lr, color="red", linestyle="--", linewidth=2,
                    label=f"Steepest descent: {suggested_lr:.2e}")
        ax2.set_xscale("log")
        ax2.set_xlabel("Learning Rate (log scale)", fontsize=11)
        ax2.set_ylabel("Loss Gradient (dL/dLR)", fontsize=11)
        ax2.set_title("Loss Gradient (steepest = best LR)", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
    except:
        ax2.text(0.5, 0.5, "Gradient plot unavailable", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 LR Finder plot saved: {save_path}")
    print(f"  ✅ Suggested LR: {suggested_lr:.2e}")

    # Cleanup
    lr_finder.reset()
    del finder_model, lr_finder
    torch.cuda.empty_cache()

    # Move original model back to device
    model.to(device)

    return float(suggested_lr)


# ======================================================================
#                     TRAINING ENGINE
# ======================================================================

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device,
                    grad_clip, accum_steps=1, use_mixup=True, mixup_prob=0.5,
                    step_scheduler_per_batch=True):
    """
    Train for one epoch with:
    - Gradient accumulation (effective batch = batch_size * accum_steps)
    - Mixup / CutMix augmentation
    - Per-batch OR per-epoch scheduler stepping
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (inputs, labels) in enumerate(tqdm(loader, desc="  Train", leave=False)):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # --- Mixup / CutMix ---
        apply_mix = use_mixup and (random.random() < mixup_prob)
        if apply_mix:
            if random.random() < 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.4)
            else:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=1.0)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(inputs)
            if apply_mix:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)
            loss = loss / accum_steps  # Scale for accumulation

        scaler.scale(loss).backward()

        # Accumulate gradients, step every `accum_steps` batches
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Step per-batch schedulers (e.g. CosineAnnealingWarmRestarts)
            if step_scheduler_per_batch and scheduler is not None \
                    and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        _, preds = torch.max(outputs, 1)
        bs = inputs.size(0)
        running_loss += loss.item() * accum_steps * bs  # Undo scaling for logging
        running_corrects += (preds == labels).sum().item()
        total += bs

    return running_loss / total, running_corrects / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="  Val  ", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        bs = inputs.size(0)
        running_loss += loss.item() * bs
        running_corrects += (preds == labels).sum().item()
        total += bs

    return running_loss / total, running_corrects / total


def train_stage(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
                device, dirs, stage_name, num_epochs, patience, grad_clip,
                accum_steps=1, use_mixup=True, mixup_prob=0.5,
                step_scheduler_per_batch=True, wandb_run=None):
    """
    Full training loop for one stage (feature extraction OR fine-tuning).
    Saves checkpoints, CSV logs, and returns (model, history, best_acc).

    step_scheduler_per_batch:
        True  → scheduler.step() each optimizer step (use with CosineAnnealingWarmRestarts)
        False → scheduler.step() once per epoch (use with CosineAnnealingLR, ReduceLROnPlateau)
    """
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_acc = 0.0
    best_model_wts = None
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    checkpoint_path = dirs["models"] / f"best_model_{stage_name}.pth"
    log_path = dirs["logs"] / f"training_log_{stage_name}.csv"

    # Delete old log for this stage
    if log_path.exists():
        log_path.unlink()

    print(f"\n{'═' * 60}")
    print(f"  🏋️ {stage_name}: {num_epochs} epochs (patience={patience})")
    print(f"    Scheduler: {'per-batch' if step_scheduler_per_batch else 'per-epoch'}")
    print(f"    Mixup/CutMix: {'ON' if use_mixup else 'OFF'} (prob={mixup_prob})")
    print(f"{'═' * 60}")

    for epoch in range(num_epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, scheduler, scaler, device,
            grad_clip, accum_steps=accum_steps, use_mixup=use_mixup, mixup_prob=mixup_prob,
            step_scheduler_per_batch=step_scheduler_per_batch,
        )
        val_loss, val_acc = validate(model, dataloaders["val"], criterion, device)

        # Get current LR (from head param group if multiple)
        current_lr = optimizer.param_groups[-1]["lr"]

        # Step per-epoch schedulers (ReduceLROnPlateau or CosineAnnealingLR when not per-batch)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif not step_scheduler_per_batch and scheduler is not None:
            scheduler.step()

        dt = time.time() - t0

        # History
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # GPU memory usage
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        gpu_max = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        # Overfitting detection
        overfit_gap = train_acc - val_acc
        overfit_flag = " ⚠️OVF" if overfit_gap > 0.10 else ""

        # Print
        print(f"  Ep {epoch+1:02d}/{num_epochs} │ "
              f"TrLoss {train_loss:.4f}  TrAcc {train_acc:.4f} │ "
              f"VaLoss {val_loss:.4f}  VaAcc {val_acc:.4f} │ "
              f"LR {current_lr:.2e} │ GPU {gpu_mem:.1f}/{gpu_max:.1f}GB │ {dt:.0f}s{overfit_flag}")

        # W&B
        if wandb_run and WANDB_AVAILABLE:
            wandb.log({
                f"{stage_name}/train_loss": train_loss,
                f"{stage_name}/train_acc": train_acc,
                f"{stage_name}/val_loss": val_loss,
                f"{stage_name}/val_acc": val_acc,
                f"{stage_name}/lr": current_lr,
            })

        # CSV log
        row = pd.DataFrame([{
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr,
        }])
        row.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

        # Checkpoint + early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            best_model_wts = copy.deepcopy(raw.state_dict())
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_model_wts,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "accuracy": best_acc,
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"            ✅ Best model saved! Val Acc: {best_acc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  ⛔ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # Restore best
    if best_model_wts is not None:
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw.load_state_dict(best_model_wts)
        print(f"  Best weights restored (Val Acc: {best_acc:.4f})")

    return model, history, best_acc


# ======================================================================
#                     EVALUATION & PLOTS
# ======================================================================

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Return (labels, preds) arrays."""
    model.eval()
    all_preds, all_labels = [], []
    for inputs, labels in tqdm(dataloader, desc="  Evaluating"):
        inputs = inputs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def plot_training_history(hist_s1, hist_s2, save_path):
    """Plot combined training curves from both stages."""
    # Combine
    train_acc = hist_s1["train_acc"] + hist_s2["train_acc"]
    val_acc = hist_s1["val_acc"] + hist_s2["val_acc"]
    train_loss = hist_s1["train_loss"] + hist_s2["train_loss"]
    val_loss = hist_s1["val_loss"] + hist_s2["val_loss"]
    lrs = hist_s1["lr"] + hist_s2["lr"]
    s1_end = len(hist_s1["train_acc"])

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    # Accuracy
    ax = axes[0]
    ax.plot(train_acc, label="Train Acc", linewidth=2, marker="o", markersize=3)
    ax.plot(val_acc, label="Val Acc", linewidth=2, marker="s", markersize=3)
    ax.axvline(x=s1_end - 0.5, color="gray", linestyle="--", alpha=0.6, label="Stage 1→2")
    best_idx = int(np.argmax(val_acc))
    ax.annotate(f"Best: {val_acc[best_idx]:.4f}\n(Ep {best_idx+1})",
                xy=(best_idx, val_acc[best_idx]),
                xytext=(best_idx + 1, val_acc[best_idx] - 0.05),
                arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red")
    ax.set_title("Accuracy", fontsize=13); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(alpha=0.3)

    # Loss
    ax = axes[1]
    ax.plot(train_loss, label="Train Loss", linewidth=2, marker="o", markersize=3)
    ax.plot(val_loss, label="Val Loss", linewidth=2, marker="s", markersize=3)
    ax.axvline(x=s1_end - 0.5, color="gray", linestyle="--", alpha=0.6, label="Stage 1→2")
    ax.set_title("Loss", fontsize=13); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # LR schedule
    ax = axes[2]
    ax.plot(lrs, color="green", linewidth=2)
    ax.axvline(x=s1_end - 0.5, color="gray", linestyle="--", alpha=0.6, label="Stage 1→2")
    ax.set_title("Learning Rate Schedule", fontsize=13); ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Training history saved: {save_path}")


def plot_confusion_matrix(labels, preds, class_names, save_path, normalize=True):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    # For >50 classes, show a summary instead of full heatmap
    if len(class_names) > 50:
        # Per-class accuracy bar chart
        per_class_acc = np.diag(cm)
        sorted_idx = np.argsort(per_class_acc)

        fig, ax = plt.subplots(figsize=(12, max(8, len(class_names) * 0.15)))
        colors = plt.cm.RdYlGn(per_class_acc[sorted_idx])
        ax.barh(range(len(sorted_idx)), per_class_acc[sorted_idx], color=colors)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([class_names[i] for i in sorted_idx], fontsize=5)
        ax.set_xlabel("Per-Class Accuracy", fontsize=11)
        ax.set_title(f"Per-Class Accuracy (Mean: {np.mean(per_class_acc):.4f})", fontsize=13, fontweight="bold")
        ax.axvline(x=np.mean(per_class_acc), color="red", linestyle="--", alpha=0.7)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(20, 18))
        sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names,
                    cmap="Blues", fmt=".2f" if normalize else "d",
                    annot=False, ax=ax, linewidths=0.1)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        plt.xticks(fontsize=5, rotation=90); plt.yticks(fontsize=5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  📊 Confusion matrix saved: {save_path}")


def plot_per_class_metrics(labels, preds, class_names, save_path):
    """Plot per-class precision, recall, F1 sorted by F1."""
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds, labels=range(len(class_names)), zero_division=0
    )

    sorted_idx = np.argsort(f1)

    fig, ax = plt.subplots(figsize=(12, max(8, len(class_names) * 0.15)))
    y_pos = np.arange(len(class_names))

    ax.barh(y_pos - 0.2, prec[sorted_idx], height=0.2, label="Precision", color="#2196F3", alpha=0.8)
    ax.barh(y_pos, rec[sorted_idx], height=0.2, label="Recall", color="#FF9800", alpha=0.8)
    ax.barh(y_pos + 0.2, f1[sorted_idx], height=0.2, label="F1-Score", color="#4CAF50", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([class_names[i] for i in sorted_idx], fontsize=5)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title("Per-Class Precision / Recall / F1 (sorted by F1)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Per-class metrics saved: {save_path}")


def save_classification_report(labels, preds, class_names, save_path, wandb_run=None):
    """Print and save the classification report."""
    report_str = classification_report(labels, preds, target_names=class_names, digits=4)
    print("\n  📋 Classification Report:")
    print(report_str)

    with open(save_path, "w") as f:
        f.write(report_str)
    print(f"  Report saved: {save_path}")

    report_dict = classification_report(labels, preds, target_names=class_names, digits=4, output_dict=True)

    if wandb_run and WANDB_AVAILABLE:
        wandb.summary["Test Accuracy"] = report_dict["accuracy"]
        wandb.summary["Test Macro F1"] = report_dict["macro avg"]["f1-score"]
        wandb.summary["Test Weighted F1"] = report_dict["weighted avg"]["f1-score"]
        wandb.log({"Confusion Matrix": wandb.plot.confusion_matrix(
            preds=preds.tolist(), y_true=labels.tolist(), class_names=class_names,
        )})

    return report_dict


# ======================================================================
#                      MODEL EXPORT
# ======================================================================

def export_model(model, class_names, num_classes, image_size, dirs, device):
    """Export to PTH + ONNX."""
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw.eval()
    raw = raw.to(device)

    # PTH
    pth_path = dirs["models"] / "best_model.pth"
    torch.save({
        "model_state_dict": raw.state_dict(),
        "class_names": class_names,
        "num_classes": num_classes,
        "architecture": "efficientnet_v2_s",
        "image_size": image_size,
    }, pth_path)
    print(f"  💾 PTH saved: {pth_path}")

    # ONNX
    onnx_path = dirs["models"] / "model.onnx"
    dummy = torch.randn(1, 3, image_size, image_size).to(device)
    try:
        torch.onnx.export(
            raw, dummy, str(onnx_path), opset_version=17,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"  💾 ONNX saved: {onnx_path}")
    except Exception as e:
        print(f"  ⚠️  ONNX export failed: {e}")

    # Class names
    json_path = dirs["models"] / "class_names.json"
    with open(json_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"  💾 Class names saved: {json_path}")


# ======================================================================
#                           MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="🌿 Leafy Plant Disease Training")
    parser.add_argument("--dry-run", action="store_true", help="Quick 1-epoch test")
    parser.add_argument("--colab", action="store_true", help="Colab T4 profile")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")
    parser.add_argument("--data-dir", default="./new_data", help="Path to train/val/test splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from Stage 2 using saved Stage 1 checkpoint")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="W&B run ID to resume (use with --resume)")
    args = parser.parse_args()

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_cuda_diagnostics()
    seed_everything(args.seed)

    # ── Output folders ──
    dirs = create_output_dirs("EfficientNetV2S")

    # ── Tee logging: save all console output ──
    tee = TeeLogger(dirs["logs"] / "full_training_log.txt")
    sys.stdout = tee

    # ── Analyze dataset (auto-detect classes) ──
    class_names, num_classes, train_counts = analyze_dataset(args.data_dir)

    # Save class distribution plot
    plot_class_distribution(train_counts, class_names, dirs["images"] / "class_distribution.png")

    # Compute dataset-specific mean/std
    dataset_mean, dataset_std = compute_dataset_stats(args.data_dir)

    # ── Hardware-aware config (nothing hardcoded) ──
    # Target: 16GB VRAM (Colab T4 or local 16GB GPU).
    if args.colab:
        s1_batch, s2_batch, num_workers = 96, 24, 2
        accum_steps_s1, accum_steps_s2 = 1, 2    # Effective: 256, 256
    else:
        s1_batch, s2_batch, num_workers = 96, 24, 6
        accum_steps_s1, accum_steps_s2 = 1, 8    # Effective: 96, 96

    s1_img_size = 224
    s2_img_size = 224
    s1_epochs = 1 if args.dry_run else 10   # More epochs for solid feature extraction
    s2_epochs = 1 if args.dry_run else 40   # Long fine-tuning with early stopping
    patience = 999 if args.dry_run else 10  # Patient early stopping
    ens_beta = 0.999
    focal_gamma = 2.0                       # Focal loss gamma
    label_smoothing = 0.1
    weight_decay = 0.01
    grad_clip = 1.0

    print(f"\n  ⚙️  CONFIG (auto-configured)")
    print(f"  {'─' * 40}")
    print(f"  NUM_CLASSES:    {num_classes} (auto-detected)")
    print(f"  Stage 1:        {s1_epochs} ep, {s1_img_size}px, batch {s1_batch}×{accum_steps_s1} = {s1_batch*accum_steps_s1} effective")
    print(f"  Stage 2:        {s2_epochs} ep, {s2_img_size}px, batch {s2_batch}×{accum_steps_s2} = {s2_batch*accum_steps_s2} effective")
    print(f"  Loss:           FocalLoss (γ={focal_gamma}) + ENS weights + label_smoothing={label_smoothing}")
    print(f"  Augmentation S1: Mixup(α=0.4) + CutMix(α=1.0) @ 50% of batches (gentler for frozen backbone)")
    print(f"  Augmentation S2: Mixup(α=0.4) + CutMix(α=1.0) @ 80% of batches")
    print(f"  Weight decay:   {weight_decay} (excluded on BN/bias)")
    print(f"  Grad clip:      {grad_clip}")
    print(f"  Patience:       {patience} epochs")
    print(f"  Workers:        {num_workers}")
    print(f"  Dataset mean:   [{dataset_mean[0]:.4f}, {dataset_mean[1]:.4f}, {dataset_mean[2]:.4f}]")
    print(f"  Dataset std:    [{dataset_std[0]:.4f}, {dataset_std[1]:.4f}, {dataset_std[2]:.4f}]")

    # ══════════════════════════════════════════════════════════════
    #  RESUME CHECK
    # ══════════════════════════════════════════════════════════════
    s1_checkpoint_path = dirs["models"] / "best_model_Stage1_FeatureExtraction.pth"
    s1_log_path = dirs["logs"] / "training_log_Stage1_FeatureExtraction.csv"

    if args.resume:
        # ── RESUME MODE: Skip Stage 1, load checkpoint ──
        print("\n" + "═" * 60)
        print("  ⏩ RESUME MODE: Skipping Stage 1, loading checkpoint")
        print("═" * 60)

        if not s1_checkpoint_path.exists():
            print(f"  ❌ Stage 1 checkpoint not found: {s1_checkpoint_path}")
            print("  Run without --resume to train from scratch.")
            sys.exit(1)

        # Build model and load Stage 1 weights
        model = build_model(num_classes, device)
        checkpoint = torch.load(s1_checkpoint_path, map_location=device, weights_only=False)
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw.load_state_dict(checkpoint["model_state_dict"])
        best_acc_s1 = checkpoint.get("accuracy", 0.0)
        print(f"  ✅ Loaded Stage 1 checkpoint (Val Acc: {best_acc_s1:.4f})")
        model = try_compile(model)

        # Reconstruct Stage 1 history from CSV log
        hist_s1 = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
        if s1_log_path.exists():
            s1_df = pd.read_csv(s1_log_path)
            hist_s1["train_loss"] = s1_df["train_loss"].tolist()
            hist_s1["train_acc"] = s1_df["train_acc"].tolist()
            hist_s1["val_loss"] = s1_df["val_loss"].tolist()
            hist_s1["val_acc"] = s1_df["val_acc"].tolist()
            hist_s1["lr"] = s1_df["lr"].tolist()
            print(f"  📄 Loaded Stage 1 history ({len(s1_df)} epochs from CSV)")
        else:
            print("  ⚠️  Stage 1 CSV log not found — training history plot will only show Stage 2")

        # Resume W&B run
        wandb_run = None
        if WANDB_AVAILABLE and not args.no_wandb:
            try:
                if args.wandb_run_id:
                    wandb_run = wandb.init(
                        project="leafy",
                        id=args.wandb_run_id,
                        resume="must",
                    )
                    print(f"  📊 Resumed W&B run: {args.wandb_run_id}")
                else:
                    wandb_run = wandb.init(
                        project="leafy",
                        name=f"EfficientNetV2S_resumed_{int(time.time())}",
                        config={
                            "architecture": "EfficientNet-V2-S",
                            "num_classes": num_classes,
                            "s1_epochs": s1_epochs, "s2_epochs": s2_epochs,
                            "s1_img_size": s1_img_size, "s2_img_size": s2_img_size,
                            "s1_batch": s1_batch, "s2_batch": s2_batch,
                            "s1_best_acc": best_acc_s1,
                            "label_smoothing": label_smoothing,
                            "weight_decay": weight_decay,
                            "dataset_mean": dataset_mean, "dataset_std": dataset_std,
                            "resumed": True,
                        },
                        tags=["efficientnet-v2-s", "plant-pathology", "resumed"],
                    )
                    print(f"  📊 Started new W&B run (no run ID provided for resume)")
            except Exception as e:
                print(f"  W&B init failed: {e}")

    else:
        # ══════════════════════════════════════════════════════════════
        #  STAGE 1: FEATURE EXTRACTION (Frozen Backbone)
        # ══════════════════════════════════════════════════════════════
        print("\n" + "═" * 60)
        print("  📌 STAGE 1: Feature Extraction (backbone frozen)")
        print("═" * 60)

        dataloaders, sizes, train_dataset = create_dataloaders(
            args.data_dir, s1_img_size, s1_batch, dataset_mean, dataset_std, num_workers,
        )
        print(f"  Train: {sizes['train']:,}  Val: {sizes['val']:,}")

        # Augmented samples visualization
        plot_augmented_samples(train_dataset, class_names, dirs["images"] / "augmented_samples_stage1.png")

        # Class weights for loss
        class_weights = compute_ens_class_weights(train_dataset, ens_beta, num_classes, device)

        # Model
        model = build_model(num_classes, device)
        model = try_compile(model)

        # Loss — Focal Loss (better than CE for imbalanced data)
        criterion = FocalLoss(weight=class_weights, gamma=focal_gamma, label_smoothing=label_smoothing)

        # ── LR Finder for Stage 1 ──
        suggested_lr_s1 = run_lr_finder(
            model, dataloaders["train"], criterion, device,
            save_path=dirs["images"] / "lr_finder_stage1.png",
            num_iter=min(200, len(dataloaders["train"])),
        )

        # Optimizer — exclude BN/bias from weight decay
        optimizer = optim.AdamW(
            get_param_groups_no_decay(model, suggested_lr_s1, weight_decay),
        )
        steps_per_epoch_s1 = len(dataloaders["train"]) // accum_steps_s1
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=steps_per_epoch_s1 * 2, T_mult=1, eta_min=suggested_lr_s1 / 100,
        )

        # W&B
        wandb_run = None
        if WANDB_AVAILABLE and not args.no_wandb:
            try:
                wandb_run = wandb.init(
                    project="leafy",
                    name=f"EfficientNetV2S_{int(time.time())}",
                    config={
                        "architecture": "EfficientNet-V2-S",
                        "num_classes": num_classes,
                        "s1_epochs": s1_epochs, "s2_epochs": s2_epochs,
                        "s1_img_size": s1_img_size, "s2_img_size": s2_img_size,
                        "s1_batch": s1_batch, "s2_batch": s2_batch,
                        "s1_lr": suggested_lr_s1,
                        "label_smoothing": label_smoothing,
                        "weight_decay": weight_decay,
                        "dataset_mean": dataset_mean, "dataset_std": dataset_std,
                    },
                    tags=["efficientnet-v2-s", "plant-pathology"],
                )
            except Exception as e:
                print(f"  W&B init failed: {e}")

        # Train Stage 1 (Mixup 50%, per-batch scheduler)
        model, hist_s1, best_acc_s1 = train_stage(
            model, dataloaders, sizes, criterion, optimizer, scheduler,
            device, dirs, "Stage1_FeatureExtraction", s1_epochs, patience, grad_clip,
            accum_steps=accum_steps_s1, use_mixup=True, mixup_prob=0.5,
            step_scheduler_per_batch=True, wandb_run=wandb_run,
        )
        print(f"\n  ✅ Stage 1 done — Best Val Acc: {best_acc_s1:.4f}")

        # Clean up Stage 1 resources before Stage 2
        del dataloaders, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()

    # ══════════════════════════════════════════════════════════════
    #  STAGE 2: FINE-TUNING (All Layers Unfrozen)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  🔓 STAGE 2: Fine-Tuning (all layers unfrozen)")
    print("═" * 60)

    # Clean up before Stage 2 dataloaders
    torch.cuda.empty_cache()
    gc.collect()

    # Higher resolution dataloaders
    dataloaders, sizes, train_dataset = create_dataloaders(
        args.data_dir, s2_img_size, s2_batch, dataset_mean, dataset_std, num_workers,
    )
    print(f"  Image size: {s2_img_size}×{s2_img_size}, Batch: {s2_batch}")

    # Augmented samples at new resolution
    plot_augmented_samples(train_dataset, class_names, dirs["images"] / "augmented_samples_stage2.png")

    # Unfreeze
    unfreeze_backbone(model)

    # New class weights + Focal Loss
    class_weights = compute_ens_class_weights(train_dataset, ens_beta, num_classes, device)
    criterion = FocalLoss(weight=class_weights, gamma=focal_gamma, label_smoothing=label_smoothing)

    # ── LR Finder for Stage 2 (discriminative LR) ──
    suggested_lr_s2 = run_lr_finder(
        model, dataloaders["train"], criterion, device,
        save_path=dirs["images"] / "lr_finder_stage2.png",
        num_iter=min(200, len(dataloaders["train"])),
    )

    # Discriminative LR: backbone at 1/10th
    lr_head = suggested_lr_s2
    lr_backbone = suggested_lr_s2 / 10
    print(f"  Discriminative LR — Head: {lr_head:.2e}, Backbone: {lr_backbone:.2e}")

    # Discriminative LR + no weight decay on BN/bias
    param_groups = get_param_groups_discriminative_no_decay(model, lr_head, lr_backbone, weight_decay)
    optimizer = optim.AdamW(param_groups)

    # Warmup (2 epochs) → CosineAnnealing — prevents catastrophic forgetting
    # when backbone is suddenly unfrozen at a potentially high LR
    warmup_epochs = 2
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=s2_epochs - warmup_epochs, eta_min=1e-7,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    print(f"  Scheduler: LinearWarmup({warmup_epochs}ep) → CosineAnnealing({s2_epochs - warmup_epochs}ep)")

    if wandb_run and WANDB_AVAILABLE:
        wandb.config.update({"s2_lr_head": lr_head, "s2_lr_backbone": lr_backbone})

    # Train Stage 2 (Mixup 80%, per-EPOCH scheduler — critical difference)
    model, hist_s2, best_acc_s2 = train_stage(
        model, dataloaders, sizes, criterion, optimizer, scheduler,
        device, dirs, "Stage2_FineTuning", s2_epochs, patience, grad_clip,
        accum_steps=accum_steps_s2, use_mixup=True, mixup_prob=0.8,
        step_scheduler_per_batch=False, wandb_run=wandb_run,  # per-EPOCH for CosineAnnealingLR
    )
    print(f"\n  ✅ Stage 2 done — Best Val Acc: {best_acc_s2:.4f}")

    # ══════════════════════════════════════════════════════════════
    #  TRAINING PLOTS
    # ══════════════════════════════════════════════════════════════
    plot_training_history(hist_s1, hist_s2, dirs["images"] / "training_history.png")

    # ══════════════════════════════════════════════════════════════
    #  FINAL EVALUATION ON TEST SET
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  🧪 FINAL EVALUATION ON TEST SET")
    print("═" * 60)

    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "test"),
        get_eval_transforms(s2_img_size, dataset_mean, dataset_std),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=s2_batch, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"  Test samples: {len(test_dataset):,}")

    test_labels, test_preds = evaluate_model(model, test_loader, device)

    # Reports & plots
    report_dict = save_classification_report(
        test_labels, test_preds, class_names,
        dirs["logs"] / "classification_report.txt", wandb_run,
    )
    plot_confusion_matrix(
        test_labels, test_preds, class_names,
        dirs["images"] / "confusion_matrix.png",
    )
    plot_per_class_metrics(
        test_labels, test_preds, class_names,
        dirs["images"] / "per_class_metrics.png",
    )

    # ══════════════════════════════════════════════════════════════
    #  EXPORT
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  📦 EXPORTING MODEL")
    print("═" * 60)

    export_model(model, class_names, num_classes, s2_img_size, dirs, device)

    # Save inference config (everything needed to reload)
    inference_config = {
        "architecture": "efficientnet_v2_s",
        "num_classes": num_classes,
        "image_size": s2_img_size,
        "mean": dataset_mean,
        "std": dataset_std,
        "class_names": class_names,
    }
    config_path = dirs["models"] / "inference_config.json"
    with open(config_path, "w") as f:
        json.dump(inference_config, f, indent=2)
    print(f"  💾 Inference config saved: {config_path}")

    # ══════════════════════════════════════════════════════════════
    #  W&B ARTIFACT
    # ══════════════════════════════════════════════════════════════
    if wandb_run and WANDB_AVAILABLE:
        try:
            artifact = wandb.Artifact("EfficientNetV2S-final", type="model",
                                      metadata={"test_accuracy": report_dict.get("accuracy", 0)})
            artifact.add_file(str(dirs["models"] / "best_model.pth"))
            artifact.add_file(str(dirs["models"] / "class_names.json"))
            artifact.add_file(str(dirs["models"] / "inference_config.json"))
            wandb_run.log_artifact(artifact)
            wandb_run.finish()
            print("  W&B run finished & artifact uploaded.")
        except Exception as e:
            print(f"  W&B finish error: {e}")

    # ══════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  🌿 TRAINING COMPLETE!")
    print("═" * 60)
    print(f"  Classes:              {num_classes}")
    print(f"  Stage 1 Best Val Acc: {best_acc_s1:.4f}")
    print(f"  Stage 2 Best Val Acc: {best_acc_s2:.4f}")
    print(f"  Test Accuracy:        {report_dict.get('accuracy', 'N/A')}")
    print(f"\n  📁 All outputs in: EfficientNetV2S/")
    print(f"     models/best_model.pth         — PyTorch weights")
    print(f"     models/model.onnx             — ONNX format")
    print(f"     models/class_names.json       — Class mapping")
    print(f"     models/inference_config.json   — Full inference config")
    print(f"     images/lr_finder_stage1.png    — LR finder (Stage 1)")
    print(f"     images/lr_finder_stage2.png    — LR finder (Stage 2)")
    print(f"     images/training_history.png    — Acc/Loss/LR curves")
    print(f"     images/confusion_matrix.png    — Per-class accuracy")
    print(f"     images/per_class_metrics.png   — P/R/F1 per class")
    print(f"     images/class_distribution.png  — Dataset distribution")
    print(f"     images/augmented_samples_*.png — Augmentation preview")
    print(f"     logs/training_log_*.csv        — Epoch-level logs")
    print(f"     logs/classification_report.txt — Full test report")
    print("═" * 60)


if __name__ == "__main__":
    main()
