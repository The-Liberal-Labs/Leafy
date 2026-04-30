import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from model_training.train_efficientnet import (
    FocalLoss,
    build_criterion,
    compute_ens_class_weights,
    create_weighted_sampler,
    cutmix_data,
    mixup_criterion,
    mixup_data,
    soft_mix_accuracy,
)


@pytest.fixture
def simple_dataset(tmp_path):
    """Create a tiny ImageFolder with 3 classes for testing."""
    for cls_idx in range(3):
        cls_dir = tmp_path / str(cls_idx)
        cls_dir.mkdir()
        for i in range(10 + cls_idx * 5):
            img = transforms.ToPILImage()(torch.rand(3, 32, 32))
            img.save(cls_dir / f"{i:03d}.jpg")
    return datasets.ImageFolder(str(tmp_path))


class TestFocalLoss:
    def test_output_shape_reduction_mean(self):
        loss_fn = FocalLoss(gamma=2.0, reduction="mean")
        inputs = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        output = loss_fn(inputs, targets)
        assert output.ndim == 0

    def test_output_shape_reduction_sum(self):
        loss_fn = FocalLoss(gamma=2.0, reduction="sum")
        inputs = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        output = loss_fn(inputs, targets)
        assert output.ndim == 0

    def test_output_shape_reduction_none(self):
        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        inputs = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        output = loss_fn(inputs, targets)
        assert output.shape == (8,)

    def test_focal_downweights_easy(self):
        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        targets = torch.tensor([0])
        loss_hard = loss_fn(torch.tensor([[0.0, 10.0]]), targets).item()
        loss_easy = loss_fn(torch.tensor([[10.0, 0.0]]), targets).item()
        assert loss_hard > loss_easy

    def test_label_smoothing(self):
        loss_fn = FocalLoss(gamma=0.0, label_smoothing=0.1, reduction="mean")
        inputs = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        loss = loss_fn(inputs, targets)
        assert loss.item() > 0

    def test_with_class_weights(self):
        weight = torch.tensor([0.5, 2.0, 1.0])
        loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.0)
        loss_fn.register_buffer("weight", weight)
        inputs = torch.randn(8, 3)
        targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        output = loss_fn(inputs, targets)
        assert output.ndim == 0


class TestBuildCriterion:
    def test_none_strategy_returns_ce(self):
        criterion = build_criterion("none", None, 2.0, 0.0)
        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_ens_loss_returns_ce_with_weights(self):
        weights = torch.tensor([1.0, 2.0, 3.0])
        criterion = build_criterion("ens_loss", weights, 2.0, 0.0)
        assert isinstance(criterion, nn.CrossEntropyLoss)
        assert criterion.weight is not None

    def test_focal_returns_focal(self):
        criterion = build_criterion("focal", None, 1.5, 0.05)
        assert isinstance(criterion, FocalLoss)
        assert criterion.gamma == 1.5
        assert criterion.label_smoothing == 0.05

    def test_sampler_ens_focal(self):
        weights = torch.tensor([0.5, 2.0, 1.0])
        criterion = build_criterion("sampler_ens_focal", weights, 1.0, 0.1)
        assert isinstance(criterion, FocalLoss)

    def test_sampler_only_returns_ce_no_weights(self):
        criterion = build_criterion("sampler", None, 2.0, 0.0)
        assert isinstance(criterion, nn.CrossEntropyLoss)
        assert criterion.weight is None

    @pytest.mark.parametrize(
        "strategy",
        ["none", "ens_loss", "focal", "sampler", "sampler_ens", "sampler_focal", "sampler_ens_focal"],
    )
    def test_all_strategies_build(self, strategy):
        weights = torch.tensor([1.0, 2.0, 3.0]) if "ens" in strategy else None
        criterion = build_criterion(strategy, weights, 1.5, 0.05)
        assert criterion is not None


class TestMixup:
    def test_mixup_output_shapes(self):
        x = torch.randn(8, 3, 32, 32)
        y = torch.randint(0, 10, (8,))
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
        assert mixed_x.shape == (8, 3, 32, 32)
        assert y_a.shape == (8,)
        assert y_b.shape == (8,)
        assert 0.0 <= lam <= 1.0

    def test_mixup_alpha_zero(self):
        x = torch.randn(8, 3, 32, 32)
        y = torch.randint(0, 10, (8,))
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.0)
        assert lam == 1.0

    def test_mixup_criterion_scalar(self):
        pred = torch.randn(8, 10)
        y_a = torch.randint(0, 10, (8,))
        y_b = torch.randint(0, 10, (8,))
        criterion = nn.CrossEntropyLoss()
        loss = mixup_criterion(criterion, pred, y_a, y_b, 0.6)
        assert loss.ndim == 0

    def test_soft_mix_accuracy_range(self):
        preds = torch.tensor([0, 1, 2])
        targets_a = torch.tensor([0, 1, 2])
        targets_b = torch.tensor([2, 0, 1])
        acc = soft_mix_accuracy(preds, targets_a, targets_b, 0.5)
        assert 0.0 <= acc <= 3.0


class TestCutMix:
    def test_cutmix_output_shapes(self):
        x = torch.randn(8, 3, 64, 64)
        y = torch.randint(0, 10, (8,))
        mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=1.0)
        assert mixed_x.shape == (8, 3, 64, 64)
        assert y_a.shape == (8,)
        assert y_b.shape == (8,)
        assert 0.0 <= lam <= 1.0

    def test_cutmix_modifies_tensor(self):
        x = torch.randn(4, 3, 64, 64)
        x_before = x.clone()
        y = torch.tensor([0, 1, 2, 3])
        mixed_x, _, _, _ = cutmix_data(x, y, alpha=1.0)
        diff_mask = (mixed_x != x_before).any(dim=(1, 2, 3))
        assert diff_mask.sum() > 0

    def test_cutmix_alpha_zero(self):
        x = torch.randn(4, 3, 64, 64)
        y = torch.randint(0, 10, (4,))
        _, _, _, lam = cutmix_data(x, y, alpha=0.0)
        assert lam == 1.0


class TestENSWeights:
    def test_weights_shape(self, simple_dataset):
        weights = compute_ens_class_weights(simple_dataset, beta=0.999, num_classes=3, device="cpu")
        assert weights.shape == (3,)

    def test_weights_sum_to_num_classes(self, simple_dataset):
        weights = compute_ens_class_weights(simple_dataset, beta=0.999, num_classes=3, device="cpu")
        assert torch.isclose(weights.sum(), torch.tensor(3.0), atol=1e-4)

    def test_minority_gets_higher_weight(self, simple_dataset):
        weights = compute_ens_class_weights(simple_dataset, beta=0.999, num_classes=3, device="cpu")
        counts = [10, 15, 20]
        max_idx = counts.index(max(counts))
        min_idx = counts.index(min(counts))
        assert weights[min_idx] > weights[max_idx]


class TestWeightedSampler:
    def test_returns_correct_types(self, simple_dataset):
        sampler, expected = create_weighted_sampler(simple_dataset, max_multiplier=3.0)
        assert isinstance(sampler, WeightedRandomSampler)
        assert isinstance(expected, np.ndarray)

    def test_expected_counts_shape(self, simple_dataset):
        _, expected = create_weighted_sampler(simple_dataset, max_multiplier=3.0)
        assert len(expected) == 3

    def test_sampler_num_samples(self, simple_dataset):
        sampler, _ = create_weighted_sampler(simple_dataset, max_multiplier=3.0)
        assert sampler.num_samples == len(simple_dataset)

    def test_max_multiplier_one_no_upsampling(self, simple_dataset):
        _, expected = create_weighted_sampler(simple_dataset, max_multiplier=1.0)
        natural = np.bincount(
            [s[1] for s in simple_dataset.samples]
        ).astype(np.float64)
        assert np.allclose(expected, natural, atol=1e-4)
