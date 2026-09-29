"""Tests for training configuration and training/evaluation helpers."""

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cnn_classifier.train import TrainingConfig, evaluate, train_one_epoch


class IdentityClassifier(nn.Module):
    """Interpret each two-value input as its class logits."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale


def test_training_config_validates_finite_learning_rate() -> None:
    with pytest.raises(ValueError, match="finite number greater than 0"):
        TrainingConfig(learning_rate=float("nan"))


def test_training_config_validates_validation_fraction() -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        TrainingConfig(validation_fraction=1.0)


def test_train_one_epoch_and_evaluate() -> None:
    values = torch.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 0.0]])
    labels = torch.tensor([0, 1, 0])
    loader = DataLoader(TensorDataset(values, labels), batch_size=2)
    model = IdentityClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss = train_one_epoch(
        model,
        loader,
        optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
    )
    metrics = evaluate(model, loader, torch.device("cpu"), ["first", "second"])

    assert math.isfinite(loss)
    assert metrics.accuracy == 1.0
    assert metrics.confusion_matrix == [[2, 0], [0, 1]]
    assert metrics.per_class[0].precision == 1.0
    assert metrics.per_class[1].recall == 1.0
    assert not model.training


def test_train_and_evaluate_reject_empty_loaders() -> None:
    loader = DataLoader(
        TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
    )
    model = IdentityClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(ValueError, match="empty dataset"):
        train_one_epoch(
            model,
            loader,
            optimizer,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="empty dataset"):
        evaluate(model, loader, torch.device("cpu"), ["first", "second"])
