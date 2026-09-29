"""Train classifiers and report held-out document classification metrics."""

import argparse
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from .checkpoint import save_checkpoint
from .data import PixelScale, make_dataloader, make_training_loaders
from .model import ModelArchitecture, create_model

OptimizerName = Literal["sgd", "adam"]


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and evaluation options for a training run."""

    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-2
    seed: int = 42
    validation_fraction: float = 0.2
    architecture: ModelArchitecture = "baseline"
    pixel_scale: PixelScale = "unit"
    optimizer: OptimizerName = "sgd"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a finite number greater than 0")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.architecture not in ("baseline", "feature"):
            raise ValueError("architecture must be 'baseline' or 'feature'")
        if self.pixel_scale not in ("byte", "unit"):
            raise ValueError("pixel_scale must be 'byte' or 'unit'")
        if self.optimizer not in ("sgd", "adam"):
            raise ValueError("optimizer must be 'sgd' or 'adam'")


@dataclass(frozen=True)
class ClassMetrics:
    """Classification metrics for one class label."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate and per-class metrics from one evaluation split."""

    accuracy: float
    confusion_matrix: list[list[int]]
    per_class: list[ClassMetrics]


@dataclass(frozen=True)
class TrainingResult:
    """Validation selection results and optional final test metrics."""

    best_validation_accuracy: float
    test_metrics: EvaluationMetrics | None


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch random sources for a training run."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train one epoch and return loss averaged over all samples."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot train on an empty dataset")
    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: list[str],
) -> EvaluationMetrics:
    """Calculate accuracy, confusion matrix, and per-class metrics."""
    model.eval()
    confusion = torch.zeros((len(classes), len(classes)), dtype=torch.int64)

    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            labels = labels.cpu()
            encoded = labels * len(classes) + predictions
            confusion += torch.bincount(encoded, minlength=len(classes) ** 2).reshape(
                len(classes), len(classes)
            )

    total = int(confusion.sum().item())
    if total == 0:
        raise ValueError("Cannot evaluate on an empty dataset")

    per_class: list[ClassMetrics] = []
    for index, label in enumerate(classes):
        true_positive = int(confusion[index, index].item())
        predicted_count = int(confusion[:, index].sum().item())
        support = int(confusion[index, :].sum().item())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class.append(ClassMetrics(label, precision, recall, f1, support))

    accuracy = float(confusion.diagonal().sum().item()) / total
    return EvaluationMetrics(
        accuracy=accuracy,
        confusion_matrix=confusion.tolist(),
        per_class=per_class,
    )


def _print_metrics(metrics: EvaluationMetrics, classes: list[str]) -> None:
    print(f"Test accuracy: {metrics.accuracy:.2%}")
    print("Per-class metrics:")
    for result in metrics.per_class:
        print(
            f"  {result.label}: precision={result.precision:.3f}, "
            f"recall={result.recall:.3f}, f1={result.f1:.3f}, "
            f"support={result.support}"
        )
    print("Confusion matrix (rows=true, columns=predicted):")
    print("  labels: " + ", ".join(classes))
    for label, row in zip(classes, metrics.confusion_matrix):
        print(f"  {label}: {row}")


def train_model(
    train_dir: str | Path,
    test_dir: str | Path | None,
    model_path: str | Path,
    *,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Select by validation accuracy and optionally evaluate on test data."""
    config = config or TrainingConfig()
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, validation_loader = make_training_loaders(
        train_dir,
        batch_size=config.batch_size,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
        pixel_scale=config.pixel_scale,
    )
    test_loader = (
        make_dataloader(
            test_dir,
            batch_size=config.batch_size,
            pixel_scale=config.pixel_scale,
        )
        if test_dir is not None
        else None
    )

    train_subset = cast(Subset, train_loader.dataset)
    train_dataset = cast(ImageFolder, train_subset.dataset)
    classes = train_dataset.classes
    test_sample_count: int | str = "not evaluated"
    if test_loader is not None:
        test_dataset = cast(ImageFolder, test_loader.dataset)
        if classes != test_dataset.classes:
            raise ValueError(
                "Training and testing directories must have identical classes "
                "in the same order"
            )
        test_sample_count = len(test_dataset.samples)
    if len(classes) < 2:
        raise ValueError("Training data must contain at least two classes")

    model = create_model(config.architecture, num_classes=len(classes)).to(device)
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    validation_subset = cast(Subset, validation_loader.dataset)

    print(f"Using device: {device}")
    print(
        f"Training architecture={config.architecture}, "
        f"pixel_scale={config.pixel_scale}, optimizer={config.optimizer}, "
        f"train={len(train_subset.indices)}, "
        f"validation={len(validation_subset.indices)}, "
        f"test={test_sample_count}"
    )
    for epoch in range(config.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        validation_metrics = evaluate(model, validation_loader, device, classes)
        print(
            f"Epoch {epoch + 1}/{config.epochs} - loss: {loss:.4f} - "
            f"validation accuracy: {validation_metrics.accuracy:.2%}"
        )
        if validation_metrics.accuracy > best_accuracy:
            best_accuracy = validation_metrics.accuracy
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)

    test_metrics = (
        evaluate(model, test_loader, device, classes)
        if test_loader is not None
        else None
    )
    destination = save_checkpoint(
        model,
        classes,
        model_path,
        seed=config.seed,
        architecture=config.architecture,
        pixel_scale=config.pixel_scale,
        optimizer=config.optimizer,
        validation_accuracy=best_accuracy,
    )
    if test_metrics is not None:
        _print_metrics(test_metrics, classes)
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print(f"Saved model to {destination}")
    return TrainingResult(best_accuracy, test_metrics)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_dir", type=Path, help="Training ImageFolder directory")
    parser.add_argument(
        "test_dir",
        type=Path,
        nargs="?",
        help="Optional held-out test directory; omit for validation-only experiments",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/model.pt"),
        help="Checkpoint output path (default: %(default)s)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--architecture", choices=("baseline", "feature"), default="baseline"
    )
    parser.add_argument("--pixel-scale", choices=("byte", "unit"), default="unit")
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run training from the command line."""
    args = _parse_args(argv)
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        architecture=args.architecture,
        pixel_scale=args.pixel_scale,
        optimizer=args.optimizer,
    )
    train_model(args.train_dir, args.test_dir, args.model, config=config)


if __name__ == "__main__":
    main()
