"""Train and evaluate the document-image CNN."""

import argparse
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .checkpoint import save_checkpoint
from .data import make_dataloader
from .model import DocumentCNN


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and runtime options for a training run."""

    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a finite number greater than 0")


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python and PyTorch sources and return a loader generator."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)


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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate a model and return sample-level classification accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise ValueError("Cannot evaluate on an empty dataset")
    return correct / total


def train_model(
    train_dir: str | Path,
    test_dir: str | Path,
    model_path: str | Path,
    *,
    config: TrainingConfig | None = None,
) -> float:
    """Train, evaluate, and save a classifier; return its test accuracy."""
    config = config or TrainingConfig()
    generator = seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_dataloader(
        train_dir,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = make_dataloader(test_dir, batch_size=config.batch_size)

    train_dataset = cast(ImageFolder, train_loader.dataset)
    test_dataset = cast(ImageFolder, test_loader.dataset)
    train_classes = train_dataset.classes
    test_classes = test_dataset.classes
    if train_classes != test_classes:
        raise ValueError(
            "Training and testing directories must have identical classes "
            "in the same order"
        )
    if len(train_classes) < 2:
        raise ValueError("Training data must contain at least two classes")

    model = DocumentCNN(num_classes=len(train_classes)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"Using device: {device}")
    for epoch in range(config.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{config.epochs} - loss: {loss:.4f}")

    accuracy = evaluate(model, test_loader, device)
    destination = save_checkpoint(
        model,
        train_classes,
        model_path,
        seed=config.seed,
    )
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Saved model to {destination}")
    return accuracy


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_dir", type=Path, help="Training ImageFolder directory")
    parser.add_argument("test_dir", type=Path, help="Testing ImageFolder directory")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/model.pt"),
        help="Checkpoint output path (default: %(default)s)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run training from the command line."""
    args = _parse_args(argv)
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    train_model(args.train_dir, args.test_dir, args.model, config=config)


if __name__ == "__main__":
    main()
