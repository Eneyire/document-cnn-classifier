"""Train and evaluate the document-image CNN."""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

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
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python and PyTorch random sources and return a loader generator."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def train_one_epoch(
    model: DocumentCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train ``model`` for one epoch and return mean batch loss."""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model: DocumentCNN, loader: DataLoader, device: torch.device) -> float:
    """Evaluate ``model`` and return classification accuracy."""
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1)
            correct += (predictions.cpu() == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


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

    if train_loader.dataset.classes != test_loader.dataset.classes:
        raise ValueError("Training and testing directories must have identical classes")

    model = DocumentCNN(num_classes=len(train_loader.dataset.classes)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"Using device: {device}")
    for epoch in range(config.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{config.epochs} - loss: {loss:.4f}")

    accuracy = evaluate(model, test_loader, device)
    destination = save_checkpoint(
        model,
        train_loader.dataset.classes,
        model_path,
        seed=config.seed,
    )
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Saved model to {destination}")
    return accuracy


def _colab_dataset_paths() -> tuple[Path, Path]:
    """Mount Drive and return the original notebook's train/test directories."""
    try:
        from google.colab import drive
    except ImportError as error:
        raise RuntimeError(
            "train_dir and test_dir are required outside Google Colab"
        ) from error

    drive.mount("/content/gdrive")
    dataset_root = Path("/content/gdrive/My Drive/Datasets/CNN/Data")
    return dataset_root / "Training_data", dataset_root / "Testing_data"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "train_dir", type=Path, nargs="?", help="Training ImageFolder directory"
    )
    parser.add_argument(
        "test_dir", type=Path, nargs="?", help="Testing ImageFolder directory"
    )
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run training from the command line."""
    args = _parse_args()
    if (args.train_dir is None) != (args.test_dir is None):
        raise SystemExit(
            "Provide both train_dir and test_dir, or neither in Google Colab."
        )
    if args.train_dir is None:
        args.train_dir, args.test_dir = _colab_dataset_paths()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    train_model(args.train_dir, args.test_dir, args.model, config=config)


if __name__ == "__main__":
    main()
