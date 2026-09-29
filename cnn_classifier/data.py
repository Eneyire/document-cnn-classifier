"""Dataset and data-loader helpers for folder-organized document images."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = (200, 200)


def create_transform() -> transforms.Compose:
    """Create the preprocessing pipeline expected by :class:`DocumentCNN`."""
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda image: image * 255.0),
        ]
    )


def make_dataloader(
    directory: str | Path,
    *,
    batch_size: int = 8,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Load an ``ImageFolder`` directory into a data loader.

    Args:
        directory: Directory containing one subdirectory for each class.
        batch_size: Number of images per batch.
        shuffle: Whether to randomize sample order.
        generator: Optional random generator used when shuffling.

    Raises:
        FileNotFoundError: If ``directory`` does not exist.
        NotADirectoryError: If ``directory`` is not a directory.
        ValueError: If ``batch_size`` is not positive or no images are found.
    """
    path = Path(directory).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {path}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    class_directories = [child for child in path.iterdir() if child.is_dir()]
    if not class_directories:
        raise ValueError(f"No class directories found in {path}")

    dataset = datasets.ImageFolder(
        str(path),
        transform=create_transform(),
        allow_empty=True,
    )
    if not dataset.samples:
        raise ValueError(f"No supported image files found in {path}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
