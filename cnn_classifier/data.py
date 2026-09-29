"""Dataset and data-loader helpers for folder-organized document images."""

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = (200, 200)


class _ScaleToOriginalRange:
    """Undo ``ToTensor`` scaling to match the source notebook's 0–255 range."""

    def __call__(self, image: Tensor) -> Tensor:
        return image * 255.0


def create_transform() -> transforms.Compose:
    """Create the RGB preprocessing pipeline expected by :class:`DocumentCNN`."""
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            _ScaleToOriginalRange(),
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

    Each immediate subdirectory is a class. Images are decoded as RGB,
    resized to ``IMAGE_SIZE``, and represented as float tensors in the
    source workflow's 0–255 pixel range.

    Raises:
        FileNotFoundError: If ``directory`` does not exist.
        NotADirectoryError: If ``directory`` is not a directory.
        ValueError: If the batch size is invalid or a class has no images.
    """
    path = Path(directory).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {path}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if not any(child.is_dir() for child in path.iterdir()):
        raise ValueError(f"No class directories found in {path}")

    dataset = datasets.ImageFolder(
        str(path),
        transform=create_transform(),
        allow_empty=True,
    )
    if not dataset.samples:
        raise ValueError(f"No supported image files found in {path}")

    samples_per_class = [0] * len(dataset.classes)
    for _, class_index in dataset.samples:
        samples_per_class[class_index] += 1
    empty_classes = [
        name
        for name, sample_count in zip(dataset.classes, samples_per_class)
        if sample_count == 0
    ]
    if empty_classes:
        raise ValueError(
            f"No supported image files found for class(es): {', '.join(empty_classes)}"
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
