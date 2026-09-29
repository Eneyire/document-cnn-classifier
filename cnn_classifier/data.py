"""Dataset construction, preprocessing, and data-loader helpers."""

from collections import defaultdict
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGE_SIZE = (200, 200)
PixelScale = Literal["byte", "unit"]


class _ScaleToByteRange:
    """Scale image tensors from 0–1 to 0–255."""

    def __call__(self, image: Tensor) -> Tensor:
        return image * 255.0


def create_transform(pixel_scale: PixelScale = "byte") -> transforms.Compose:
    """Create preprocessing for RGB images at the requested numeric scale."""
    if pixel_scale not in ("byte", "unit"):
        raise ValueError("pixel_scale must be 'byte' or 'unit'")

    pipeline = [transforms.Resize(IMAGE_SIZE), transforms.ToTensor()]
    if pixel_scale == "byte":
        pipeline.append(_ScaleToByteRange())
    return transforms.Compose(pipeline)


def _load_image_folder(
    directory: str | Path,
    *,
    pixel_scale: PixelScale,
) -> datasets.ImageFolder:
    path = Path(directory).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {path}")
    if not any(child.is_dir() for child in path.iterdir()):
        raise ValueError(f"No class directories found in {path}")

    dataset = datasets.ImageFolder(
        str(path),
        transform=create_transform(pixel_scale),
        allow_empty=True,
    )
    if not dataset.samples:
        raise ValueError(f"No supported image files found in {path}")

    samples_per_class = [0] * len(dataset.classes)
    for _, class_index in dataset.samples:
        samples_per_class[class_index] += 1
    empty_classes = [
        name for name, count in zip(dataset.classes, samples_per_class) if count == 0
    ]
    if empty_classes:
        raise ValueError(
            f"No supported image files found for class(es): {', '.join(empty_classes)}"
        )
    return dataset


def make_dataloader(
    directory: str | Path,
    *,
    batch_size: int = 8,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
    pixel_scale: PixelScale = "byte",
) -> DataLoader:
    """Load an image-folder dataset into a data loader."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    dataset = _load_image_folder(directory, pixel_scale=pixel_scale)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def make_training_loaders(
    directory: str | Path,
    *,
    batch_size: int = 8,
    validation_fraction: float = 0.2,
    seed: int = 42,
    pixel_scale: PixelScale = "byte",
) -> tuple[DataLoader, DataLoader]:
    """Create seeded, per-class stratified training and validation loaders."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    dataset = _load_image_folder(directory, pixel_scale=pixel_scale)
    indices_by_class: dict[int, list[int]] = defaultdict(list)
    for index, class_index in enumerate(dataset.targets):
        indices_by_class[class_index].append(index)

    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_name, class_index in dataset.class_to_idx.items():
        class_indices = indices_by_class[class_index]
        if len(class_indices) < 2:
            raise ValueError(
                f"Class '{class_name}' needs at least two images for a "
                "training/validation split"
            )
        order = torch.randperm(len(class_indices), generator=generator).tolist()
        validation_count = min(
            len(class_indices) - 1,
            max(1, round(len(class_indices) * validation_fraction)),
        )
        validation_indices.extend(
            class_indices[index] for index in order[:validation_count]
        )
        train_indices.extend(class_indices[index] for index in order[validation_count:])

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, validation_loader
