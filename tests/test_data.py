"""Tests for dataset validation and preprocessing."""

from pathlib import Path
from typing import cast

import pytest
from PIL import Image
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder

from cnn_classifier.data import make_dataloader, make_training_loaders


def test_missing_directory_has_actionable_error() -> None:
    with pytest.raises(FileNotFoundError, match="Dataset directory does not exist"):
        make_dataloader("does-not-exist")


def test_empty_directory_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No class directories"):
        make_dataloader(tmp_path)


def test_class_directory_without_images_has_actionable_error(tmp_path: Path) -> None:
    (tmp_path / "empty_class").mkdir()

    with pytest.raises(ValueError, match="No supported image files"):
        make_dataloader(tmp_path)


def test_batch_size_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        make_dataloader(tmp_path, batch_size=0)


def test_rejects_class_without_supported_images(tmp_path: Path) -> None:
    (tmp_path / "empty_class").mkdir()
    populated_class = tmp_path / "populated_class"
    populated_class.mkdir()
    Image.new("RGB", (8, 8)).save(populated_class / "sample.png")

    with pytest.raises(ValueError, match="empty_class"):
        make_dataloader(tmp_path)


def test_preprocessing_preserves_rgb_channel_order_and_0_to_255_scale(
    tmp_path: Path,
) -> None:
    class_dir = tmp_path / "driving_license"
    class_dir.mkdir()
    Image.new("RGB", (8, 8), color=(30, 120, 240)).save(class_dir / "sample.png")

    images, labels = next(iter(make_dataloader(tmp_path)))

    assert images.shape == (1, 3, 200, 200)
    assert images[0, :, 100, 100].tolist() == pytest.approx([30, 120, 240])
    assert labels.tolist() == [0]


def test_unit_pixel_scale_is_available(tmp_path: Path) -> None:
    class_dir = tmp_path / "driving_license"
    class_dir.mkdir()
    Image.new("RGB", (8, 8), color=(30, 120, 240)).save(class_dir / "sample.png")

    images, _ = next(iter(make_dataloader(tmp_path, pixel_scale="unit")))

    assert images[0, :, 100, 100].tolist() == pytest.approx(
        [30 / 255, 120 / 255, 240 / 255]
    )


def test_training_validation_split_is_stratified_and_reproducible(
    tmp_path: Path,
) -> None:
    for class_name, color in (("alpha", 40), ("beta", 100), ("gamma", 180)):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index in range(10):
            Image.new("RGB", (8, 8), color=(color, color, color)).save(
                class_dir / f"{index}.png"
            )

    train_loader, validation_loader = make_training_loaders(
        tmp_path, batch_size=4, validation_fraction=0.2, seed=17
    )
    repeated_train, repeated_validation = make_training_loaders(
        tmp_path, batch_size=4, validation_fraction=0.2, seed=17
    )
    train_subset = cast(Subset, train_loader.dataset)
    validation_subset = cast(Subset, validation_loader.dataset)
    repeated_train_subset = cast(Subset, repeated_train.dataset)
    repeated_validation_subset = cast(Subset, repeated_validation.dataset)
    train_indices = train_subset.indices
    validation_indices = validation_subset.indices

    assert set(train_indices).isdisjoint(validation_indices)
    assert train_indices == repeated_train_subset.indices
    assert validation_indices == repeated_validation_subset.indices
    dataset = cast(ImageFolder, train_subset.dataset)
    for class_index in range(len(dataset.classes)):
        assert (
            sum(dataset.targets[index] == class_index for index in train_indices) == 8
        )
        assert (
            sum(dataset.targets[index] == class_index for index in validation_indices)
            == 2
        )
