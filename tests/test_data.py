"""Tests for dataset validation and preprocessing."""

from pathlib import Path

import pytest
from PIL import Image

from cnn_classifier.data import make_dataloader


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
