"""Tests for checkpoint persistence."""

from pathlib import Path

from cnn_classifier.checkpoint import load_checkpoint, save_checkpoint
from cnn_classifier.model import DocumentCNN


def test_saved_checkpoint_can_be_loaded(tmp_path: Path) -> None:
    model = DocumentCNN(num_classes=3)
    classes = ["driving_license", "others", "social_security"]
    path = tmp_path / "model.pt"

    save_checkpoint(model, classes, path, seed=42)
    loaded_model, loaded_classes, metadata = load_checkpoint(path)

    assert loaded_classes == classes
    assert metadata["seed"] == 42
    assert loaded_model.linear_layers.out_features == len(classes)
