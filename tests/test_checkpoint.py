"""Tests for checkpoint persistence."""

from pathlib import Path

from cnn_classifier.checkpoint import load_checkpoint, save_checkpoint
from cnn_classifier.model import DocumentCNN, FeatureCNN


def test_saved_checkpoint_can_be_loaded(tmp_path: Path) -> None:
    model = DocumentCNN(num_classes=3)
    classes = ["driving_license", "others", "social_security"]
    path = tmp_path / "model.pt"

    original_state = {
        name: tensor.clone() for name, tensor in model.state_dict().items()
    }

    save_checkpoint(model, classes, path, seed=42)
    loaded_model, loaded_classes, metadata = load_checkpoint(path)

    assert loaded_classes == classes
    assert metadata["seed"] == 42
    assert metadata["checkpoint_version"] == 1
    assert metadata["architecture"] == "baseline"
    assert metadata["pixel_scale"] == "byte"
    assert loaded_model.linear_layers.out_features == len(classes)
    assert model.training
    for name, tensor in model.state_dict().items():
        assert tensor.equal(original_state[name])


def test_feature_architecture_metadata_survives_checkpoint_reload(
    tmp_path: Path,
) -> None:
    model = FeatureCNN(num_classes=3)
    classes = ["driving_license", "others", "social_security"]
    path = tmp_path / "feature.pt"

    save_checkpoint(
        model,
        classes,
        path,
        seed=7,
        architecture="feature",
        pixel_scale="unit",
        optimizer="adam",
    )
    loaded_model, loaded_classes, metadata = load_checkpoint(path)

    assert isinstance(loaded_model, FeatureCNN)
    assert loaded_classes == classes
    assert metadata["architecture"] == "feature"
    assert metadata["pixel_scale"] == "unit"
    assert metadata["optimizer"] == "adam"
