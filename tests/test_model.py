"""Tests for the document CNN architecture."""

import pytest
import torch

from cnn_classifier.model import DocumentCNN, FeatureCNN, create_model


def test_forward_returns_one_logit_per_class() -> None:
    model = DocumentCNN(num_classes=3)

    result = model(torch.randn(2, 3, 200, 200))

    assert result.shape == (2, 3)
    assert torch.isfinite(result).all()


def test_forward_rejects_an_unexpected_image_shape() -> None:
    model = DocumentCNN(num_classes=3)

    with pytest.raises(ValueError, match="Expected images"):
        model(torch.randn(1, 3, 100, 100))


def test_model_requires_at_least_two_classes() -> None:
    with pytest.raises(ValueError, match="num_classes must be at least 2"):
        DocumentCNN(num_classes=1)


def test_feature_model_returns_one_logit_per_class() -> None:
    model = create_model("feature", num_classes=3)

    assert isinstance(model, FeatureCNN)
    assert model(torch.randn(2, 3, 200, 200)).shape == (2, 3)
