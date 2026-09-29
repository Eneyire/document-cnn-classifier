"""CNN architectures for document-image classification."""

from typing import Literal

from torch import Tensor, nn

from .data import IMAGE_SIZE

ModelArchitecture = Literal["baseline", "feature"]


class DocumentCNN(nn.Module):
    """Baseline CNN retained for compatibility with existing checkpoints."""

    def __init__(self, num_classes: int = 3) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 3, kernel_size=50, stride=1),
            nn.MaxPool2d(kernel_size=1, stride=1),
        )
        self.linear_layers = nn.Linear(3, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        _validate_images(images)
        features = self.cnn_layers(images)
        return self.linear_layers(features.flatten(start_dim=1))


class FeatureCNN(nn.Module):
    """Convolutional feature extractor with global pooling and a linear head."""

    def __init__(self, num_classes: int = 3) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        _validate_images(images)
        features = self.features(images).flatten(start_dim=1)
        return self.classifier(features)


def create_model(
    architecture: ModelArchitecture,
    *,
    num_classes: int,
) -> nn.Module:
    """Construct a classifier by architecture name."""
    if architecture == "baseline":
        return DocumentCNN(num_classes=num_classes)
    if architecture == "feature":
        return FeatureCNN(num_classes=num_classes)
    raise ValueError(f"Unsupported architecture: {architecture}")


def _validate_images(images: Tensor) -> None:
    expected_shape = (3, *IMAGE_SIZE)
    if images.ndim != 4 or tuple(images.shape[1:]) != expected_shape:
        raise ValueError(
            "Expected images with shape (batch, 3, 200, 200); "
            f"received {tuple(images.shape)}"
        )
