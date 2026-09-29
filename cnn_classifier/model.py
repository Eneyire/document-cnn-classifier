"""CNN architecture used by the document classification workflow."""

from torch import Tensor, nn

from .data import IMAGE_SIZE


class DocumentCNN(nn.Module):
    """Classify RGB document images resized to 200 by 200 pixels."""

    def __init__(self, num_classes: int = 3) -> None:
        """Initialize the network.

        Args:
            num_classes: Number of output document categories.
        """
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        super().__init__()
        # Kept consistent with the original notebook architecture.
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 3, kernel_size=50, stride=1),
            nn.MaxPool2d(kernel_size=1, stride=1),
        )
        self.linear_layers = nn.Linear(3, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """Return unnormalized class scores for a batch of RGB images."""
        expected_shape = (3, *IMAGE_SIZE)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected_shape:
            raise ValueError(
                "Expected images with shape (batch, 3, 200, 200); "
                f"received {tuple(images.shape)}"
            )

        features = self.cnn_layers(images)
        return self.linear_layers(features.flatten(start_dim=1))
