"""Checkpoint persistence for trained document classifiers."""

from pathlib import Path
from typing import Any

import torch

from .model import DocumentCNN


def save_checkpoint(
    model: DocumentCNN,
    classes: list[str],
    destination: str | Path,
    *,
    seed: int,
) -> Path:
    """Save model weights and metadata, returning the destination path."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "classes": classes,
            "seed": seed,
        },
        path,
    )
    return path


def load_checkpoint(
    source: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[DocumentCNN, list[str], dict[str, Any]]:
    """Load a saved classifier and its metadata onto ``device``."""
    checkpoint = torch.load(source, map_location=device, weights_only=True)
    classes = checkpoint.get("classes")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(classes, list) or not all(
        isinstance(name, str) for name in classes
    ):
        raise TypeError("Checkpoint is missing a valid 'classes' list")
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint is missing model weights")

    model = DocumentCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, classes, checkpoint
