"""Checkpoint persistence for trained document classifiers."""

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .data import PixelScale
from .model import ModelArchitecture, create_model

CHECKPOINT_VERSION = 1


def save_checkpoint(
    model: nn.Module,
    classes: list[str],
    destination: str | Path,
    *,
    seed: int,
    architecture: ModelArchitecture = "baseline",
    pixel_scale: PixelScale = "byte",
    optimizer: str = "sgd",
    validation_accuracy: float | None = None,
) -> Path:
    """Atomically save model weights, labels, and training metadata."""
    if not all(isinstance(name, str) and name for name in classes):
        raise ValueError("classes must contain non-empty strings")
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("classes must contain at least two unique class names")
    if pixel_scale not in ("byte", "unit"):
        raise ValueError("pixel_scale must be 'byte' or 'unit'")
    if optimizer not in ("sgd", "adam"):
        raise ValueError("optimizer must be 'sgd' or 'adam'")

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "state_dict": state_dict,
        "classes": list(classes),
        "seed": seed,
        "architecture": architecture,
        "pixel_scale": pixel_scale,
        "optimizer": optimizer,
        "validation_accuracy": validation_accuracy,
    }

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return path


def load_checkpoint(
    source: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, list[str], dict[str, Any]]:
    """Load a saved classifier and its metadata onto ``device``."""
    checkpoint = torch.load(source, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a dictionary")
    checkpoint.setdefault("architecture", "baseline")
    checkpoint.setdefault("pixel_scale", "byte")
    checkpoint.setdefault("optimizer", "sgd")

    classes = checkpoint.get("classes")
    state_dict = checkpoint.get("state_dict")
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or not all(isinstance(name, str) and name for name in classes)
        or len(set(classes)) != len(classes)
    ):
        raise TypeError("Checkpoint is missing a valid 'classes' list")
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint is missing model weights")

    architecture = checkpoint.get("architecture", "baseline")
    if architecture not in ("baseline", "feature"):
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    model = create_model(architecture, num_classes=len(classes)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, classes, checkpoint
