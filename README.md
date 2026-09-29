# Document CNN Classifier

A PyTorch CNN for classifying document images into driving licenses, social security cards, and other documents. The model accepts RGB images resized to `200 × 200` and uses an `ImageFolder` dataset.

## Dataset layout

Use matching class directories for both splits:

```text
Data/
├── Training_data/
│   ├── driving_license/
│   ├── others/
│   └── social_security/
└── Testing_data/
    ├── driving_license/
    ├── others/
    └── social_security/
```

Document images may contain sensitive information. Keep datasets outside Git; repository ignore rules exclude contents of `data/` while retaining its usage note.

## Setup

Install the locked dependencies from the repository root with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Train

For a dataset on `D:` in Windows PowerShell:

```powershell
uv run document-cnn-train `
  "D:\Program Files\Datasets\CNN\Data\Training_data" `
  "D:\Program Files\Datasets\CNN\Data\Testing_data" `
  --model artifacts\model.pt --epochs 10 --batch-size 8 --seed 42
```

The equivalent module invocation is:

```bash
uv run python -m cnn_classifier.train <train_dir> <test_dir>
```

The loader converts images to RGB, resizes them to 200×200, and keeps the notebook's original 0–255 pixel scale. The notebook's malformed HWC-to-CHW reshape is not retained; the package uses correct channel-first RGB tensors. The default checkpoint path is `artifacts/model.pt`. Checkpoints store model weights, ordered class names, and the random seed. Generated model files in `artifacts/` are excluded from Git.

## Load a checkpoint

```python
from cnn_classifier import load_checkpoint

model, classes, metadata = load_checkpoint("artifacts/model.pt")
```

The model is returned in evaluation mode. Pass preprocessed `(batch, 3, 200, 200)` tensors to obtain class scores.

## Tests

Run the test suite with pytest:

```bash
uv run pytest
```
