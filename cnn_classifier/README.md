# Document image classifier

A CNN implementation of the document-classification workflow
from `CNN.ipynb`. The CNN accepts RGB images resized to `200 × 200` and
classifies them using an `ImageFolder` dataset.

## Dataset layout

Use the same class directories in both splits:

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

Dataset images may contain sensitive documents. Keep them outside source control;
`data/` is ignored by Git for this purpose.

## Setup

Install the locked dependencies from the repository root:

```bash
uv sync
```

## Train

On Windows with the dataset located on `D:`:

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

The default model destination is `artifacts/model.pt`. A checkpoint contains the
model weights, ordered class names, and random seed.

## Test

Run the automated checks with:

```bash
uv run pytest
```

See the repository [README](../README.md) for setup and training instructions.

## Use a checkpoint

```python
from cnn_classifier import load_checkpoint

model, classes, metadata = load_checkpoint("artifacts/model.pt")
```

`model` is returned in evaluation mode. Pass a batch of preprocessed
`(batch, 3, 200, 200)` tensors to obtain unnormalized class scores.
