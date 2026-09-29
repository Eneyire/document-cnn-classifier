# Document CNN Classifier

A PyTorch image classifier for driving licenses, social security cards, and other documents. It expects RGB images resized to `200 × 200` in an `ImageFolder` directory structure.

## Dataset layout

Provide matching class directories in the training and testing splits:

```text
<dataset-root>/
├── Training_data/
│   ├── driving_license/
│   ├── others/
│   └── social_security/
└── Testing_data/
    ├── driving_license/
    ├── others/
    └── social_security/
```

Keep sensitive dataset images outside the repository. Git ignores the contents of `data/` and generated files in `artifacts/`.

## Setup

Install the locked dependencies from the repository root with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Train and compare options

The command creates a seeded, stratified validation split from the training directory (20% by default; change it with `--validation-fraction`). The best epoch is selected by validation accuracy. For candidate comparisons, omit the optional test directory so the held-out test data is not read or used for model selection.

Choose an architecture and input pixel scale:

- `baseline`: the compact classifier used as the project's reference model.
- `feature`: a deeper convolutional feature extractor with global pooling.
- `byte`: pixel values remain in the 0–255 range.
- `unit`: pixel values are scaled to the 0–1 range.
- `sgd` or `adam`: optimizer choice; tune it using validation results.

To compare candidates, omit the test directory. This runs validation-only experiments and prevents test-set results from influencing model choice. Use a distinct checkpoint path for each run:

```bash
uv run document-cnn-train "<local-dataset-root>/Training_data" \
  --architecture baseline --pixel-scale byte --optimizer sgd \
  --learning-rate 0.0001 --model artifacts/baseline-byte.pt \
  --epochs 10 --seed 42
```

Repeat with `--pixel-scale unit`, `--architecture feature`, and/or `--optimizer adam`. Keep the validation fraction, seed, and other settings consistent. Learning rate may need tuning for each model/scale/optimizer combination; compare those choices using validation results only.

After choosing a configuration, evaluate it once on the held-out test split:

```bash
uv run document-cnn-train \
  "<local-dataset-root>/Training_data" \
  "<local-dataset-root>/Testing_data" \
  --architecture baseline --pixel-scale unit --optimizer sgd \
  --model artifacts/model.pt --epochs 10 --learning-rate 0.01 --seed 42
```

Defaults are 10 epochs, batch size 8, seed 42, the baseline architecture, unit-scale inputs, SGD, and learning rate `0.01`. This baseline/unit/SGD configuration achieved 91.67% validation accuracy in the seeded comparison for this project. Its single held-out test evaluation was 90.00%; these are results from one split and are not a performance guarantee for other datasets. When a test directory is supplied, training reports test accuracy, per-class precision/recall/F1/support, and a confusion matrix (rows are true classes; columns are predicted classes). Checkpoints record the architecture, pixel scale, optimizer, ordered classes, seed, and best validation accuracy.

The training module can also be invoked directly:

```bash
uv run python -m cnn_classifier.train \
  "<local-dataset-root>/Training_data" \
  "<local-dataset-root>/Testing_data" \
  --architecture baseline --pixel-scale unit --optimizer sgd \
  --learning-rate 0.01 --epochs 10 --seed 42
```

## Load a checkpoint

```python
from cnn_classifier.checkpoint import load_checkpoint

model, classes, metadata = load_checkpoint("artifacts/model.pt")
```

The returned model is in evaluation mode. Use `metadata["pixel_scale"]` to apply the matching input scale (`byte` or `unit`) before prediction.

## Tests

Run the automated suite with pytest:

```bash
uv run pytest
```
