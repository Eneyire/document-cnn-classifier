# Classifier package

The `cnn_classifier` package contains the document-image CNN, `ImageFolder`
data loading, training/evaluation functions, and checkpoint persistence.

The implementation preserves the notebook's 200×200 input size, architecture,
SGD optimizer, learning rate, batch size, and 0–255 pixel scale while correcting
its malformed image reshape and device-dependent evaluation issues.

For installation, dataset layout, and command-line usage, see the repository
[README](../README.md).
