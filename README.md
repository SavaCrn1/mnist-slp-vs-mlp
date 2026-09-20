# From Perceptron to MLP — Handwritten Digit Recognition

CS/DATA 3440-51 Machine Learning, Project 1 (Fall 2026) — John Carroll University.

A Single-Layer Perceptron written in class was adapted to the MNIST dataset (784 inputs
instead of 64) and compared against scikit-learn Multilayer Perceptrons across four
hidden-layer architectures. Two design decisions are investigated: the learning rate of
the SLP, and the hidden-layer architecture of the MLP.

## Data

MNIST via `fetch_openml("mnist_784", version=1)` — 70,000 images, 784 features, 10 classes,
pixel values 0-255 normalized to 0-1. Split 80/20 with `random_state=42`, stratified on the
target, giving 56,000 training and 14,000 test images. The same split is used for every model.

## Results

SLP across learning rates, 10 epochs each, seed and split held constant:

| Learning Rate | Epochs | Training Accuracy (%) | Test Accuracy (%) |
| --- | --- | --- | --- |
| 0.001 | 10 | 79.63 | 89.47 |
| 0.01 | 10 | 79.47 | 86.58 |
| 1.0 | 10 | 79.32 | 85.89 |

SLP against the MLP architectures:

| Model | Hidden Layer(s) | Architecture | Training Accuracy (%) | Test Accuracy (%) |
| --- | --- | --- | --- | --- |
| SLP | None | 784 → 10 | 79.63 | 89.47 |
| MLP 1 | 32 | 784 → 32 → 10 | 99.99 | 96.50 |
| MLP 2 | 64 | 784 → 64 → 10 | 100.00 | 97.61 |
| MLP 3 | 128 | 784 → 128 → 10 | 99.98 | 97.96 |
| MLP 4 | 64, 32 | 784 → 64 → 32 → 10 | 100.00 | 97.52 |

Best model on unseen data: MLP 3, the 784 → 128 → 10 network, at 97.96% test accuracy.

## Files

| File | Contents |
| --- | --- |
| `Project_Code.py` | Full implementation: data preparation, SLP training/testing, MLP training/testing, evaluation |
| `Project_1_Report.pdf` | Written report with tables, figures and analysis |
| `Insights and Analysis.txt` | Source text for the report's written analysis |
| `Tables for Project_1.xlsx` | Result tables |
| `Plots/` | Learning-rate plot, confusion matrices, correct and incorrect prediction images |

## Running

```
pip install numpy matplotlib scikit-learn
python Project_Code.py
```

MNIST downloads on first run and is cached locally. The script is organized in `#%%` cells
for Spyder; running it top to bottom reproduces every number reported above.
