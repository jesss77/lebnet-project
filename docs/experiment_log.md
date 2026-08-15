# Experiment Log

## Experiment 1 — ResNet-18 Baseline

Date: 2026-08-15

Dataset:
- CIFAR-10

Model:
- ResNet-18
- CIFAR-10-compatible first convolution
- Removed ImageNet max-pooling layer
- 10 output classes

Training:
- SGD
- Learning rate: 0.1
- Momentum: 0.9
- Weight decay: 5e-4
- Epochs: 10
- Batch size: 128
- Random seed: 42

Dataset split:
- 85% training
- 15% calibration
- 10,000 test samples

Results:

| Metric | Test |
|---|---:|
| Accuracy | 0.7432 |
| ECE | 0.0849 |
| NLL | 0.7967 |
| Brier | 0.3712 |

---

## Experiment 2 — Temperature Scaling

Purpose:

Establish a standard post-hoc calibration baseline.

Calibration:
- Held-out 15% calibration split
- Single learned temperature

Learned temperature:

    T = 1.3989

Results:

| Metric | Baseline | Temperature Scaling |
|---|---:|---:|
| Accuracy | 0.7432 | 0.7432 |
| ECE | 0.0849 | 0.0169 |
| NLL | 0.7967 | 0.7437 |
| Brier | 0.3712 | 0.3583 |

Observation:

Temperature scaling substantially improves calibration while
preserving classification accuracy.

---

## Experiment 3 — Distribution Calibrated Classification

Status: NEXT

Purpose:

Reproduce Algorithm 3 from Kuleshov & Deshpande (2022).

Input:

Full 10-dimensional softmax probability vector from ResNet-18.

Recalibrator:

Dense neural network mapping the probability simplex to the
probability simplex.

Training:

- Calibration set only
- Cross-entropy/log loss
- ResNet-18 weights frozen

Evaluation:

- Accuracy
- ECE
- NLL
- Brier
- Reliability diagram

Comparison:

Baseline vs Temperature Scaling vs Distribution Recalibration.

---

## Final Research Question

Does the distribution recalibration method from Kuleshov &
Deshpande improve calibration over the original classifier while
preserving predictive performance?

A secondary question is whether the more expressive distribution
recalibrator provides benefits over simple one-parameter
temperature scaling.