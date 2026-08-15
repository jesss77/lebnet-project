import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.calibration.distribution import DistributionRecalibrator
from src.calibration.metrics import (
    accuracy,
    brier_score,
    calibration_bins,
    expected_calibration_error,
    negative_log_likelihood,
)
from src.data.dataset import get_cifar10
from src.models.resnet import get_resnet18


SEED = 42
BATCH_SIZE = 128

HIDDEN_DIM = 32

LEARNING_RATE = 1e-4
EPOCHS = 100
PATIENCE = 15


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_predictions(
    model,
    loader,
    device,
):
    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)

            logits = model(images)

            probs = torch.softmax(
                logits,
                dim=1,
            )

            probabilities.append(
                probs.cpu()
            )

            labels.append(
                targets.cpu()
            )

    return (
        torch.cat(probabilities),
        torch.cat(labels),
    )


def evaluate_probabilities(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
):
    logits = torch.log(
        probabilities.clamp_min(1e-8)
    )

    return {
        "accuracy": accuracy(
            logits,
            labels,
        ),
        "ece": expected_calibration_error(
            logits,
            labels,
        ),
        "nll": negative_log_likelihood(
            logits,
            labels,
        ),
        "brier": brier_score(
            logits,
            labels,
        ),
    }


def entropy(
    probabilities: torch.Tensor,
) -> torch.Tensor:
    return -(
        probabilities
        * torch.log(
            probabilities.clamp_min(1e-8)
        )
    ).sum(dim=1)


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Random seed: {SEED}")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    train_loader, calibration_loader, test_loader = (
        get_cifar10(
            batch_size=BATCH_SIZE
        )
    )

    # ---------------------------------------------------------
    # Load frozen baseline
    # ---------------------------------------------------------

    model = get_resnet18().to(device)

    checkpoint_path = Path(
        "results/baseline_resnet18.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find {checkpoint_path}. "
            "Run scripts.train_baseline first."
        )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    print("Baseline model loaded.")

    # ---------------------------------------------------------
    # Collect predictions
    # ---------------------------------------------------------

    print("Collecting calibration predictions...")

    (
        calibration_probabilities,
        calibration_labels,
    ) = collect_predictions(
        model,
        calibration_loader,
        device,
    )

    print("Collecting test predictions...")

    (
        test_probabilities,
        test_labels,
    ) = collect_predictions(
        model,
        test_loader,
        device,
    )

    num_classes = (
        calibration_probabilities.shape[1]
    )

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------

    baseline_metrics = evaluate_probabilities(
        test_probabilities,
        test_labels,
    )

    # ---------------------------------------------------------
    # Distribution recalibrator
    # ---------------------------------------------------------

    recalibrator = DistributionRecalibrator(
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    optimizer = Adam(
        recalibrator.parameters(),
        lr=LEARNING_RATE,
    )

    calibration_probabilities = (
        calibration_probabilities.to(device)
    )

    calibration_labels = (
        calibration_labels.to(device)
    )

    best_loss = float("inf")
    best_state = None

    epochs_without_improvement = 0

    print()
    print("=" * 60)
    print("DISTRIBUTION RECALIBRATION")
    print("=" * 60)

    for epoch in range(EPOCHS):

        recalibrator.train()

        calibrated_probabilities = (
            recalibrator(
                calibration_probabilities
            )
        )

        loss = F.nll_loss(
            torch.log(
                calibrated_probabilities.clamp_min(
                    1e-8
                )
            ),
            calibration_labels,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        current_loss = float(
            loss.item()
        )

        if current_loss < best_loss:

            best_loss = current_loss

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in recalibrator.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
        ):
            print(
                f"Epoch {epoch + 1:03d}/{EPOCHS} | "
                f"Calibration NLL: "
                f"{current_loss:.6f}"
            )

        if (
            epochs_without_improvement
            >= PATIENCE
        ):
            print(
                f"Early stopping at epoch "
                f"{epoch + 1}."
            )
            break

    if best_state is None:
        raise RuntimeError(
            "Distribution recalibrator "
            "failed to train."
        )

    recalibrator.load_state_dict(
        best_state
    )

    recalibrator.eval()

    # ---------------------------------------------------------
    # Test prediction
    # ---------------------------------------------------------

    test_probabilities_device = (
        test_probabilities.to(device)
    )

    with torch.no_grad():

        calibrated_test_probabilities = (
            recalibrator(
                test_probabilities_device
            )
            .cpu()
        )

    # ---------------------------------------------------------
    # Validate probability distribution
    # ---------------------------------------------------------

    row_sums = (
        calibrated_test_probabilities
        .sum(dim=1)
    )

    min_row_sum = float(
        row_sums.min().item()
    )

    max_row_sum = float(
        row_sums.max().item()
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    calibrated_metrics = (
        evaluate_probabilities(
            calibrated_test_probabilities,
            test_labels,
        )
    )

    # ---------------------------------------------------------
    # Confidence / entropy
    # ---------------------------------------------------------

    baseline_confidence = (
        test_probabilities
        .max(dim=1)
        .values
    )

    calibrated_confidence = (
        calibrated_test_probabilities
        .max(dim=1)
        .values
    )

    baseline_entropy = entropy(
        test_probabilities
    )

    calibrated_entropy = entropy(
        calibrated_test_probabilities
    )

    # ---------------------------------------------------------
    # Reliability diagrams
    # ---------------------------------------------------------

    baseline_bins = calibration_bins(
        test_probabilities,
        test_labels,
    )

    calibrated_bins = calibration_bins(
        calibrated_test_probabilities,
        test_labels,
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    results = {

        "method":
            "distribution_recalibration",

        "seed":
            SEED,

        "batch_size":
            BATCH_SIZE,

        "hidden_dim":
            HIDDEN_DIM,

        "learning_rate":
            LEARNING_RATE,

        "epochs":
            EPOCHS,

        "patience":
            PATIENCE,

        "num_classes":
            num_classes,

        "baseline":
            baseline_metrics,

        "distribution_recalibration":
            calibrated_metrics,

        "changes": {
            metric:
                calibrated_metrics[metric]
                - baseline_metrics[metric]
            for metric in baseline_metrics
        },

        "confidence": {

            "baseline_mean":
                float(
                    baseline_confidence
                    .mean()
                    .item()
                ),

            "calibrated_mean":
                float(
                    calibrated_confidence
                    .mean()
                    .item()
                ),
        },

        "entropy": {

            "baseline_mean":
                float(
                    baseline_entropy
                    .mean()
                    .item()
                ),

            "calibrated_mean":
                float(
                    calibrated_entropy
                    .mean()
                    .item()
                ),
        },

        "probability_validity": {

            "minimum_row_sum":
                min_row_sum,

            "maximum_row_sum":
                max_row_sum,
        },

        "reliability": {

            "baseline":
                baseline_bins,

            "distribution_recalibration":
                calibrated_bins,
        },
    }

    output_path = Path(
        "results/"
        "distribution_recalibration_results.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("BASELINE")
    print("=" * 60)

    for metric, value in (
        baseline_metrics.items()
    ):
        print(
            f"{metric.upper():<10}"
            f"{value:.4f}"
        )

    print()
    print("=" * 60)
    print("DISTRIBUTION RECALIBRATION")
    print("=" * 60)

    for metric, value in (
        calibrated_metrics.items()
    ):
        print(
            f"{metric.upper():<10}"
            f"{value:.4f}"
        )

    print()
    print("=" * 60)
    print("CHANGES")
    print("=" * 60)

    for metric, value in (
        results["changes"].items()
    ):
        print(
            f"{metric.upper():<10}"
            f"{value:+.4f}"
        )

    print()
    print(
        "Baseline mean confidence: "
        f"{results['confidence']['baseline_mean']:.4f}"
    )

    print(
        "Calibrated mean confidence: "
        f"{results['confidence']['calibrated_mean']:.4f}"
    )

    print(
        "Baseline mean entropy: "
        f"{results['entropy']['baseline_mean']:.4f}"
    )

    print(
        "Calibrated mean entropy: "
        f"{results['entropy']['calibrated_mean']:.4f}"
    )

    print()
    print(
        "Probability validity:"
    )

    print(
        f"  Minimum row sum: "
        f"{min_row_sum:.6f}"
    )

    print(
        f"  Maximum row sum: "
        f"{max_row_sum:.6f}"
    )

    print()
    print(
        f"Results saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()