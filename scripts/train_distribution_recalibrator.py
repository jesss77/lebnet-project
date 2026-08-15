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


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

EPOCHS = 100

PATIENCE = 15

# Different recalibrator capacities.
#
# ()          -> linear recalibrator
# (32,)       -> 10 -> 32 -> 10
# (64,)       -> 10 -> 64 -> 10
# (64, 64)    -> 10 -> 64 -> 64 -> 10
#
# The final architecture is the one used previously.
EXPERIMENTS = {
    "linear": (),
    "hidden_32": (32,),
    "hidden_64": (64,),
    "hidden_64x64": (64, 64),
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# MODEL PREDICTIONS
# ============================================================

def collect_predictions(
    model,
    loader,
    device,
):
    """
    Collect frozen model logits, probabilities, and labels.
    """

    model.eval()

    logits_list = []

    probabilities_list = []

    labels_list = []

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(device)

            logits = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            logits_list.append(
                logits.cpu()
            )

            probabilities_list.append(
                probabilities.cpu()
            )

            labels_list.append(
                targets.cpu()
            )

    return (
        torch.cat(logits_list),
        torch.cat(probabilities_list),
        torch.cat(labels_list),
    )


# ============================================================
# PROBABILITY METRICS
# ============================================================

def evaluate_probabilities(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
):
    """
    Evaluate a probability distribution using the project's
    standard classification metrics.
    """

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


# ============================================================
# ENTROPY
# ============================================================

def mean_entropy(
    probabilities: torch.Tensor,
) -> float:
    """
    Compute mean predictive entropy.
    """

    entropy = -(
        probabilities
        * torch.log(
            probabilities.clamp_min(1e-8)
        )
    ).sum(dim=1)

    return float(
        entropy.mean().item()
    )


# ============================================================
# TRAIN ONE RECALIBRATOR
# ============================================================

def train_recalibrator(
    calibration_probabilities: torch.Tensor,
    calibration_labels: torch.Tensor,
    num_classes: int,
    hidden_dims: tuple[int, ...],
    device: torch.device,
):
    """
    Train one distribution recalibrator on the calibration set.
    """

    recalibrator = DistributionRecalibrator(
        num_classes=num_classes,
        hidden_dims=hidden_dims,
    ).to(device)

    optimizer = Adam(
        recalibrator.parameters(),
        lr=LEARNING_RATE,
    )

    best_loss = float("inf")

    best_state = None

    epochs_without_improvement = 0

    calibration_probabilities = (
        calibration_probabilities.to(device)
    )

    calibration_labels = (
        calibration_labels.to(device)
    )

    print()

    print(
        f"Architecture: "
        f"{num_classes}"
        f"{''.join(f' -> {dim}' for dim in hidden_dims)}"
        f" -> {num_classes}"
    )

    for epoch in range(EPOCHS):

        recalibrator.train()

        calibrated_probabilities = recalibrator(
            calibration_probabilities
        )

        loss = F.nll_loss(
            torch.log(
                calibrated_probabilities.clamp_min(1e-8)
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

    return recalibrator


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    print(
        f"Random seed: {SEED}"
    )

    # --------------------------------------------------------
    # Load CIFAR-10
    # --------------------------------------------------------

    (
        train_loader,
        calibration_loader,
        test_loader,
    ) = get_cifar10(
        batch_size=BATCH_SIZE
    )

    # --------------------------------------------------------
    # Load frozen baseline model
    # --------------------------------------------------------

    model = get_resnet18().to(device)

    checkpoint_path = Path(
        "results/baseline_resnet18.pt"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Could not find "
            f"{checkpoint_path}. "
            f"Run scripts.train_baseline first."
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

    print(
        "Baseline model loaded."
    )

    # --------------------------------------------------------
    # Collect calibration predictions
    # --------------------------------------------------------

    print(
        "Collecting calibration predictions..."
    )

    (
        calibration_logits,
        calibration_probabilities,
        calibration_labels,
    ) = collect_predictions(
        model,
        calibration_loader,
        device,
    )

    # --------------------------------------------------------
    # Collect test predictions
    # --------------------------------------------------------

    print(
        "Collecting test predictions..."
    )

    (
        test_logits,
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

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_metrics = evaluate_probabilities(
        test_probabilities,
        test_labels,
    )

    baseline_confidence = (
        test_probabilities.max(
            dim=1
        ).values
    )

    baseline_entropy = mean_entropy(
        test_probabilities
    )

    baseline_bins = calibration_bins(
        test_probabilities,
        test_labels,
    )

    # --------------------------------------------------------
    # Store experiment results
    # --------------------------------------------------------

    experiment_results = {}

    # --------------------------------------------------------
    # Run capacity experiments
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "DISTRIBUTION RECALIBRATION "
        "CAPACITY ABLATION"
    )

    print("=" * 60)

    for experiment_name, hidden_dims in EXPERIMENTS.items():

        print()

        print("=" * 60)

        print(
            f"EXPERIMENT: "
            f"{experiment_name}"
        )

        print("=" * 60)

        # Reset seed before every experiment so
        # initialization is reproducible.
        set_seed(SEED)

        recalibrator = train_recalibrator(
            calibration_probabilities,
            calibration_labels,
            num_classes,
            hidden_dims,
            device,
        )

        # ----------------------------------------------------
        # Apply recalibrator to test set
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Validate probability distributions
        # ----------------------------------------------------

        row_sums = (
            calibrated_test_probabilities.sum(
                dim=1
            )
        )

        min_row_sum = float(
            row_sums.min().item()
        )

        max_row_sum = float(
            row_sums.max().item()
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        calibrated_metrics = (
            evaluate_probabilities(
                calibrated_test_probabilities,
                test_labels,
            )
        )

        calibrated_confidence = (
            calibrated_test_probabilities.max(
                dim=1
            ).values
        )

        calibrated_entropy = (
            mean_entropy(
                calibrated_test_probabilities
            )
        )

        calibrated_bins = calibration_bins(
            calibrated_test_probabilities,
            test_labels,
        )

        # ----------------------------------------------------
        # Changes
        # ----------------------------------------------------

        changes = {
            metric:
                calibrated_metrics[metric]
                - baseline_metrics[metric]
            for metric in baseline_metrics
        }

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        experiment_results[
            experiment_name
        ] = {

            "hidden_dims": list(
                hidden_dims
            ),

            "baseline": baseline_metrics,

            "distribution_recalibration":
                calibrated_metrics,

            "changes": changes,

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
                    baseline_entropy,

                "calibrated_mean":
                    calibrated_entropy,
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

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print()

        print(
            "Baseline:"
        )

        for metric, value in (
            baseline_metrics.items()
        ):

            print(
                f"  {metric.upper():<10}"
                f"{value:.4f}"
            )

        print()

        print(
            "Distribution Recalibration:"
        )

        for metric, value in (
            calibrated_metrics.items()
        ):

            print(
                f"  {metric.upper():<10}"
                f"{value:.4f}"
            )

        print()

        print(
            "Changes:"
        )

        for metric, value in changes.items():

            print(
                f"  {metric.upper():<10}"
                f"{value:+.4f}"
            )

        print()

        print(
            "Mean confidence:"
        )

        print(
            f"  Baseline:   "
            f"{baseline_confidence.mean().item():.4f}"
        )

        print(
            f"  Calibrated: "
            f"{calibrated_confidence.mean().item():.4f}"
        )

        print()

        print(
            "Mean entropy:"
        )

        print(
            f"  Baseline:   "
            f"{baseline_entropy:.4f}"
        )

        print(
            f"  Calibrated: "
            f"{calibrated_entropy:.4f}"
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

    # --------------------------------------------------------
    # Final results object
    # --------------------------------------------------------

    results = {

        "method":
            "distribution_recalibration_capacity_ablation",

        "seed":
            SEED,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "epochs":
            EPOCHS,

        "patience":
            PATIENCE,

        "num_classes":
            num_classes,

        "baseline":
            {

                "metrics":
                    baseline_metrics,

                "mean_confidence":
                    float(
                        baseline_confidence
                        .mean()
                        .item()
                    ),

                "mean_entropy":
                    baseline_entropy,

                "reliability":
                    baseline_bins,
            },

        "experiments":
            experiment_results,
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Final comparison table
    # --------------------------------------------------------

    print()

    print("=" * 80)

    print(
        "FINAL DISTRIBUTION RECALIBRATION "
        "CAPACITY COMPARISON"
    )

    print("=" * 80)

    print(
        f"{'Method':<20}"
        f"{'Accuracy':>12}"
        f"{'ECE':>12}"
        f"{'NLL':>12}"
        f"{'Brier':>12}"
    )

    print("-" * 80)

    print(
        f"{'Baseline':<20}"
        f"{baseline_metrics['accuracy']:>12.4f}"
        f"{baseline_metrics['ece']:>12.4f}"
        f"{baseline_metrics['nll']:>12.4f}"
        f"{baseline_metrics['brier']:>12.4f}"
    )

    for experiment_name, experiment in (
        experiment_results.items()
    ):

        metrics = experiment[
            "distribution_recalibration"
        ]

        print(
            f"{experiment_name:<20}"
            f"{metrics['accuracy']:>12.4f}"
            f"{metrics['ece']:>12.4f}"
            f"{metrics['nll']:>12.4f}"
            f"{metrics['brier']:>12.4f}"
        )

    print("=" * 80)

    print()

    print(
        f"Results saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
    