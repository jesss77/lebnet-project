import json
from pathlib import Path

import torch

from src.calibration.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)
from src.data.dataset import get_cifar10
from src.models.resnet import get_resnet18


SEED = 42
CHECKPOINT_PATH = Path("results/baseline_resnet18.pt")
OUTPUT_PATH = Path("results/isotonic_recalibration_results.json")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
):
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_logits), torch.cat(all_labels)


def logits_to_confidence(
    logits: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    confidence, _ = probabilities.max(dim=1)

    return confidence


def logits_to_predictions(
    logits: torch.Tensor,
) -> torch.Tensor:
    return logits.argmax(dim=1)


class IsotonicCalibrator:
    """
    Isotonic regression implemented using the
    Pool Adjacent Violators Algorithm (PAVA).

    The calibrator learns a monotonically non-decreasing
    mapping from model confidence to empirical correctness.
    """

    def __init__(self):
        self.x = None
        self.y = None

    def fit(
        self,
        confidence: torch.Tensor,
        correct: torch.Tensor,
    ) -> None:
        confidence = confidence.detach().cpu().float()
        correct = correct.detach().cpu().float()

        if confidence.numel() == 0:
            raise ValueError(
                "Cannot fit isotonic regression on empty data."
            )

        if confidence.shape != correct.shape:
            raise ValueError(
                "confidence and correct must have the same shape."
            )

        order = torch.argsort(confidence)

        x = confidence[order]
        y = correct[order]

        block_x = []
        block_y = []
        block_weight = []

        for i in range(len(x)):
            block_x.append(float(x[i].item()))
            block_y.append(float(y[i].item()))
            block_weight.append(1.0)

            while (
                len(block_y) >= 2
                and block_y[-2] > block_y[-1]
            ):
                weight_left = block_weight[-2]
                weight_right = block_weight[-1]

                merged_weight = (
                    weight_left + weight_right
                )

                merged_y = (
                    block_y[-2] * weight_left
                    + block_y[-1] * weight_right
                ) / merged_weight

                merged_x = block_x[-1]

                block_x[-2] = merged_x
                block_y[-2] = merged_y
                block_weight[-2] = merged_weight

                block_x.pop()
                block_y.pop()
                block_weight.pop()

        self.x = torch.tensor(
            block_x,
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            block_y,
            dtype=torch.float32,
        )

    def predict(
        self,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        if self.x is None or self.y is None:
            raise RuntimeError(
                "IsotonicCalibrator must be fitted before prediction."
            )

        confidence = confidence.detach().cpu().float()

        indices = torch.bucketize(
            confidence,
            self.x,
            right=True,
        ) - 1

        indices = indices.clamp(
            min=0,
            max=len(self.y) - 1,
        )

        return self.y[indices]


def recalibrate_probabilities(
    logits: torch.Tensor,
    calibrated_confidence: torch.Tensor,
) -> torch.Tensor:
    """
    Recalibrate the top-class probability while preserving
    the original relative distribution among the remaining
    classes.

    If the original model assigns probabilities:

        p_top, p_1, p_2, ..., p_k

    and isotonic regression changes p_top to q, the remaining
    probability mass (1-q) is distributed according to the
    original proportions of the non-top classes.

    This is substantially less destructive than replacing the
    remaining probability mass uniformly.
    """

    probabilities = torch.softmax(logits, dim=1)

    predictions = probabilities.argmax(dim=1)

    num_samples, num_classes = probabilities.shape

    calibrated_confidence = calibrated_confidence.clamp(
        min=1e-6,
        max=1.0 - 1e-6,
    )

    calibrated_probabilities = probabilities.clone()

    row_indices = torch.arange(
        num_samples,
        dtype=torch.long,
    )

    original_top_probability = probabilities[
        row_indices,
        predictions,
    ]

    non_top_mask = torch.ones_like(
        probabilities,
        dtype=torch.bool,
    )

    non_top_mask[
        row_indices,
        predictions,
    ] = False

    non_top_probabilities = probabilities[
        non_top_mask
    ].reshape(num_samples, num_classes - 1)

    non_top_sum = non_top_probabilities.sum(
        dim=1,
        keepdim=True,
    )

    normalized_non_top = (
        non_top_probabilities
        / non_top_sum.clamp_min(1e-12)
    )

    remaining_probability = (
        1.0 - calibrated_confidence
    ).unsqueeze(1)

    recalibrated_non_top = (
        normalized_non_top
        * remaining_probability
    )

    calibrated_probabilities[
        non_top_mask
    ] = recalibrated_non_top.reshape(-1)

    calibrated_probabilities[
        row_indices,
        predictions,
    ] = calibrated_confidence

    return calibrated_probabilities


def probabilities_to_logits(
    probabilities: torch.Tensor,
) -> torch.Tensor:
    return torch.log(
        probabilities.clamp_min(1e-12)
    )


def evaluate_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
):
    return {
        "accuracy": accuracy(logits, labels),
        "ece": expected_calibration_error(logits, labels),
        "nll": negative_log_likelihood(logits, labels),
        "brier": brier_score(logits, labels),
    }


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find baseline checkpoint: "
            f"{CHECKPOINT_PATH}\n"
            "Run `python -m scripts.train_baseline` first."
        )

    # ---------------------------------------------------------
    # DATA
    # ---------------------------------------------------------

    _, calibration_loader, test_loader = get_cifar10(
        batch_size=128,
        seed=SEED,
    )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    model = get_resnet18().to(device)

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    print("Baseline model loaded.")

    # ---------------------------------------------------------
    # COLLECT PREDICTIONS
    # ---------------------------------------------------------

    print("Collecting calibration predictions...")

    calibration_logits, calibration_labels = collect_predictions(
        model,
        calibration_loader,
        device,
    )

    print("Collecting test predictions...")

    test_logits, test_labels = collect_predictions(
        model,
        test_loader,
        device,
    )

    # ---------------------------------------------------------
    # BASELINE
    # ---------------------------------------------------------

    baseline_metrics = evaluate_logits(
        test_logits,
        test_labels,
    )

    # ---------------------------------------------------------
    # FIT ISOTONIC REGRESSION
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("ISOTONIC REGRESSION")
    print("=" * 60)

    calibration_confidence = logits_to_confidence(
        calibration_logits
    )

    calibration_predictions = logits_to_predictions(
        calibration_logits
    )

    calibration_correct = (
        calibration_predictions == calibration_labels
    ).float()

    calibrator = IsotonicCalibrator()

    calibrator.fit(
        calibration_confidence,
        calibration_correct,
    )

    # ---------------------------------------------------------
    # APPLY TO TEST SET
    # ---------------------------------------------------------

    test_confidence = logits_to_confidence(
        test_logits
    )

    calibrated_test_confidence = calibrator.predict(
        test_confidence
    )

    calibrated_test_probabilities = (
        recalibrate_probabilities(
            test_logits,
            calibrated_test_confidence,
        )
    )

    calibrated_test_logits = probabilities_to_logits(
        calibrated_test_probabilities
    )

    isotonic_metrics = evaluate_logits(
        calibrated_test_logits,
        test_labels,
    )

    # ---------------------------------------------------------
    # VERIFY PROBABILITY VALIDITY
    # ---------------------------------------------------------

    probability_sums = (
        calibrated_test_probabilities.sum(dim=1)
    )

    print()
    print("Probability validity:")
    print(
        f"  Minimum row sum: "
        f"{probability_sums.min().item():.6f}"
    )
    print(
        f"  Maximum row sum: "
        f"{probability_sums.max().item():.6f}"
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print()
    print("Baseline:")
    print(
        f"  Accuracy: {baseline_metrics['accuracy']:.4f}"
    )
    print(
        f"  ECE:      {baseline_metrics['ece']:.4f}"
    )
    print(
        f"  NLL:      {baseline_metrics['nll']:.4f}"
    )
    print(
        f"  Brier:    {baseline_metrics['brier']:.4f}"
    )

    print()
    print("Isotonic Regression:")
    print(
        f"  Accuracy: {isotonic_metrics['accuracy']:.4f}"
    )
    print(
        f"  ECE:      {isotonic_metrics['ece']:.4f}"
    )
    print(
        f"  NLL:      {isotonic_metrics['nll']:.4f}"
    )
    print(
        f"  Brier:    {isotonic_metrics['brier']:.4f}"
    )

    print()
    print("Changes:")

    for metric in [
        "accuracy",
        "ece",
        "nll",
        "brier",
    ]:
        change = (
            isotonic_metrics[metric]
            - baseline_metrics[metric]
        )

        print(
            f"  {metric.upper():<10} "
            f"{change:+.4f}"
        )

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    results = {
        "experiment": "isotonic_recalibration",
        "seed": SEED,
        "dataset": "CIFAR-10",
        "model": "ResNet18",
        "calibration_fraction": 0.15,
        "baseline": baseline_metrics,
        "isotonic_regression": isotonic_metrics,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print()
    print(
        f"Results saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()