import json
from pathlib import Path

import torch

from src.data.dataset import get_cifar10
from src.models.resnet import get_resnet18
from src.calibration.temperature import TemperatureScaler
from src.calibration.metrics import (
    accuracy,
    brier_score,
    calibration_bins,
    expected_calibration_error,
    negative_log_likelihood,
)


def collect_predictions(model, loader, device):
    """
    Collect model logits and labels for an entire dataset split.
    """
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

    return (
        torch.cat(all_logits),
        torch.cat(all_labels),
    )


def calculate_metrics(logits, labels):
    """
    Calculate all calibration metrics for a set of logits.
    """
    return {
        "accuracy": accuracy(logits, labels),
        "ece": expected_calibration_error(logits, labels),
        "nll": negative_log_likelihood(logits, labels),
        "brier": brier_score(logits, labels),
    }


def print_metrics(name, metrics):
    """
    Print a consistent metrics block.
    """
    print(f"\n{name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ECE:      {metrics['ece']:.4f}")
    print(f"  NLL:      {metrics['nll']:.4f}")
    print(f"  Brier:    {metrics['brier']:.4f}")


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    _, calibration_loader, test_loader = get_cifar10(
        batch_size=128,
    )

    # ---------------------------------------------------------
    # Load baseline model
    # ---------------------------------------------------------

    checkpoint_path = Path(
        "results/baseline_resnet18.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {checkpoint_path}\n"
            "Run `python -m scripts.train_baseline` first."
        )

    model = get_resnet18().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint)

    print("Baseline model loaded.")

    # ---------------------------------------------------------
    # Collect predictions
    # ---------------------------------------------------------

    print("Collecting calibration predictions...")

    calibration_logits, calibration_labels = (
        collect_predictions(
            model,
            calibration_loader,
            device,
        )
    )

    print("Collecting test predictions...")

    test_logits, test_labels = collect_predictions(
        model,
        test_loader,
        device,
    )

    # ---------------------------------------------------------
    # Baseline evaluation
    # ---------------------------------------------------------

    baseline_metrics = calculate_metrics(
        test_logits,
        test_labels,
    )

    print("\n" + "=" * 60)
    print("BASELINE")
    print("=" * 60)

    print("\nTest Set:")

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

    # ---------------------------------------------------------
    # Temperature scaling
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TEMPERATURE SCALING")
    print("=" * 60)

    scaler = TemperatureScaler(
        initial_temperature=1.0
    )

    print(
        "Initial temperature: "
        f"{scaler.get_temperature():.4f}"
    )

    learned_temperature = scaler.fit(
        calibration_logits,
        calibration_labels,
    )

    print(
        f"Learned temperature: "
        f"{learned_temperature:.4f}"
    )

    calibrated_test_logits = scaler.transform(
        test_logits
    )

    calibrated_metrics = calculate_metrics(
        calibrated_test_logits,
        test_labels,
    )

    print("\nTemperature-Scaled Test Set:")

    print(
        f"  Accuracy: {calibrated_metrics['accuracy']:.4f}"
    )

    print(
        f"  ECE:      {calibrated_metrics['ece']:.4f}"
    )

    print(
        f"  NLL:      {calibrated_metrics['nll']:.4f}"
    )

    print(
        f"  Brier:    {calibrated_metrics['brier']:.4f}"
    )

    # ---------------------------------------------------------
    # Calibration-bin statistics
    # ---------------------------------------------------------

    baseline_bins = calibration_bins(
        test_logits,
        test_labels,
        num_bins=10,
    )

    calibrated_bins = calibration_bins(
        calibrated_test_logits,
        test_labels,
        num_bins=10,
    )

    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------

    changes = {
        "accuracy": (
            calibrated_metrics["accuracy"]
            - baseline_metrics["accuracy"]
        ),
        "ece": (
            calibrated_metrics["ece"]
            - baseline_metrics["ece"]
        ),
        "nll": (
            calibrated_metrics["nll"]
            - baseline_metrics["nll"]
        ),
        "brier": (
            calibrated_metrics["brier"]
            - baseline_metrics["brier"]
        ),
    }

    print("\n" + "=" * 60)
    print("BASELINE vs TEMPERATURE SCALING")
    print("=" * 60)

    print(
        "Metric             Baseline     Calibrated      Change"
    )
    print("-" * 60)

    print(
        f"ACCURACY            "
        f"{baseline_metrics['accuracy']:.4f}"
        f"         "
        f"{calibrated_metrics['accuracy']:.4f}"
        f"      "
        f"{changes['accuracy']:.4f}"
    )

    print(
        f"ECE                 "
        f"{baseline_metrics['ece']:.4f}"
        f"         "
        f"{calibrated_metrics['ece']:.4f}"
        f"     "
        f"{changes['ece']:.4f}"
    )

    print(
        f"NLL                 "
        f"{baseline_metrics['nll']:.4f}"
        f"         "
        f"{calibrated_metrics['nll']:.4f}"
        f"     "
        f"{changes['nll']:.4f}"
    )

    print(
        f"BRIER               "
        f"{baseline_metrics['brier']:.4f}"
        f"         "
        f"{calibrated_metrics['brier']:.4f}"
        f"     "
        f"{changes['brier']:.4f}"
    )

    print("=" * 60)

    # ---------------------------------------------------------
    # Save experiment results
    # ---------------------------------------------------------

    results = {
        "experiment": "temperature_scaling",
        "dataset": "CIFAR-10",
        "model": "ResNet-18",
        "calibration_fraction": 0.15,
        "temperature": learned_temperature,
        "baseline": baseline_metrics,
        "temperature_scaled": calibrated_metrics,
        "changes": changes,
        "reliability": {
            "baseline": baseline_bins,
            "temperature_scaled": calibrated_bins,
        },
    }

    results_path = Path(
        "results/temperature_scaling_results.json"
    )

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        f"\nResults saved to: {results_path}"
    )


if __name__ == "__main__":
    main()