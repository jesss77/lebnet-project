import json
from pathlib import Path

import torch

from src.data.dataset import get_cifar10
from src.models.resnet import get_resnet18
from src.calibration.temperature import TemperatureScaler
from src.calibration.metrics import (
    expected_calibration_error,
    negative_log_likelihood,
    brier_score,
    accuracy,
)


RESULTS_DIR = Path("results")


def collect_predictions(model, loader, device):
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
    return {
        "accuracy": float(accuracy(logits, labels)),
        "ece": float(expected_calibration_error(logits, labels)),
        "nll": float(negative_log_likelihood(logits, labels)),
        "brier": float(brier_score(logits, labels)),
    }


def print_metrics(name, metrics):
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _, calibration_loader, test_loader = get_cifar10(
        batch_size=128
    )

    model = get_resnet18().to(device)

    checkpoint_path = RESULTS_DIR / "baseline_resnet18.pt"

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    )

    print("Baseline model loaded.")

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
    # Baseline
    # ---------------------------------------------------------

    baseline_metrics = calculate_metrics(
        test_logits,
        test_labels,
    )

    print("\n" + "=" * 60)
    print("BASELINE")
    print("=" * 60)

    print_metrics(
        "Test Set",
        baseline_metrics,
    )

    # ---------------------------------------------------------
    # Temperature scaling
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TEMPERATURE SCALING")
    print("=" * 60)

    scaler = TemperatureScaler()

    print(f"Initial temperature: {scaler.temperature.item():.4f}")

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

    print_metrics(
        "Temperature-Scaled Test Set",
        calibrated_metrics,
    )

    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------

    changes = {
        metric: calibrated_metrics[metric]
        - baseline_metrics[metric]
        for metric in baseline_metrics
    }

    print("\n" + "=" * 60)
    print("BASELINE vs TEMPERATURE SCALING")
    print("=" * 60)

    print(
        f"{'Metric':<15}"
        f"{'Baseline':>12}"
        f"{'Calibrated':>15}"
        f"{'Change':>12}"
    )

    print("-" * 60)

    for metric in [
        "accuracy",
        "ece",
        "nll",
        "brier",
    ]:
        print(
            f"{metric.upper():<15}"
            f"{baseline_metrics[metric]:>12.4f}"
            f"{calibrated_metrics[metric]:>15.4f}"
            f"{changes[metric]:>12.4f}"
        )

    print("=" * 60)

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results = {
        "experiment": "temperature_scaling",
        "dataset": "CIFAR-10",
        "model": "ResNet-18",
        "calibration_fraction": 0.15,
        "temperature": float(learned_temperature),
        "baseline": baseline_metrics,
        "temperature_scaled": calibrated_metrics,
        "changes": changes,
    }

    output_path = RESULTS_DIR / "temperature_scaling_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()