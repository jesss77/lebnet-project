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


def print_metrics(name, logits, labels):
    acc = accuracy(logits, labels)
    ece = expected_calibration_error(logits, labels)
    nll = negative_log_likelihood(logits, labels)
    brier = brier_score(logits, labels)

    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  ECE:      {ece:.4f}")
    print(f"  NLL:      {nll:.4f}")
    print(f"  Brier:    {brier:.4f}")

    return {
        "accuracy": acc,
        "ece": ece,
        "nll": nll,
        "brier": brier,
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Load the exact same dataset split used during
    # baseline evaluation.
    _, calibration_loader, test_loader = get_cifar10(
        batch_size=128
    )

    # Recreate the exact baseline architecture.
    model = get_resnet18(num_classes=10)

    checkpoint = torch.load(
        "results/baseline_resnet18.pt",
        map_location=device,
    )

    model.load_state_dict(checkpoint)
    model.to(device)

    print("Baseline model loaded.")

    # ---------------------------------------------------------
    # Collect logits
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
    # Baseline test metrics
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("BASELINE")
    print("=" * 60)

    baseline_metrics = print_metrics(
        "Test Set",
        test_logits,
        test_labels,
    )

    # ---------------------------------------------------------
    # Fit temperature using ONLY calibration data
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("TEMPERATURE SCALING")
    print("=" * 60)

    scaler = TemperatureScaler(
        initial_temperature=1.0
    ).to(device)

    print(
        f"Initial temperature: "
        f"{scaler.get_temperature():.4f}"
    )

    scaler.fit(
        calibration_logits,
        calibration_labels,
    )

    temperature = scaler.get_temperature()

    print(
        f"Learned temperature: "
        f"{temperature:.4f}"
    )

    # ---------------------------------------------------------
    # Apply learned temperature to TEST logits
    # ---------------------------------------------------------

    calibrated_test_logits = scaler(
        test_logits.to(device)
    ).cpu()

    # ---------------------------------------------------------
    # Calibrated test metrics
    # ---------------------------------------------------------

    calibrated_metrics = print_metrics(
        "Temperature-Scaled Test Set",
        calibrated_test_logits,
        test_labels,
    )

    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("BASELINE vs TEMPERATURE SCALING")
    print("=" * 60)

    print(
        f"{'Metric':<12}"
        f"{'Baseline':>14}"
        f"{'Calibrated':>16}"
        f"{'Change':>14}"
    )

    print("-" * 56)

    for metric in ["accuracy", "ece", "nll", "brier"]:
        baseline = baseline_metrics[metric]
        calibrated = calibrated_metrics[metric]
        change = calibrated - baseline

        print(
            f"{metric.upper():<12}"
            f"{baseline:>14.4f}"
            f"{calibrated:>16.4f}"
            f"{change:>14.4f}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()