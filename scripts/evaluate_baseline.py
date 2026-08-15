import torch

from src.data.dataset import get_cifar10
from src.models.resnet import get_resnet18
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


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Load CIFAR-10.
    # The training split is divided into:
    #   - 42,500 training samples
    #   - 7,500 calibration samples
    # The official test set contains 10,000 samples.
    _, calibration_loader, test_loader = get_cifar10(
        batch_size=128
    )

    # Create the exact same ResNet-18 architecture
    # used during baseline training.
    model = get_resnet18(num_classes=10)

    checkpoint = torch.load(
        "results/baseline_resnet18.pt",
        map_location=device,
    )

    model.load_state_dict(checkpoint)
    model.to(device)

    print("Model loaded.")

    # Collect predictions on the calibration set.
    print("Evaluating calibration set...")

    calibration_logits, calibration_labels = collect_predictions(
        model,
        calibration_loader,
        device,
    )

    # Collect predictions on the untouched test set.
    print("Evaluating test set...")

    test_logits, test_labels = collect_predictions(
        model,
        test_loader,
        device,
    )

    # ---------------------------------------------------------
    # Calibration-set metrics
    # ---------------------------------------------------------

    calibration_accuracy = accuracy(
        calibration_logits,
        calibration_labels,
    )

    calibration_ece = expected_calibration_error(
        calibration_logits,
        calibration_labels,
    )

    calibration_nll = negative_log_likelihood(
        calibration_logits,
        calibration_labels,
    )

    calibration_brier = brier_score(
        calibration_logits,
        calibration_labels,
    )

    # ---------------------------------------------------------
    # Test-set metrics
    # ---------------------------------------------------------

    test_accuracy = accuracy(
        test_logits,
        test_labels,
    )

    test_ece = expected_calibration_error(
        test_logits,
        test_labels,
    )

    test_nll = negative_log_likelihood(
        test_logits,
        test_labels,
    )

    test_brier = brier_score(
        test_logits,
        test_labels,
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print()
    print("=" * 50)
    print("BASELINE CALIBRATION RESULTS")
    print("=" * 50)

    print("\nCalibration Set:")
    print(f"  Accuracy: {calibration_accuracy:.4f}")
    print(f"  ECE:      {calibration_ece:.4f}")
    print(f"  NLL:      {calibration_nll:.4f}")
    print(f"  Brier:    {calibration_brier:.4f}")

    print("\nTest Set:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  ECE:      {test_ece:.4f}")
    print(f"  NLL:      {test_nll:.4f}")
    print(f"  Brier:    {test_brier:.4f}")

    print("=" * 50)


if __name__ == "__main__":
    main()