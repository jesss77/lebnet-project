import torch
import torch.nn.functional as F


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    """
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def negative_log_likelihood(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute multiclass negative log-likelihood.
    """
    loss = F.cross_entropy(logits, labels)
    return float(loss.item())


def brier_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute the multiclass Brier score.
    """
    probabilities = F.softmax(logits, dim=1)

    targets = F.one_hot(
        labels,
        num_classes=logits.shape[1],
    ).float()

    score = torch.sum(
        (probabilities - targets) ** 2,
        dim=1,
    ).mean()

    return float(score.item())


def expected_calibration_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Predictions are divided into confidence bins.
    For each bin, the difference between confidence and
    accuracy is weighted by the fraction of samples in that bin.
    """
    probabilities = F.softmax(logits, dim=1)

    confidences, predictions = probabilities.max(dim=1)

    correct = predictions.eq(labels)

    bin_boundaries = torch.linspace(
        0.0,
        1.0,
        num_bins + 1,
        device=logits.device,
    )

    ece = torch.tensor(
        0.0,
        device=logits.device,
    )

    for i in range(num_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            in_bin = (confidences >= lower) & (
                confidences <= upper
            )
        else:
            in_bin = (confidences >= lower) & (
                confidences < upper
            )

        if not in_bin.any():
            continue

        bin_accuracy = correct[in_bin].float().mean()
        bin_confidence = confidences[in_bin].mean()
        bin_fraction = in_bin.float().mean()

        ece += torch.abs(
            bin_confidence - bin_accuracy
        ) * bin_fraction

    return float(ece.item())


def calibration_bins(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10,
):
    """
    Return confidence-bin statistics for reliability diagrams.

    Returns:
        Dictionary containing:
        - bin_confidence
        - bin_accuracy
        - bin_count
    """
    probabilities = F.softmax(logits, dim=1)

    confidences, predictions = probabilities.max(dim=1)

    correct = predictions.eq(labels)

    bin_boundaries = torch.linspace(
        0.0,
        1.0,
        num_bins + 1,
        device=logits.device,
    )

    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for i in range(num_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            in_bin = (confidences >= lower) & (
                confidences <= upper
            )
        else:
            in_bin = (confidences >= lower) & (
                confidences < upper
            )

        count = int(in_bin.sum().item())

        bin_counts.append(count)

        if count == 0:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)
        else:
            bin_confidences.append(
                float(confidences[in_bin].mean().item())
            )

            bin_accuracies.append(
                float(correct[in_bin].float().mean().item())
            )

    return {
        "bin_confidence": bin_confidences,
        "bin_accuracy": bin_accuracies,
        "bin_count": bin_counts,
    }