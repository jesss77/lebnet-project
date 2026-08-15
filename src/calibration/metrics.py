import torch
import torch.nn.functional as F


def expected_calibration_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between model confidence
    and empirical accuracy across confidence bins.
    """

    probabilities = F.softmax(logits, dim=1)
    confidences, predictions = probabilities.max(dim=1)

    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(
        0.0,
        1.0,
        n_bins + 1,
        device=logits.device,
    )

    ece = torch.zeros(1, device=logits.device)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        if mask.any():
            bin_accuracy = accuracies[mask].float().mean()
            bin_confidence = confidences[mask].mean()

            bin_fraction = mask.float().mean()

            ece += torch.abs(
                bin_accuracy - bin_confidence
            ) * bin_fraction

    return ece.item()


def negative_log_likelihood(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute multiclass negative log-likelihood."""

    return F.cross_entropy(logits, labels).item()


def brier_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute multiclass Brier score."""

    probabilities = F.softmax(logits, dim=1)

    targets = F.one_hot(
        labels,
        num_classes=logits.shape[1],
    ).float()

    return torch.mean(
        torch.sum((probabilities - targets) ** 2, dim=1)
    ).item()


def accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute classification accuracy."""

    predictions = logits.argmax(dim=1)

    return (
        predictions.eq(labels)
        .float()
        .mean()
        .item()
    )