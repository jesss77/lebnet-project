import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """
    Temperature scaling for neural-network calibration.

    A single scalar temperature is learned on a held-out
    calibration set. The model's logits are divided by
    this temperature before applying softmax.

    T > 1:
        Produces softer / less confident probabilities.

    T < 1:
        Produces sharper / more confident probabilities.
    """

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()

        if initial_temperature <= 0:
            raise ValueError(
                "initial_temperature must be greater than 0."
            )

        self.temperature = nn.Parameter(
            torch.tensor(
                [initial_temperature],
                dtype=torch.float32,
            )
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        """

        temperature = self.temperature.clamp_min(1e-6)

        return logits / temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 100,
    ):
        """
        Learn the temperature by minimizing NLL
        on the calibration set.
        """

        if logits.ndim != 2:
            raise ValueError(
                "logits must have shape [N, num_classes]."
            )

        if labels.ndim != 1:
            raise ValueError(
                "labels must have shape [N]."
            )

        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                "logits and labels must contain the same "
                "number of samples."
            )

        # Keep calibration data on the same device
        # as the temperature parameter.
        device = self.temperature.device

        logits = logits.to(device)
        labels = labels.to(device)

        criterion = nn.CrossEntropyLoss()

        # LBFGS works well for optimizing the single
        # temperature parameter.
        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=0.01,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()

            scaled_logits = self.forward(logits)

            loss = criterion(
                scaled_logits,
                labels,
            )

            loss.backward()

            return loss

        optimizer.step(closure)

        # Ensure temperature remains positive.
        with torch.no_grad():
            self.temperature.clamp_(min=1e-6)

        return self

    def get_temperature(self) -> float:
        """
        Return the learned temperature as a Python float.
        """

        return self.temperature.item()


def apply_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Apply a fixed temperature to logits.
    """

    if temperature <= 0:
        raise ValueError(
            "temperature must be greater than 0."
        )

    return logits / temperature