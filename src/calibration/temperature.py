import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """
    Temperature scaling for neural network calibration.

    A learned temperature is applied to model logits before converting
    them into probabilities. The temperature is fitted using the
    calibration split and must not be fitted on the test set.
    """

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()

        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive.")

        self.temperature = nn.Parameter(
            torch.tensor(float(initial_temperature))
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        """
        temperature = torch.clamp(
            self.temperature,
            min=1e-6,
        )

        return logits / temperature

    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the learned temperature to logits.

        This is an explicit alias for forward() so that the scaler can
        be used with either scaler(logits) or scaler.transform(logits).
        """
        return self.forward(logits)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
    ) -> float:
        """
        Learn the temperature by minimizing cross-entropy loss on
        calibration predictions.

        Args:
            logits: Model logits from the calibration set.
            labels: Ground-truth calibration labels.
            max_iter: Maximum number of LBFGS iterations.

        Returns:
            Learned temperature as a Python float.
        """
        logits = logits.detach()
        labels = labels.detach()

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=0.01,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()

            temperature = torch.clamp(
                self.temperature,
                min=1e-6,
            )

            scaled_logits = logits / temperature

            loss = criterion(
                scaled_logits,
                labels,
            )

            loss.backward()

            return loss

        optimizer.step(closure)

        with torch.no_grad():
            self.temperature.clamp_(min=1e-6)

        return float(self.temperature.detach().item())

    def get_temperature(self) -> float:
        """
        Return the learned temperature as a Python float.
        """
        return float(self.temperature.detach().item())