import torch
import torch.nn as nn


class DistributionRecalibrator(nn.Module):
    """
    Distribution recalibrator based on Kuleshov et al.

    Maps a K-dimensional probability vector produced by a
    pretrained classifier to a recalibrated K-dimensional
    probability vector.

    The architecture is configurable so that different
    recalibrator
    capacities can be compared experimentally.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (64, 64),
    ):
        super().__init__()

        layers = []

        input_dim = num_classes

        for hidden_dim in hidden_dims:
            layers.append(
                nn.Linear(
                    input_dim,
                    hidden_dim,
                )
            )

            layers.append(nn.ReLU())

            input_dim = hidden_dim

        layers.append(
            nn.Linear(
                input_dim,
                num_classes,
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recalibrate a batch of probability distributions.

        Args:
            probabilities:
                Tensor of shape [batch_size, num_classes].

        Returns:
            Recalibrated probability distributions with
            rows summing to one.
        """

        logits = self.network(probabilities)

        return torch.softmax(
            logits,
            dim=1,
        )