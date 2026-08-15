import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionRecalibrator(nn.Module):
    """
    Distribution recalibrator for multiclass classification.

    The recalibrator receives the pretrained model's probability
    distribution and learns a correction to its logits.

    The correction is initialized to zero, so the initial
    recalibrator behaves like the identity mapping.

    This makes the experiment a conservative recalibration step
    rather than an independently initialized classifier.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.input_layer = nn.Linear(
            num_classes,
            hidden_dim,
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            num_classes,
        )

        self.network = nn.Sequential(
            self.input_layer,
            nn.ReLU(),
            self.output_layer,
        )

        # Start with zero correction.
        #
        # This means:
        #
        # calibrated_logits = original_logits + 0
        #
        # at initialization.
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a probability distribution into a recalibrated
        probability distribution.

        The original logits are recovered from log probabilities.
        A learned residual correction is then added.
        """

        probabilities = probabilities.clamp_min(1e-8)

        original_logits = torch.log(probabilities)

        correction = self.network(probabilities)

        calibrated_logits = (
            original_logits + correction
        )

        return F.softmax(
            calibrated_logits,
            dim=1,
        )