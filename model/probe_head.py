import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """
    Deterministic linear probe used for task-specific prediction.
    This module corresponds to f_{ψ_p} and is trained or fixed
    depending on the experimental protocol.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, input_dim]

        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        return self.linear(x)
