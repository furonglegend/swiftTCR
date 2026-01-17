import torch
import torch.nn as nn


class AdapterModule(nn.Module):
    """
    Lightweight adapter module with low-rank structure.
    This module is applied additively to frozen encoder features.
    """

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)

        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, dim]

        Returns:
            Tensor of shape [batch_size, dim]
        """
        return x + self.up(self.down(x))
