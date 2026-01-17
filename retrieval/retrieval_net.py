import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalNet(nn.Module):
    """
    G_φ: maps task descriptor z_t → prototype logits v_t
    """

    def __init__(self, input_dim, num_prototypes, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_prototypes)
        )

    def forward(self, z_t):
        """
        Args:
            z_t: Tensor [B, D]

        Returns:
            logits: Tensor [B, K]
        """
        return self.net(z_t)
