import numpy as np
import torch


def compute_descriptor(
    hidden_states: torch.Tensor,
    probe_outputs: torch.Tensor,
    r: int
):
    """
    Compute task descriptor z_t.

    z_t = Concat(
        mean(hidden_states),
        std(hidden_states),
        order statistics (quantiles),
        top-r projection of probe outputs
    )

    Args:
        hidden_states: Tensor [N, H]
        probe_outputs: Tensor [N, D]
        r: int, projection rank

    Returns:
        z_t: Tensor [descriptor_dim]
    """
    with torch.no_grad():
        mu = hidden_states.mean(dim=0)
        sigma = hidden_states.std(dim=0)

        q25 = torch.quantile(hidden_states, 0.25, dim=0)
        q50 = torch.quantile(hidden_states, 0.50, dim=0)
        q75 = torch.quantile(hidden_states, 0.75, dim=0)

        U, S, Vh = torch.linalg.svd(probe_outputs, full_matrices=False)
        proj = (U[:, :r] @ torch.diag(S[:r])).mean(dim=0)

        z_t = torch.cat([mu, sigma, q25, q50, q75, proj], dim=0)

    return z_t
