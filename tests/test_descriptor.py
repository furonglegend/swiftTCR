import torch
import numpy as np

from descriptors.task_descriptor import compute_task_descriptor


def test_descriptor_dimension():
    """
    Ensure descriptor has the expected dimensionality.
    """
    torch.manual_seed(0)

    H = torch.randn(100, 64)   # hidden states
    gp = torch.randn(64)       # probe vector

    z = compute_task_descriptor(
        hidden_states=H,
        probe_vector=gp,
        order_k=4,
        proj_rank=8
    )

    assert z.ndim == 1, "Descriptor must be a vector"
    assert z.numel() > 0, "Descriptor is empty"


def test_descriptor_determinism():
    """
    Descriptor computation should be deterministic.
    """
    torch.manual_seed(42)

    H = torch.randn(80, 32)
    gp = torch.randn(32)

    z1 = compute_task_descriptor(H, gp)
    z2 = compute_task_descriptor(H, gp)

    assert torch.allclose(z1, z2), "Descriptor is not deterministic"
