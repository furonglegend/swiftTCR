import torch
import numpy as np

from retrieval.proximal_solver import solve_proximal


def test_top_r_sparsity():
    """
    Verify that the proximal solver enforces top-r sparsity.
    """
    torch.manual_seed(0)

    d = 50
    r = 5

    v = torch.randn(d)
    M = torch.eye(d)

    w = solve_proximal(
        logits=v,
        prototype_matrix=M,
        r=r,
        lmbda=0.1,
        max_iter=50
    )

    non_zero = (w.abs() > 1e-6).sum().item()
    assert non_zero <= r, "Proximal solver violates sparsity constraint"


def test_energy_decrease():
    """
    Check that the objective value decreases across iterations.
    """
    torch.manual_seed(1)

    d = 30
    r = 4
    v = torch.randn(d)
    M = torch.eye(d)

    history = []

    def hook(obj):
        history.append(obj)

    _ = solve_proximal(
        logits=v,
        prototype_matrix=M,
        r=r,
        lmbda=0.05,
        max_iter=40,
        callback=hook
    )

    assert all(
        history[i] >= history[i + 1] - 1e-6
        for i in range(len(history) - 1)
    ), "Objective is not non-increasing"
