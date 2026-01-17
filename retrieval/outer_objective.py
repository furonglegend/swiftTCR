import torch
import torch.nn.functional as F


def entropy_penalty(a, eps=1e-8):
    """
    Encourage confident sparse selections.
    """
    p = F.softmax(a, dim=-1)
    return -(p * torch.log(p + eps)).sum()


def sparsity_penalty(a):
    """
    L1 proxy for sparsity.
    """
    return torch.norm(a, p=1)


def outer_loss(
    theta_hat,
    theta_true,
    a,
    lambda_entropy,
    lambda_sparse
):
    """
    Outer objective (Eq. outer).

    Returns:
        scalar loss
    """
    recon = F.mse_loss(theta_hat, theta_true)
    ent = entropy_penalty(a)
    sparse = sparsity_penalty(a)

    return recon + lambda_entropy * ent + lambda_sparse * sparse
