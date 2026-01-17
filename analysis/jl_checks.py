import numpy as np


def random_jl_projection(d_in, d_out, seed=0):
    """
    Generate a Gaussian Johnson–Lindenstrauss projection.

    Args:
        d_in: original dimension
        d_out: projected dimension

    Returns:
        projection matrix of shape [d_out, d_in]
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1.0 / np.sqrt(d_out), size=(d_out, d_in))


def jl_distance_preservation(X, proj):
    """
    Evaluate pairwise distance distortion after JL projection.

    Args:
        X: array [n, d_in]
        proj: array [d_out, d_in]

    Returns:
        distortion: float
    """
    X_proj = X @ proj.T

    d_orig = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
    d_proj = np.linalg.norm(X_proj[:, None] - X_proj[None, :], axis=-1)

    mask = d_orig > 0
    ratio = d_proj[mask] / d_orig[mask]

    return np.max(np.abs(ratio - 1.0))
