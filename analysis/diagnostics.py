import numpy as np


def spectrum(M):
    """
    Compute singular value spectrum of prototype matrix.

    Args:
        M: array [K, d]

    Returns:
        singular values
    """
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    return s


def mutual_coherence(M):
    """
    Compute mutual coherence μ(M).

    Args:
        M: array [K, d]

    Returns:
        float
    """
    M_norm = M / np.linalg.norm(M, axis=1, keepdims=True)
    G = M_norm @ M_norm.T
    np.fill_diagonal(G, 0.0)
    return np.max(np.abs(G))


def condition_number(M):
    """
    Compute condition number κ(M^T).

    Args:
        M: array [K, d]

    Returns:
        float
    """
    s = spectrum(M)
    return s.max() / s.min()
