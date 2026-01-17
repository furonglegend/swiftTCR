import numpy as np


def canonicalize_adapter(theta: np.ndarray, eps: float = 1e-8):
    """
    Canonicalize adapter parameters to enforce scale and
    sign invariance across equivalent representations.

    Args:
        theta: array of shape [d]
        eps: numerical stability constant

    Returns:
        theta_canonical: array of shape [d]
    """
    norm = np.linalg.norm(theta)
    if norm < eps:
        return theta

    theta = theta / norm

    # Fix global sign ambiguity
    if theta[np.argmax(np.abs(theta))] < 0:
        theta = -theta

    return theta


def batch_canonicalize(thetas: np.ndarray):
    """
    Apply canonicalization to a batch of adapters.

    Args:
        thetas: array of shape [num_tasks, d]

    Returns:
        array of shape [num_tasks, d]
    """
    return np.stack([canonicalize_adapter(t) for t in thetas], axis=0)
