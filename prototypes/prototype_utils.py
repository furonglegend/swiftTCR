import numpy as np


def normalize_prototypes(M):
    """
    Normalize prototype rows to unit norm.

    Args:
        M: array [K, d]

    Returns:
        normalized M
    """
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def merge_close_prototypes(M, threshold):
    """
    Merge prototypes with cosine similarity above threshold.

    Args:
        M: array [K, d]
        threshold: float in (0, 1)

    Returns:
        reduced M
    """
    M_norm = normalize_prototypes(M)
    keep = []

    for i in range(len(M)):
        if all(np.dot(M_norm[i], M_norm[j]) < threshold for j in keep):
            keep.append(i)

    return M[keep]
