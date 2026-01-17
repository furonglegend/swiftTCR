import numpy as np
from sklearn.decomposition import PCA


def build_theta_matrix(theta_list):
    """
    Stack per-task adapter vectors into a single matrix.

    Args:
        theta_list: list[np.ndarray], each of shape [d]

    Returns:
        Theta: array of shape [num_tasks, d]
    """
    return np.stack(theta_list, axis=0)


def svd_select_rank(Theta, ratio_threshold: float):
    """
    Select intrinsic rank r using singular value energy ratio.

    Args:
        Theta: array of shape [T, d]
        ratio_threshold: float in (0, 1)

    Returns:
        r: selected rank
        singular_values: array
    """
    _, s, _ = np.linalg.svd(Theta, full_matrices=False)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    r = int(np.searchsorted(energy, ratio_threshold) + 1)
    return r, s


def fisher_eigen_analysis(Theta, labels):
    """
    Perform Fisher-style eigen analysis between tasks.

    Args:
        Theta: array of shape [T, d]
        labels: array of shape [T], task group identifiers

    Returns:
        eigenvalues: array
        eigenvectors: array
    """
    classes = np.unique(labels)
    mean_global = Theta.mean(axis=0)

    Sb = np.zeros((Theta.shape[1], Theta.shape[1]))
    Sw = np.zeros_like(Sb)

    for c in classes:
        group = Theta[labels == c]
        mean_c = group.mean(axis=0)
        diff = (mean_c - mean_global).reshape(-1, 1)
        Sb += group.shape[0] * (diff @ diff.T)

        centered = group - mean_c
        Sw += centered.T @ centered

    eigvals, eigvecs = np.linalg.eig(
        np.linalg.pinv(Sw) @ Sb
    )

    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx].real, eigvecs[:, idx].real


def bootstrap_svd_rank(Theta, ratio_threshold, n_bootstrap=500, seed=0):
    """
    Bootstrap confidence intervals for rank selection.

    Args:
        Theta: array [T, d]
        ratio_threshold: float
        n_bootstrap: int

    Returns:
        ranks: list[int]
    """
    rng = np.random.default_rng(seed)
    ranks = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(Theta), size=len(Theta), replace=True)
        Theta_b = Theta[idx]
        r, _ = svd_select_rank(Theta_b, ratio_threshold)
        ranks.append(r)

    return ranks
