import numpy as np


def coverage_error(theta, M):
    """
    Compute distance to closest prototype span.

    Args:
        theta: array [d]
        M: array [K, d]

    Returns:
        float
    """
    coeffs, _, _, _ = np.linalg.lstsq(M.T, theta, rcond=None)
    recon = M.T @ coeffs
    return np.linalg.norm(theta - recon)


def estimate_coverage(theta_matrix, M):
    """
    Estimate empirical prototype coverage error.

    Args:
        theta_matrix: array [T, d]
        M: array [K, d]

    Returns:
        mean_error: float
        errors: array
    """
    errors = np.array([coverage_error(t, M) for t in theta_matrix])
    return errors.mean(), errors


def bootstrap_coverage(theta_matrix, M, n_bootstrap=500, seed=0):
    """
    Bootstrap confidence interval for coverage error.

    Returns:
        upper_90: float
    """
    rng = np.random.default_rng(seed)
    means = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(theta_matrix), size=len(theta_matrix), replace=True)
        mean_err, _ = estimate_coverage(theta_matrix[idx], M)
        means.append(mean_err)

    return np.percentile(means, 90)
