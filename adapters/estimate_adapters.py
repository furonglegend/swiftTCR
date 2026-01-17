import numpy as np


def estimate_task_adapter(
    features: np.ndarray,
    targets: np.ndarray,
    lambda_reg: float
):
    """
    Estimate per-task adapter parameters using ridge regression.

    Args:
        features: array of shape [n_samples, d]
        targets: array of shape [n_samples, d]
        lambda_reg: ridge regularization coefficient

    Returns:
        theta_hat: array of shape [d]
    """
    d = features.shape[1]
    xtx = features.T @ features
    reg = lambda_reg * np.eye(d)

    theta_hat = np.linalg.solve(
        xtx + reg,
        features.T @ targets
    )

    return theta_hat
