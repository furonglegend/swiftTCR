import numpy as np


def estimate_pi0(pvals, lambda_thr=0.5):
    """
    Storey estimator for pi0.
    """
    pvals = np.asarray(pvals)
    return np.mean(pvals > lambda_thr) / (1 - lambda_thr)


def bootstrap_pi0(pvals, n_boot=1000):
    """
    Bootstrap confidence intervals for pi0.
    """
    pvals = np.asarray(pvals)
    estimates = []

    for _ in range(n_boot):
        sample = np.random.choice(pvals, size=len(pvals), replace=True)
        estimates.append(estimate_pi0(sample))

    return np.mean(estimates), np.percentile(estimates, [2.5, 97.5])
