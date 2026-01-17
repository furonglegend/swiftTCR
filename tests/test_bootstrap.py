import numpy as np

from utils.metrics import bootstrap_ci, compute_auc


def test_bootstrap_ci_bounds():
    """
    Bootstrap CI should contain the point estimate.
    """
    np.random.seed(0)

    y_true = np.random.randint(0, 2, size=200)
    y_score = np.random.rand(200)

    mean, (lo, hi) = bootstrap_ci(
        compute_auc,
        y_true,
        y_score,
        n_boot=200
    )

    assert lo <= mean <= hi, "Mean not inside CI bounds"


def test_bootstrap_ci_width():
    """
    CI width should shrink with more samples.
    """
    np.random.seed(1)

    y_true = np.random.randint(0, 2, size=1000)
    y_score = np.random.rand(1000)

    _, (lo, hi) = bootstrap_ci(
        compute_auc,
        y_true,
        y_score,
        n_boot=300
    )

    assert (hi - lo) < 0.5, "CI too wide for large sample size"
