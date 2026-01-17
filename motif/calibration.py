import numpy as np
from scipy.stats import ttest_1samp


def nested_cv_threshold(scores, labels, folds=5):
    """
    Nested CV to estimate calibration threshold τ_c.
    """
    scores = np.array(scores)
    labels = np.array(labels)

    thresholds = []
    idx = np.arange(len(scores))
    np.random.shuffle(idx)

    splits = np.array_split(idx, folds)
    for i in range(folds):
        val_idx = splits[i]
        train_idx = np.hstack([splits[j] for j in range(folds) if j != i])

        pos_scores = scores[train_idx][labels[train_idx] == 1]
        thresholds.append(pos_scores.mean())

    return np.mean(thresholds)


def calibration_ttest(scores, tau_c):
    """
    One-sample t-test for calibration significance.
    """
    stat, pval = ttest_1samp(scores, tau_c)
    return stat, pval
