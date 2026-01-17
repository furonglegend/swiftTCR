import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def expected_calibration_error(probs, labels, n_bins=10):
    """
    Compute ECE.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += abs(acc - conf) * mask.mean()

    return ece


def evaluate(preds, probs, labels):
    """
    Compute standard metrics.
    """
    auc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds)
    ece = expected_calibration_error(probs, labels)

    return {
        "AUC": auc,
        "F1": f1,
        "ECE": ece
    }
