import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_prototype_heatmap(
    M,
    normalize=True,
    save_path=None
):
    """
    Plot prototype matrix heatmap.

    Args:
        M: prototype matrix [K, D]
    """
    X = M.copy()
    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    plt.figure(figsize=(8, 4))
    sns.heatmap(X, cmap="viridis", cbar=True)
    plt.xlabel("Feature dimension")
    plt.ylabel("Prototype index")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
