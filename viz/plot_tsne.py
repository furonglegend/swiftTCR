import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_embedding(
    X,
    labels=None,
    method="tsne",
    n_components=2,
    perplexity=30,
    save_path=None
):
    """
    Visualize embeddings using t-SNE or PCA.

    Args:
        X: array-like, shape [N, D]
        labels: optional class labels
        method: 'tsne' or 'pca'
    """
    if method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            init="pca",
            random_state=42
        )
        Z = reducer.fit_transform(X)
    elif method == "pca":
        reducer = PCA(n_components=n_components)
        Z = reducer.fit_transform(X)
    else:
        raise ValueError("Unknown method")

    plt.figure(figsize=(6, 6))
    if labels is None:
        plt.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.7)
    else:
        for lab in np.unique(labels):
            idx = labels == lab
            plt.scatter(Z[idx, 0], Z[idx, 1], s=8, alpha=0.7, label=str(lab))
        plt.legend(frameon=False)

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
