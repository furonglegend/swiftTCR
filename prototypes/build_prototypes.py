import numpy as np
from sklearn.cluster import KMeans
from adapters.canonicalize import batch_canonicalize


def build_prototypes(theta_matrix, num_prototypes, seed=0):
    """
    Construct geometry-aware prototypes via clustering.

    Args:
        theta_matrix: array [T, d]
        num_prototypes: int

    Returns:
        M: array [K, d]
    """
    kmeans = KMeans(
        n_clusters=num_prototypes,
        random_state=seed,
        n_init=10
    )
    kmeans.fit(theta_matrix)
    centers = kmeans.cluster_centers_

    centers = batch_canonicalize(centers)
    return centers
