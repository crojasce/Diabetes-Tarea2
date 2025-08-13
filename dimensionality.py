from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from utils import compute_k_from_evr

def pca_reduce(X: np.ndarray, var_threshold: float = 0.9):
    """PCA para numéricas (devuelve X_reducida, k, evr)."""
    if X is None or X.shape[1] == 0:
        return np.empty((X.shape[0], 0)), 0, np.array([])
    pca = PCA(n_components=None, svd_solver="full", random_state=0)
    X_pca = pca.fit_transform(X)
    k = compute_k_from_evr(pca.explained_variance_ratio_, var_threshold)
    return X_pca[:, :k], k, pca.explained_variance_ratio_

def mca_reduce(X_ohe, var_threshold: float = 0.9, max_components: int = 50):
    """
    Aproximación MCA con TruncatedSVD (funciona con matrices esparsas).
    Devuelve X_reducida, k, evr, método.
    """
    import numpy as np
    if X_ohe is None or X_ohe.shape[1] == 0:
        return np.empty((X_ohe.shape[0], 0)), 0, np.array([]), "none"
    n_comp = min(max_components, max(1, X_ohe.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=0)
    X_svd = svd.fit_transform(X_ohe)
    evr = svd.explained_variance_ratio_
    k = compute_k_from_evr(evr, var_threshold)
    return X_svd[:, :k], k, evr, "TruncatedSVD (MCA aprox)"

