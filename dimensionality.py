from typing import Tuple, List
import numpy as np

def pca_reduce(X: np.ndarray, var_threshold: float = 0.9) -> Tuple[np.ndarray, int, np.ndarray]:
    if X.shape[1] == 0:
        return np.empty((X.shape[0], 0)), 0, np.array([])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=None, svd_solver='full', random_state=0)
    X_pca = pca.fit_transform(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, var_threshold) + 1)
    return X_pca[:, :k], k, pca.explained_variance_ratio_

def mca_reduce(X_ohe: np.ndarray, var_threshold: float = 0.9) -> Tuple[np.ndarray, int, np.ndarray, str]:
    if X_ohe.shape[1] == 0:
        return np.empty((X_ohe.shape[0], 0)), 0, np.array([]), "none"
    # Intenta prince.MCA; si no está disponible, usa TruncatedSVD como aproximación
    try:
        import prince
        mca = prince.MCA(n_components=min(X_ohe.shape[1]-1, 50), random_state=0)
        X_mca = mca.fit(X_ohe).transform(X_ohe).values
        if hasattr(mca, 'eigenvalues_'):
            ev = np.array(mca.eigenvalues_)
            evr = ev / ev.sum() if ev.sum() > 0 else np.ones_like(ev)/len(ev)
        else:
            evr = np.ones(X_mca.shape[1]) / X_mca.shape[1]
        cum = np.cumsum(evr)
        k = int(np.searchsorted(cum, var_threshold) + 1)
        method = "prince.MCA"
        return X_mca[:, :k], k, evr, method
    except Exception:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=min(X_ohe.shape[1]-1, 50), random_state=0)
        X_svd = svd.fit_transform(X_ohe)
        evr = svd.explained_variance_ratio_
        cum = np.cumsum(evr)
        k = int(np.searchsorted(cum, var_threshold) + 1)
        method = "TruncatedSVD (MCA aprox)"
        return X_svd[:, :k], k, evr, method
