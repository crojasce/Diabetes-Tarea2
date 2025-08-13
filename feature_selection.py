from typing import Tuple
import numpy as np
from sklearn.feature_selection import f_classif, chi2
from utils import cumulative_k_from_scores
# ... (resto igual)


def select_numeric_features(X_num: np.ndarray, y, threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    if X_num.shape[1] == 0:
        return np.array([], dtype=int), np.array([])
    f_scores, _ = f_classif(X_num, y)
    k = cumulative_k_from_scores(f_scores, threshold=threshold)
    idx_sorted = np.argsort(f_scores)[::-1]
    selected_idx = idx_sorted[:k]
    return selected_idx, f_scores

def select_categorical_features(X_cat_ohe: np.ndarray, y, threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    if X_cat_ohe.shape[1] == 0:
        return np.array([], dtype=int), np.array([])
    # chi2 requiere no-negativos; OHE cumple
    scores, _ = chi2(X_cat_ohe, y)
    k = cumulative_k_from_scores(scores, threshold=threshold)
    idx_sorted = np.argsort(scores)[::-1]
    selected_idx = idx_sorted[:k]
    return selected_idx, scores
