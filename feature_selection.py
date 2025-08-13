from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.feature_selection import f_classif, chi2
from utils import cumulative_k_from_scores

def select_numeric_features(X_num: np.ndarray, y, threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """ANOVA F (clasificación). Devuelve (idx_seleccionados, f_scores)."""
    if X_num is None or X_num.shape[1] == 0:
        return np.array([], dtype=int), np.array([])
    f_scores, _ = f_classif(X_num, y)
    k = cumulative_k_from_scores(f_scores, threshold=threshold)
    selected_idx = np.argsort(f_scores)[::-1][:k]
    return selected_idx, f_scores

def select_categorical_features(X_cat_ohe, y, threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Chi² sobre OHE (acepta esparso). Devuelve (idx_seleccionados, chi2_scores)."""
    if X_cat_ohe is None or X_cat_ohe.shape[1] == 0:
        return np.array([], dtype=int), np.array([])
    scores, _ = chi2(X_cat_ohe, y)  # requiere no-negativos; OHE cumple
    k = cumulative_k_from_scores(scores, threshold=threshold)
    selected_idx = np.argsort(scores)[::-1][:k]
    return selected_idx, scores
