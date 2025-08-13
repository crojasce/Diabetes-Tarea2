from typing import List, Tuple
import numpy as np
import pandas as pd

def split_num_cat(df: pd.DataFrame, exclude: List[str] | None = None) -> Tuple[List[str], List[str]]:
    """Separa columnas numéricas y categóricas."""
    exclude = set(exclude or [])
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    return num_cols, cat_cols

def cumulative_k_from_scores(scores: np.ndarray, threshold: float = 0.8) -> int:
    """Dado un vector de scores, elige K mínimo que acumule 'threshold' de importancia."""
    s = np.maximum(scores, 0)
    if s.sum() == 0:
        return max(1, min(10, len(scores)))
    w = s / s.sum()
    cum = np.cumsum(np.sort(w)[::-1])
    k = int(np.searchsorted(cum, threshold) + 1)
    return max(1, min(k, len(scores)))

def compute_k_from_evr(evr: np.ndarray, var_threshold: float) -> int:
    """K mínimo tal que varianza explicada acumulada ≥ var_threshold."""
    if evr is None or len(evr) == 0:
        return 0
    cum = np.cumsum(evr)
    return int(np.searchsorted(cum, var_threshold) + 1)
