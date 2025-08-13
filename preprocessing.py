from __future__ import annotations
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def _make_ohe():
    # Esparso para memoria baja; dtype float32
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)
    except TypeError:  # compatibilidad vieja
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def apply_preprocessor(
    X: pd.DataFrame, num_cols: List[str], cat_cols: List[str]
) -> tuple[np.ndarray, "scipy.sparse.spmatrix", List[str], List[str], object, object]:
    """Imputa + escala numéricas (float32) y codifica categóricas con OHE esparso."""
    # Numéricas
    num_imputer = SimpleImputer(strategy="median")
    X_num = num_imputer.fit_transform(X[num_cols]).astype(np.float32) if num_cols else np.empty((len(X), 0), dtype=np.float32)
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num) if num_cols else np.empty((len(X), 0), dtype=np.float32)
    if hasattr(X_num_scaled, "astype"):  # dense
        X_num_scaled = X_num_scaled.astype(np.float32)

    # Categóricas
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_cat = cat_imputer.fit_transform(X[cat_cols]) if cat_cols else np.empty((len(X), 0))
    ohe = _make_ohe()
    X_cat_ohe = ohe.fit_transform(X_cat) if cat_cols else None

    # Nombres
    num_feature_names = num_cols
    if cat_cols:
        try:
            cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
        except Exception:
            cat_feature_names = [f"cat_{i}" for i in range(X_cat_ohe.shape[1])]
    else:
        cat_feature_names = []

    return X_num_scaled, X_cat_ohe, num_feature_names, cat_feature_names, scaler, ohe
