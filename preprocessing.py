from __future__ import annotations
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction import FeatureHasher

def _make_ohe():
    # One-Hot es PARSO y float32 para memoria baja
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)
    except TypeError:  # compat scikit viejas
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def _sanitize_cat(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    # Reemplaza marcadores tipo "?" por NaN para que el imputador funcione bien
    Xc = df[cols].copy()
    for c in cols:
        Xc[c] = Xc[c].replace({"?": np.nan})
    return Xc

def apply_preprocessor(
    X: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    high_card_threshold: int = 100,
    hash_features: int = 64,
) -> Tuple[np.ndarray, "sparse.spmatrix", List[str], Dict[str, int], object, Dict[str, object]]:
    """
    - Numéricas: imputación (mediana) + estandarización (float32)
    - Categóricas baja cardinalidad: imputación (más frecuente) + OHE esparso
    - Categóricas alta cardinalidad: imputación + hashing esparso (FeatureHasher)

    Returns:
      X_num_scaled (np.ndarray, float32),
      X_cat_combined (scipy.sparse),
      low_card_feature_names (list),
      hashed_meta (dict {col: n_features}),
      scaler,
      encoders (dict: { 'ohe': OneHotEncoder or None, 'hashers': {col: FeatureHasher} })
    """
    # ======= Numéricas =======
    num_imputer = SimpleImputer(strategy="median")
    X_num = num_imputer.fit_transform(X[num_cols]).astype(np.float32) if num_cols else np.empty((len(X), 0), dtype=np.float32)
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num) if num_cols else np.empty((len(X), 0), dtype=np.float32)
    if hasattr(X_num_scaled, "astype"):
        X_num_scaled = X_num_scaled.astype(np.float32)

    # ======= Categóricas =======
    if not cat_cols:
        return X_num_scaled, sparse.csr_matrix((len(X), 0), dtype=np.float32), [], {}, scaler, {"ohe": None, "hashers": {}}

    X_cat_raw = _sanitize_cat(X, cat_cols)

    # Detectar cardinalidad
    nunique = {c: X_cat_raw[c].nunique(dropna=True) for c in cat_cols}
    low_cols = [c for c in cat_cols if nunique[c] <= high_card_threshold]
    high_cols = [c for c in cat_cols if nunique[c] > high_card_threshold]

    # --- Baja cardinalidad -> OHE esparso ---
    ohe = None
    X_low = sparse.csr_matrix((len(X), 0), dtype=np.float32)
    low_names: List[str] = []
    if low_cols:
        low_imputer = SimpleImputer(strategy="most_frequent")
        X_low_imp = low_imputer.fit_transform(X_cat_raw[low_cols])
        ohe = _make_ohe()
        X_low = ohe.fit_transform(X_low_imp)
        try:
            low_names = ohe.get_feature_names_out(low_cols).tolist()
        except Exception:
            low_names = [f"low_{i}" for i in range(X_low.shape[1])]

    # --- Alta cardinalidad -> hashing por columna (esparso) ---
    hashers = {}
    hashed_blocks = []
    hashed_meta: Dict[str, int] = {}
    if high_cols:
        high_imputer = SimpleImputer(strategy="most_frequent")
        X_high_imp = pd.DataFrame(high_imputer.fit_transform(X_cat_raw[high_cols]), columns=high_cols, index=X.index)
        for col in high_cols:
            tok = X_high_imp[col].astype(str)
            # Un token por muestra: "col=valor"
            tokens = [[f"{col}={v}"] for v in tok]
            fh = FeatureHasher(n_features=int(hash_features), input_type="string", alternate_sign=False)
            mat = fh.transform(tokens)  # (n_samples, hash_features)
            hashed_blocks.append(mat)
            hashers[col] = fh
            hashed_meta[col] = int(hash_features)

    X_high = sparse.hstack(hashed_blocks).tocsr() if hashed_blocks else sparse.csr_matrix((len(X), 0), dtype=np.float32)

    # Combinar categóricas
    X_cat_combined = sparse.hstack([X_low, X_high]).tocsr() if (X_low.shape[1] or X_high.shape[1]) else sparse.csr_matrix((len(X), 0), dtype=np.float32)

    encoders = {"ohe": ohe, "hashers": hashers}
    return X_num_scaled, X_cat_combined, low_names, hashed_meta, scaler, encoders

