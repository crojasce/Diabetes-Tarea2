from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def _make_ohe():
    # scikit-learn 1.2+ usa 'sparse_output'; versiones previas usan 'sparse'
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

def apply_preprocessor(X: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], object, object]:
    # Numéricas: imputación (mediana) + estandarización
    num_imputer = SimpleImputer(strategy='median')
    X_num = num_imputer.fit_transform(X[num_cols]) if num_cols else np.empty((len(X),0))
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num) if num_cols else np.empty((len(X),0))

    # Categóricas: imputación (más frecuente) + one-hot
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_cat = cat_imputer.fit_transform(X[cat_cols]) if cat_cols else np.empty((len(X),0))
    ohe = _make_ohe()
    X_cat_ohe = ohe.fit_transform(X_cat) if cat_cols else np.empty((len(X),0))

    # Nombres de features
    num_feature_names = num_cols
    if cat_cols:
        try:
            ohe_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
        except Exception:
            ohe_feature_names = [f"cat_{i}" for i in range(X_cat_ohe.shape[1])]
    else:
        ohe_feature_names = []

    return X_num_scaled, X_cat_ohe, num_feature_names, ohe_feature_names, scaler, ohe
