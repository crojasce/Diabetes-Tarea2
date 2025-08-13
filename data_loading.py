from __future__ import annotations
import pandas as pd

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce uso de memoria: float64→float32, int64→int32/int16, object→category (cuando conviene)."""
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    # textos con cardinalidad relativamente baja -> category
    limit = max(200, int(0.02 * len(df)))  # heurística
    for c in df.select_dtypes(include=["object"]).columns:
        nun = df[c].nunique(dropna=True)
        if 0 < nun <= limit:
            df[c] = df[c].astype("category")
    return df

def read_csv_smart(path_or_buffer) -> pd.DataFrame:
    """Lectura tolerante y optimizada."""
    df = pd.read_csv(path_or_buffer, low_memory=False)
    df = optimize_dtypes(df)
    return df

def merge_by_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_key: str,
    right_key: str,
    how: str = "left",
    coerce_to_str: bool = True,
    deduplicate_right: bool = True,
) -> pd.DataFrame:
    """Merge robusto con armonización de tipos y deduplicación opcional en 'right'."""
    L, R = left.copy(), right.copy()
    if deduplicate_right and right_key in R.columns and R[right_key].duplicated().any():
        R = R.drop_duplicates(subset=[right_key], keep="first")
    if coerce_to_str:
        L[left_key] = L[left_key].astype(str)
        R[right_key] = R[right_key].astype(str)
    return L.merge(R, how=how, left_on=left_key, right_on=right_key)
