# app-tarea2.py — One-file Streamlit app (sin dependencias locales)
import os, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ========= Utilidades =========
def split_num_cat(df: pd.DataFrame, exclude=None):
    exclude = exclude or []
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    return num_cols, cat_cols

def cumulative_k_from_scores(scores: np.ndarray, threshold: float = 0.8) -> int:
    s = np.maximum(scores, 0)
    if s.sum() == 0:
        return max(1, min(10, len(scores)))
    w = s / s.sum()
    cum = np.cumsum(np.sort(w)[::-1])
    k = int(np.searchsorted(cum, threshold) + 1)
    return max(1, min(k, len(scores)))

# ========= Preprocesamiento =========
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

def apply_preprocessor(X: pd.DataFrame, num_cols, cat_cols):
    # Numéricas
    num_imputer = SimpleImputer(strategy='median')
    X_num = num_imputer.fit_transform(X[num_cols]) if num_cols else np.empty((len(X),0))
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num) if num_cols else np.empty((len(X),0))
    # Categóricas
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_cat = cat_imputer.fit_transform(X[cat_cols]) if cat_cols else np.empty((len(X),0))
    ohe = _make_ohe()
    X_cat_ohe = ohe.fit_transform(X_cat) if cat_cols else np.empty((len(X),0))
    # Nombres
    num_feature_names = num_cols
    if cat_cols:
        try:
            cat_ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
        except Exception:
            cat_ohe_names = [f"cat_{i}" for i in range(X_cat_ohe.shape[1])]
    else:
        cat_ohe_names = []
    return X_num_scaled, X_cat_ohe, num_feature_names, cat_ohe_names, scaler, ohe

# ========= Selección de variables =========
from sklearn.feature_selection import f_classif, chi2

def select_numeric_features(X_num: np.ndarray, y, threshold: float = 0.8):
    if X_num.shape[1] == 0:
        return np.array([], dtype=int), np.array([])
    f_scores, _ = f_classif(X_num, y)
    k = cumulative_k_from_scores(f_scores, threshold=threshold)
    idx_sorted = np.argsort(f_scores)[::-1]
    selected_idx = idx_sorted[:k]
    return selected_idx, f_scores

def select_categorical_features(X_cat_ohe: np.ndarray, y, threshold: float = 0.8):
    if X_cat_ohe.shape[1] == 0:
        return np.array([], dtype=int), np.array([])
    scores, _ = chi2(X_cat_ohe, y)  # OHE es no-negativo
    k = cumulative_k_from_scores(scores, threshold=threshold)
    idx_sorted = np.argsort(scores)[::-1]
    selected_idx = idx_sorted[:k]
    return selected_idx, scores

# ========= Reducción de dimensionalidad =========
from sklearn.decomposition import PCA, TruncatedSVD

def pca_reduce(X: np.ndarray, var_threshold: float = 0.9):
    if X.shape[1] == 0:
        return np.empty((X.shape[0], 0)), 0, np.array([])
    pca = PCA(n_components=None, svd_solver='full', random_state=0)
    X_pca = pca.fit_transform(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, var_threshold) + 1)
    return X_pca[:, :k], k, pca.explained_variance_ratio_

def mca_reduce(X_ohe: np.ndarray, var_threshold: float = 0.9):
    if X_ohe.shape[1] == 0:
        return np.empty((X_ohe.shape[0], 0)), 0, np.array([]), "none"
    try:
        import prince  # opcional
        mca = prince.MCA(n_components=min(X_ohe.shape[1]-1, 50), random_state=0)
        X_mca = mca.fit(X_ohe).transform(X_ohe).values
        if hasattr(mca, 'eigenvalues_'):
            ev = np.array(mca.eigenvalues_)
            evr = ev / ev.sum() if ev.sum() > 0 else np.ones_like(ev)/len(ev)
        else:
            evr = np.ones(X_mca.shape[1]) / X_mca.shape[1]
        cum = np.cumsum(evr)
        k = int(np.searchsorted(cum, var_threshold) + 1)
        return X_mca[:, :k], k, evr, "prince.MCA"
    except Exception:
        svd = TruncatedSVD(n_components=min(X_ohe.shape[1]-1, 50), random_state=0)
        X_svd = svd.fit_transform(X_ohe)
        evr = svd.explained_variance_ratio_
        cum = np.cumsum(evr)
        k = int(np.searchsorted(cum, var_threshold) + 1)
        return X_svd[:, :k], k, evr, "TruncatedSVD (MCA aprox)"

# ========= App =========
st.set_page_config(page_title="Tarea 2 - Selección + PCA/MCA", layout="wide")
st.title("Tarea 2 — Selección de Variables + PCA/MCA (One-file)")
st.caption("Dataset: diabetic_data.csv + IDS_mapping.csv")

with st.sidebar:
    st.header("Parámetros")
    sel_threshold = st.slider("Umbral acumulado selección (ANOVA/χ²)", 0.5, 0.99, 0.80, 0.01)
    var_threshold = st.slider("Umbral varianza explicada (PCA/MCA)", 0.5, 0.99, 0.90, 0.01)
    debug_mode = st.checkbox("Modo debug (mostrar trazas)", value=False)
    st.markdown("---")
    st.write("Carga de archivos (opcional; o usa rutas por defecto)")
    diabetic_file = st.file_uploader("diabetic_data.csv", type=["csv"])
    ids_file = st.file_uploader("IDS_mapping.csv", type=["csv"])

# Carga datasets
default_diabetic = os.path.join("data", "diabetic_data.csv")
default_ids = os.path.join("data", "IDS_mapping.csv")

try:
    if diabetic_file is not None:
        diabetic_df = pd.read_csv(diabetic_file, low_memory=False)
    else:
        diabetic_df = pd.read_csv(default_diabetic, low_memory=False) if os.path.exists(default_diabetic) else None

    if ids_file is not None:
        ids_df = pd.read_csv(ids_file, low_memory=False)
    else:
        ids_df = pd.read_csv(default_ids, low_memory=False) if os.path.exists(default_ids) else None
except Exception as e:
    st.error("Error leyendo CSVs.")
    if debug_mode:
        st.code(traceback.format_exc())
    st.stop()

if diabetic_df is None:
    st.warning("Sube `diabetic_data.csv` o colócalo en `data/diabetic_data.csv`.")
    st.stop()
if ids_df is None:
    st.warning("Sube `IDS_mapping.csv` o colócalo en `data/IDS_mapping.csv`.")
    st.stop()

# UI para merge robusto
st.sidebar.markdown("---")
st.sidebar.subheader("Emparejar IDs (merge)")

def _candidates(cols):
    cands = [c for c in cols if 'id' in c.lower()]
    prefer = ["encounter_id","patient_nbr","admission_type_id","discharge_disposition_id","admission_source_id"]
    for p in prefer:
        if p in cols and p not in cands:
            cands.append(p)
    return cands or list(cols)

left_key = st.sidebar.selectbox("Columna en diabetic_data", _candidates(diabetic_df.columns), index=0)
right_key = st.sidebar.selectbox("Columna en IDS_mapping", _candidates(ids_df.columns), index=0)
coerce_to_str = st.sidebar.checkbox("Forzar llaves a texto antes del merge", value=True)
join_type = st.sidebar.selectbox("Tipo de join", ["left","inner","right","outer"], index=0)

df = diabetic_df.copy()
if left_key and right_key:
    LKEY, RKEY = "__merge_left_key__", "__merge_right_key__"
    try:
        if coerce_to_str:
            df[LKEY] = df[left_key].astype(str)
            ids_df[RKEY] = ids_df[right_key].astype(str)
        else:
            df[LKEY] = df[left_key]
            ids_df[RKEY] = ids_df[right_key]
        df = df.merge(ids_df, how=join_type, left_on=LKEY, right_on=RKEY)
        if LKEY in df.columns:
            df.drop(columns=[LKEY], inplace=True, errors="ignore")
    except Exception as e:
        st.error(f"Error al hacer el merge con {left_key} ↔ {right_key}")
        if debug_mode:
            st.code(traceback.format_exc())
        st.stop()

st.subheader("Vista previa")
st.dataframe(df.head(20))

# Selección de target
candidate_targets = [c for c in df.columns if c.lower() in ["readmitted","outcome","target","label"]]
target_col = st.selectbox("Selecciona la columna objetivo (y)", candidate_targets or df.columns.tolist(), index=0)

if target_col not in df.columns:
    st.error("La columna objetivo seleccionada no existe en el DataFrame.")
    st.stop()

y = df[target_col]
X = df.drop(columns=[target_col])

# División num/cat
num_cols, cat_cols = split_num_cat(X)
st.write(f"Variables numéricas: {len(num_cols)} | Variables categóricas: {len(cat_cols)}")

# Preprocesamiento
try:
    X_num_scaled, X_cat_ohe, num_names, cat_ohe_names, scaler, ohe = apply_preprocessor(X, num_cols, cat_cols)
except Exception as e:
    st.error("Error en preprocesamiento.")
    if debug_mode:
        st.code(traceback.format_exc())
    st.stop()

# Selección
try:
    num_idx, num_scores = select_numeric_features(X_num_scaled, y, threshold=sel_threshold)
    cat_idx, cat_scores = select_categorical_features(X_cat_ohe, y, threshold=sel_threshold)
except Exception as e:
    st.error("Error en selección de variables.")
    if debug_mode:
        st.code(traceback.format_exc())
    st.stop()

X_num_sel = X_num_scaled[:, num_idx] if X_num_scaled.shape[1] else np.empty((len(X),0))
X_cat_sel = X_cat_ohe[:, cat_idx] if X_cat_ohe.shape[1] else np.empty((len(X),0))

st.subheader("Selección de variables")
st.write(f"Seleccionadas num: {X_num_sel.shape[1]} / {X_num_scaled.shape[1]} | "
         f"cat: {X_cat_sel.shape[1]} / {X_cat_ohe.shape[1]}")

# Reducción
try:
    X_pca, k_pca, evr_pca = pca_reduce(X_num_sel, var_threshold=var_threshold)
    X_mca, k_mca, evr_mca, mca_method = mca_reduce(X_cat_sel, var_threshold=var_threshold)
except Exception as e:
    st.error("Error en reducción de dimensionalidad.")
    if debug_mode:
        st.code(traceback.format_exc())
    st.stop()

st.subheader("Reducción de dimensionalidad")
col1, col2 = st.columns(2)
with col1:
    st.write(f"PCA componentes: {k_pca}")
    if len(evr_pca):
        fig1, ax1 = plt.subplots()
        ax1.plot(np.cumsum(evr_pca))
        ax1.set_xlabel("Componentes")
        ax1.set_ylabel("Varianza explicada acumulada")
        ax1.set_title("PCA — Varianza acumulada")
        st.pyplot(fig1)
with col2:
    st.write(f"MCA componentes: {k_mca} (método: {mca_method})")
    if len(evr_mca):
        fig2, ax2 = plt.subplots()
        ax2.plot(np.cumsum(evr_mca))
        ax2.set_xlabel("Componentes")
        ax2.set_ylabel("Varianza explicada acumulada (aprox)")
        ax2.set_title("MCA — Varianza acumulada")
        st.pyplot(fig2)

# Salida final
Z = np.concatenate([X_pca, X_mca], axis=1) if X_mca.size or X_pca.size else np.empty((len(X),0))
out = pd.DataFrame(Z, index=df.index)
out[target_col] = y.values

st.subheader("Resultado")
st.dataframe(out.head(20))

# Guardado/descarga
os.makedirs("outputs", exist_ok=True)
out_fp = os.path.join("outputs", "dataset_pca_mca_sel.csv")
try:
    out.to_csv(out_fp, index=False)
    st.success(f"Archivo guardado en {out_fp}")
except Exception as e:
    st.warning("No se pudo escribir en disco (entornos con permisos restringidos).")
    if debug_mode:
        st.code(traceback.format_exc())

st.download_button("Descargar CSV procesado",
                   data=out.to_csv(index=False).encode("utf-8"),
                   file_name="dataset_pca_mca_sel.csv",
                   mime="text/csv")

# Panel de diagnóstico opcional
if debug_mode:
    st.markdown("---")
    st.subheader("Diagnóstico")
    st.write({
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "cwd": os.getcwd(),
        "diabetic_shape": diabetic_df.shape,
        "ids_shape": ids_df.shape,
        "cols_diabetic_head": diabetic_df.columns[:10].tolist(),
        "cols_ids_head": ids_df.columns[:10].tolist(),
        "target_col": target_col
    })
