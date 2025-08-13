import os, sys, traceback, gc
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data_loading import read_csv_smart, merge_by_keys
from utils import split_num_cat
from preprocessing import apply_preprocessor
from feature_selection import select_numeric_features, select_categorical_features
from dimensionality import pca_reduce, mca_reduce

st.set_page_config(page_title="Tarea 2 - Selección + PCA/MCA", layout="wide")
st.title("Tarea 2 — Selección de Variables + PCA/MCA")
st.caption("Dataset: diabetic_data.csv + IDS_mapping.csv")

# ===== Sidebar =====
with st.sidebar:
    st.header("Parámetros")
    sel_threshold = st.slider("Umbral acumulado selección (ANOVA/χ²)", 0.5, 0.99, 0.80, 0.01)
    var_threshold = st.slider("Umbral varianza explicada (PCA/MCA)", 0.5, 0.99, 0.90, 0.01)
    st.markdown("**Categóricas (alto cardinalidad)**")
    high_card_threshold = st.number_input("Umbral alto cardinalidad", 20, 1000, 100, 10)
    hash_features = st.number_input("Hash features por col. alta", 16, 512, 64, 16)
    st.markdown("---")
    debug = st.checkbox("Modo debug (mostrar traza)", value=False)
    st.markdown("---")
    st.write("Carga de archivos (opcional; o usa rutas por defecto)")
    diabetic_file = st.file_uploader("diabetic_data.csv", type=["csv"])
    ids_file = st.file_uploader("IDS_mapping.csv", type=["csv"])

@st.cache_data(show_spinner=False)
def _read(diabetic_file, ids_file, default_d, default_i):
    if diabetic_file is not None:
        d = read_csv_smart(diabetic_file)
    else:
        d = read_csv_smart(default_d) if os.path.exists(default_d) else None
    if ids_file is not None:
        i = read_csv_smart(ids_file)
    else:
        i = read_csv_smart(default_i) if os.path.exists(default_i) else None
    return d, i

default_diabetic = os.path.join("data", "diabetic_data.csv")
default_ids = os.path.join("data", "IDS_mapping.csv")
diabetic_df, ids_df = _read(diabetic_file, ids_file, default_diabetic, default_ids)

if diabetic_df is None:
    st.warning("Sube `diabetic_data.csv` o colócalo en `data/diabetic_data.csv`.")
    st.stop()
if ids_df is None:
    st.warning("Sube `IDS_mapping.csv` o colócalo en `data/IDS_mapping.csv`.")
    st.stop()

# ===== Merge UI =====
def _candidates(cols):
    cands = [c for c in cols if "id" in c.lower()]
    preferred = ["encounter_id","patient_nbr","admission_type_id","discharge_disposition_id","admission_source_id"]
    for p in preferred:
        if p in cols and p not in cands:
            cands.append(p)
    return cands or list(cols)

st.sidebar.markdown("---")
st.sidebar.subheader("Emparejar IDs (merge)")
left_key = st.sidebar.selectbox("Columna en diabetic_data", _candidates(diabetic_df.columns), index=0)
right_key = st.sidebar.selectbox("Columna en IDS_mapping", _candidates(ids_df.columns), index=0)
coerce = st.sidebar.checkbox("Forzar llaves a texto", value=True)
dedup_r = st.sidebar.checkbox("Deduplicar IDS por su llave", value=True)
join_type = st.sidebar.selectbox("Tipo de join", ["left", "inner", "right", "outer"], index=0)

try:
    df = merge_by_keys(diabetic_df, ids_df, left_key, right_key, how=join_type, coerce_to_str=coerce, deduplicate_right=dedup_r)
except Exception:
    st.error(f"Error al hacer el merge con {left_key} ↔ {right_key}")
    if debug:
        st.code(traceback.format_exc())
    st.stop()

st.subheader("Vista previa tras merge")
st.dataframe(df.head(20))

# ===== Target =====
candidate_targets = [c for c in df.columns if c.lower() in ["readmitted","outcome","target","label"]]
target_col = st.selectbox("Selecciona la columna objetivo (y)", candidate_targets or df.columns.tolist(), index=0)
if target_col not in df.columns:
    st.error("La columna objetivo seleccionada no existe en el DataFrame.")
    st.stop()

y = df[target_col]
X = df.drop(columns=[target_col])

# ===== Prepro =====
num_cols, cat_cols = split_num_cat(X)
st.write(f"Variables numéricas: {len(num_cols)} | Variables categóricas: {len(cat_cols)}")

try:
    X_num_scaled, X_cat_sparse, low_names, hashed_meta, scaler, encs = apply_preprocessor(
        X, num_cols, cat_cols,
        high_card_threshold=int(high_card_threshold),
        hash_features=int(hash_features),
    )
except Exception:
    st.error("Error en preprocesamiento.")
    if debug:
        st.code(traceback.format_exc())
    st.stop()

# ===== Selección =====
try:
    num_idx, _ = select_numeric_features(X_num_scaled, y, threshold=sel_threshold)
    cat_idx, _ = select_categorical_features(X_cat_sparse, y, threshold=sel_threshold)
except Exception:
    st.error("Error en selección de variables.")
    if debug:
        st.code(traceback.format_exc())
    st.stop()

X_num_sel = X_num_scaled[:, num_idx] if X_num_scaled.size and len(num_idx) else np.empty((len(X), 0))
X_cat_sel = X_cat_sparse[:, cat_idx] if (X_cat_sparse is not None and X_cat_sparse.shape[1] and len(cat_idx)) else None

st.subheader("Selección de variables")
st.write(f"Seleccionadas num: {X_num_sel.shape[1]} / {X_num_scaled.shape[1]} | "
         f"cat: {(X_cat_sel.shape[1] if X_cat_sel is not None else 0)} / {(X_cat_sparse.shape[1] if X_cat_sparse is not None else 0)}")

# ===== Reducción =====
try:
    X_pca, k_pca, evr_pca = pca_reduce(X_num_sel, var_threshold=var_threshold) if X_num_sel.shape[1] else (np.empty((len(X),0)), 0, np.array([]))
    X_mca, k_mca, evr_mca, mca_method = mca_reduce(X_cat_sel, var_threshold=var_threshold, max_components=50) if (X_cat_sel is not None and X_cat_sel.shape[1]) else (np.empty((len(X),0)), 0, np.array([]), "none")
except Exception:
    st.error("Error en reducción de dimensionalidad.")
    if debug:
        st.code(traceback.format_exc())
    st.stop()

st.subheader("Reducción de dimensionalidad")
c1, c2 = st.columns(2)
with c1:
    st.write(f"PCA componentes: {k_pca}")
    if len(evr_pca):
        fig1, ax1 = plt.subplots()
        ax1.plot(np.cumsum(evr_pca))
        ax1.set_xlabel("Componentes"); ax1.set_ylabel("Varianza explicada acumulada")
        ax1.set_title("PCA — Varianza acumulada")
        st.pyplot(fig1)
with c2:
    st.write(f"MCA componentes: {k_mca} (método: {mca_method})")
    if len(evr_mca):
        fig2, ax2 = plt.subplots()
        ax2.plot(np.cumsum(evr_mca))
        ax2.set_xlabel("Componentes"); ax2.set_ylabel("Varianza explicada acumulada (aprox)")
        ax2.set_title("MCA — Varianza acumulada")
        st.pyplot(fig2)

# ===== Resultado final =====
Z = np.concatenate([X_pca, X_mca], axis=1) if (X_pca.shape[1] or X_mca.shape[1]) else np.empty((len(X),0))
out = pd.DataFrame(Z, index=df.index)
out[target_col] = y.values

st.subheader("Resultado")
st.dataframe(out.head(20))

os.makedirs("outputs", exist_ok=True)
out_fp = os.path.join("outputs", "dataset_pca_mca_sel.csv")
try:
    out.to_csv(out_fp, index=False)
    st.success(f"Archivo guardado en {out_fp}")
except Exception:
    st.info("En algunos entornos no se puede escribir en disco; usa el botón de descarga.")

st.download_button(
    "Descargar CSV procesado",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="dataset_pca_mca_sel.csv",
    mime="text/csv"
)

# Limpieza
del X_num_scaled, X_cat_sparse, X_num_sel, X_cat_sel
gc.collect()

if debug:
    st.markdown("---")
    st.subheader("Diagnóstico")
    st.write({
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "shape_diabetic": diabetic_df.shape,
        "shape_ids": ids_df.shape,
        "left_key": left_key, "right_key": right_key, "join_type": join_type,
        "high_card_threshold": int(high_card_threshold),
        "hash_features": int(hash_features),
        "low_card_features": len(low_names),
        "hashed_cols": len(hashed_meta),
        "hashed_meta": hashed_meta
    })
