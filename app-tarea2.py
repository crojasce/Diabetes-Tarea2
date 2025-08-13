import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.data_loading import load_diabetic_and_ids
from src.preprocessing import apply_preprocessor
from src.feature_selection import select_numeric_features, select_categorical_features
from src.dimensionality import pca_reduce, mca_reduce
from src.utils import split_num_cat

st.set_page_config(page_title="Tarea 2 - Selección + PCA/MCA", layout="wide")

st.title("Tarea 2 — Selección de Variables + PCA/MCA")
st.caption("Dataset: diabetic_data.csv + IDS_mapping.csv")

with st.sidebar:
    st.header("Parámetros")
    sel_threshold = st.slider("Umbral acumulado selección (ANOVA/χ²)", 0.5, 0.99, 0.80, 0.01)
    var_threshold = st.slider("Umbral varianza explicada (PCA/MCA)", 0.5, 0.99, 0.90, 0.01)
    st.markdown("---")
    st.write("Carga de archivos (opcional; o usa rutas por defecto)")
    diabetic_file = st.file_uploader("diabetic_data.csv", type=["csv"])
    ids_file = st.file_uploader("IDS_mapping.csv", type=["csv"])

# Carga por defecto desde carpeta data/
default_diabetic = os.path.join("data", "diabetic_data.csv")
default_ids = os.path.join("data", "IDS_mapping.csv")

if diabetic_file is not None:
    diabetic_df = pd.read_csv(diabetic_file, low_memory=False)
else:
    diabetic_df = pd.read_csv(default_diabetic, low_memory=False) if os.path.exists(default_diabetic) else None

if ids_file is not None:
    ids_df = pd.read_csv(ids_file, low_memory=False)
else:
    ids_df = pd.read_csv(default_ids, low_memory=False) if os.path.exists(default_ids) else None

if diabetic_df is None:
    st.warning("Sube `diabetic_data.csv` o colócalo en `data/diabetic_data.csv`.")
    st.stop()

if ids_df is None:
    st.warning("Sube `IDS_mapping.csv` o colócalo en `data/IDS_mapping.csv`.")
    st.stop()

# Merge (heurística por columnas con 'id')
df = diabetic_df.merge(
        ids_df,
        how="left",
        left_on=[c for c in diabetic_df.columns if 'id' in c.lower()][0] if any('id' in c.lower() for c in diabetic_df.columns) else None,
        right_on=[c for c in ids_df.columns if 'id' in c.lower()][0] if any('id' in c.lower() for c in ids_df.columns) else None
    ) if any('id' in c.lower() for c in diabetic_df.columns) and any('id' in c.lower() for c in ids_df.columns) else diabetic_df

st.subheader("Vista previa")
st.dataframe(df.head(20))

# Target
candidate_targets = [c for c in df.columns if c.lower() in ["readmitted","outcome","target","label"]]
target_col = st.selectbox("Selecciona la columna objetivo (y)", candidate_targets or df.columns.tolist(), index=0)

y = df[target_col]
X = df.drop(columns=[target_col])

num_cols, cat_cols = split_num_cat(X)
st.write(f"Variables numéricas: {len(num_cols)} | Variables categóricas: {len(cat_cols)}")

# Preprocesamiento
X_num_scaled, X_cat_ohe, num_names, cat_ohe_names, scaler, ohe = apply_preprocessor(X, num_cols, cat_cols)

# Selección
num_idx, num_scores = select_numeric_features(X_num_scaled, y, threshold=sel_threshold)
cat_idx, cat_scores = select_categorical_features(X_cat_ohe, y, threshold=sel_threshold)

X_num_sel = X_num_scaled[:, num_idx] if X_num_scaled.shape[1] else np.empty((len(X),0))
X_cat_sel = X_cat_ohe[:, cat_idx] if X_cat_ohe.shape[1] else np.empty((len(X),0))

st.subheader("Selección de variables")
st.write(f"Seleccionadas num: {X_num_sel.shape[1]} / {X_num_scaled.shape[1]} | cat: {X_cat_sel.shape[1]} / {X_cat_ohe.shape[1]}")

# Reducción
X_pca, k_pca, evr_pca = pca_reduce(X_num_sel, var_threshold=var_threshold)
X_mca, k_mca, evr_mca, mca_method = mca_reduce(X_cat_sel, var_threshold=var_threshold)

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

# Resultado
Z = np.concatenate([X_pca, X_mca], axis=1) if X_mca.size or X_pca.size else np.empty((len(X),0))
out = pd.DataFrame(Z, index=df.index)
out[target_col] = y.values

st.subheader("Resultado")
st.dataframe(out.head(20))

# Guardar/descargar
os.makedirs("outputs", exist_ok=True)
out_fp = os.path.join("outputs", "dataset_pca_mca_sel.csv")
out.to_csv(out_fp, index=False)
st.success(f"Archivo guardado en {out_fp}")
st.download_button("Descargar CSV procesado", data=out.to_csv(index=False).encode("utf-8"), file_name="dataset_pca_mca_sel.csv", mime="text/csv")
