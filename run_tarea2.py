import os
import argparse
import pandas as pd
import numpy as np
from data_loading import load_diabetic_and_ids
from preprocessing import apply_preprocessor
from feature_selection import select_numeric_features, select_categorical_features
from dimensionality import pca_reduce, mca_reduce
from utils import split_num_cat

# ... (resto igual; en el ZIP está completo)

def main(args):
    df = load_diabetic_and_ids(args.diabetic, args.ids, left_key=args.merge_left_key, right_key=args.merge_right_key)

    # Autodetección de target
    target = args.target
    if target is None:
        for cand in ["readmitted", "Outcome", "outcome", "target", "label"]:
            if cand in df.columns:
                target = cand
                break
    if target is None:
        raise ValueError("No se encontró columna objetivo. Usa --target para especificarla.")

    y = df[target]
    X = df.drop(columns=[target])

    # División num/cat
    num_cols, cat_cols = split_num_cat(X)

    # Preprocesamiento
    X_num_scaled, X_cat_ohe, num_names, cat_ohe_names, scaler, ohe = apply_preprocessor(X, num_cols, cat_cols)

    # Selección de variables (ANOVA F y χ²)
    num_idx, num_scores = select_numeric_features(X_num_scaled, y, threshold=args.sel_threshold)
    cat_idx, cat_scores = select_categorical_features(X_cat_ohe, y, threshold=args.sel_threshold)

    X_num_sel = X_num_scaled[:, num_idx] if X_num_scaled.shape[1] else np.empty((len(X),0))
    X_cat_sel = X_cat_ohe[:, cat_idx] if X_cat_ohe.shape[1] else np.empty((len(X),0))

    # Reducción de dimensionalidad
    X_pca, k_pca, evr_pca = pca_reduce(X_num_sel, var_threshold=args.var_threshold)
    X_mca, k_mca, evr_mca, mca_method = mca_reduce(X_cat_sel, var_threshold=args.var_threshold)

    # Concatenación
    Z = np.concatenate([X_pca, X_mca], axis=1) if X_mca.size or X_pca.size else np.empty((len(X),0))

    # Salida
    out = pd.DataFrame(Z, index=df.index)
    out[target] = y.values

    os.makedirs(args.outputs, exist_ok=True)
    out_fp = os.path.join(args.outputs, "dataset_pca_mca_sel.csv")
    out.to_csv(out_fp, index=False)

    # Metadatos
    meta = {
        "target": target,
        "num_selected": int(X_pca.shape[1]) if X_pca.size else 0,
        "cat_selected": int(X_mca.shape[1]) if X_mca.size else 0,
        "k_pca": int(k_pca),
        "k_mca": int(k_mca),
        "mca_method": mca_method,
        "sel_threshold": args.sel_threshold,
        "var_threshold": args.var_threshold,
        "merge_left_key": args.merge_left_key,
        "merge_right_key": args.merge_right_key
    }
    pd.Series(meta).to_json(os.path.join(args.outputs, "metadata_tarea2.json"), indent=2)
    print(f"OK -> {out_fp}")
    return out_fp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diabetic", type=str, required=True, help="Ruta a diabetic_data.csv")
    parser.add_argument("--ids", type=str, required=True, help="Ruta a IDS_mapping.csv")
    parser.add_argument("--target", type=str, default=None, help="Columna objetivo")
    parser.add_argument("--sel_threshold", type=float, default=0.80, help="Umbral acumulado para selección de variables")
    parser.add_argument("--var_threshold", type=float, default=0.90, help="Umbral varianza explicada para PCA/MCA")
    parser.add_argument("--outputs", type=str, default="outputs", help="Carpeta de salida")
    parser.add_argument("--merge_left_key", type=str, default=None, help="Columna de la tabla principal (diabetic) para merge")
    parser.add_argument("--merge_right_key", type=str, default=None, help="Columna del mapping (IDS) para merge")
    args = parser.parse_args()
    main(args)
