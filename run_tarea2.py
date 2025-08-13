from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
from data_loading import read_csv_smart, merge_by_keys
from utils import split_num_cat
from preprocessing import apply_preprocessor
from feature_selection import select_numeric_features, select_categorical_features
from dimensionality import pca_reduce, mca_reduce

def main(args):
    # Carga
    d = read_csv_smart(args.diabetic)
    m = read_csv_smart(args.ids)

    # Merge robusto
    df = merge_by_keys(
        d, m,
        left_key=args.merge_left_key,
        right_key=args.merge_right_key,
        how=args.join_type,
        coerce_to_str=args.coerce_keys_to_str,
        deduplicate_right=args.deduplicate_right
    )

    # Target
    target = args.target
    if target is None:
        for cand in ["readmitted", "Outcome", "outcome", "target", "label"]:
            if cand in df.columns:
                target = cand
                break
    if target is None or target not in df.columns:
        raise ValueError("No se encontró columna objetivo. Usa --target para especificarla.")
    y = df[target]
    X = df.drop(columns=[target])

    # División y preprocesamiento
    num_cols, cat_cols = split_num_cat(X)
    X_num_scaled, X_cat_ohe, _, _, _, _ = apply_preprocessor(X, num_cols, cat_cols)

    # Selección
    num_idx, num_scores = select_numeric_features(X_num_scaled, y, threshold=args.sel_threshold)
    cat_idx, cat_scores = select_categorical_features(X_cat_ohe, y, threshold=args.sel_threshold)

    X_num_sel = X_num_scaled[:, num_idx] if X_num_scaled.size and len(num_idx) else np.empty((len(X), 0))
    X_cat_sel = X_cat_ohe[:, cat_idx] if (X_cat_ohe is not None and X_cat_ohe.shape[1] and len(cat_idx)) else None

    # Reducción
    X_pca, k_pca, evr_pca = pca_reduce(X_num_sel, var_threshold=args.var_threshold) if X_num_sel.shape[1] else (np.empty((len(X),0)), 0, np.array([]))
    X_mca, k_mca, evr_mca, mca_method = mca_reduce(X_cat_sel, var_threshold=args.var_threshold, max_components=args.max_mca_components) if (X_cat_sel is not None and X_cat_sel.shape[1]) else (np.empty((len(X),0)), 0, np.array([]), "none")

    # Concatenación y salida
    Z = np.concatenate([X_pca, X_mca], axis=1) if (X_pca.shape[1] or X_mca.shape[1]) else np.empty((len(X),0))
    out = pd.DataFrame(Z, index=df.index)
    out[target] = y.values

    os.makedirs(args.outputs, exist_ok=True)
    out_fp = os.path.join(args.outputs, "dataset_pca_mca_sel.csv")
    out.to_csv(out_fp, index=False)

    meta = {
        "target": target,
        "num_selected": int(X_pca.shape[1]),
        "cat_selected": int(X_mca.shape[1]),
        "k_pca": int(k_pca),
        "k_mca": int(k_mca),
        "mca_method": mca_method,
        "sel_threshold": args.sel_threshold,
        "var_threshold": args.var_threshold,
        "merge_left_key": args.merge_left_key,
        "merge_right_key": args.merge_right_key,
        "join_type": args.join_type,
        "coerce_keys_to_str": bool(args.coerce_keys_to_str),
        "deduplicate_right": bool(args.deduplicate_right)
    }
    pd.Series(meta).to_json(os.path.join(args.outputs, "metadata_tarea2.json"), indent=2)
    print(f"OK -> {out_fp}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--diabetic", required=True, help="Ruta a diabetic_data.csv")
    p.add_argument("--ids", required=True, help="Ruta a IDS_mapping.csv")
    p.add_argument("--target", default=None, help="Columna objetivo")
    p.add_argument("--sel_threshold", type=float, default=0.80, help="Umbral acumulado para selección")
    p.add_argument("--var_threshold", type=float, default=0.90, help="Umbral varianza explicada PCA/MCA")
    p.add_argument("--max_mca_components", type=int, default=50, help="Tope de componentes para SVD (MCA)")
    p.add_argument("--outputs", default="outputs", help="Carpeta de salida")
    p.add_argument("--merge_left_key", required=True, help="Columna clave en diabetic")
    p.add_argument("--merge_right_key", required=True, help="Columna clave en IDS")
    p.add_argument("--join_type", default="left", choices=["left","inner","right","outer"], help="Tipo de join")
    p.add_argument("--coerce_keys_to_str", action="store_true", help="Forzar llaves a texto antes del merge")
    p.add_argument("--deduplicate_right", action="store_true", help="Eliminar duplicados por clave en IDS antes de unir")
    args = p.parse_args()
    main(args)

