# Tarea 2 — Selección de Variables + PCA/MCA (Flat, memoria baja)

## Estructura
Todos los módulos en la raíz:
- utils.py, data_loading.py, preprocessing.py, feature_selection.py, dimensionality.py
- run_tarea2.py, app-tarea2.py
- data/ (coloca aquí diabetic_data.csv e IDS_mapping.csv)

## CLI
pip install -r requirements.txt
python run_tarea2.py \
  --diabetic data/diabetic_data.csv \
  --ids data/IDS_mapping.csv \
  --target readmitted \
  --merge_left_key admission_type_id \
  --merge_right_key admission_type_id \
  --coerce_keys_to_str \
  --deduplicate_right

## App
streamlit run app-tarea2.py
