import pandas as pd
from typing import Optional

def load_diabetic_and_ids(diabetic_path: str, ids_path: str, left_key: Optional[str] = None, right_key: Optional[str] = None) -> pd.DataFrame:
    """Carga ambos CSV y realiza merge. Si se entregan claves explícitas, las usa; de lo contrario intenta heurística por columnas que contengan 'id'."""
    d = pd.read_csv(diabetic_path, low_memory=False)
    m = pd.read_csv(ids_path, low_memory=False)

    if left_key and right_key and left_key in d.columns and right_key in m.columns:
        return d.merge(m, left_on=left_key, right_on=right_key, how="left")

    # Heurística de merge por columnas con "id"
    d_keys = [c for c in d.columns if "id" in c.lower()]
    m_keys = [c for c in m.columns if "id" in c.lower()]
    if d_keys and m_keys:
        common = set(d_keys).intersection(set(m_keys))
        if common:
            on = list(common)[0]
            return d.merge(m, on=on, how="left")
        else:
            return d.merge(m, left_on=d_keys[0], right_on=m_keys[0], how="left")
    # Si no hay claves claras, devolver sin merge
    return d
