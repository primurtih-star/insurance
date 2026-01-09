import os
import pandas as pd
from datetime import datetime

DATASET_PATH = os.path.join("data", "dataset", "insurance.csv")
DATASET_DIR = os.path.dirname(DATASET_PATH)

os.makedirs(DATASET_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_dataset():
    """Return dataset as pandas DataFrame."""
    if not os.path.exists(DATASET_PATH):
        return pd.DataFrame()  # kosong jika file belum ada

    return pd.read_csv(DATASET_PATH)



# ============================================================
# 2. SAVE UPDATED DATASET
# ============================================================

def save_dataset(df: pd.DataFrame):
    """Replace entire dataset with new DataFrame."""
    df.to_csv(DATASET_PATH, index=False)
    return True



# ============================================================
# 3. APPEND 1 ROW (untuk form insert)
# ============================================================

def append_row(data: dict):
    """
    data: {
        'age': 30,
        'bmi': 25.3,
        'children': 2,
        'smoker': 'yes',
        'region': 'southwest',
        'charges': 4500
    }
    """

    df = load_dataset()

    new_row = pd.DataFrame([data])
    df = pd.concat([df, new_row], ignore_index=True)

    save_dataset(df)
    return True



# ============================================================
# 4. EXPORT DATASET FILE
# ============================================================

def export_dataset():
    """Return dataset for download (used by export controller)."""
    df = load_dataset()
    return df.to_csv(index=False)



# ============================================================
# 5. GET SUMMARY (dipakai dashboard)
# ============================================================

def dataset_summary():
    df = load_dataset()
    if df.empty:
        return {
            "total_rows": 0,
            "avg_age": 0,
            "avg_bmi": 0,
            "avg_charge": 0,
            "smoker_ratio": 0
        }

    return {
        "total_rows": len(df),
        "avg_age": round(df["age"].mean(), 2),
        "avg_bmi": round(df["bmi"].mean(), 2),
        "avg_charge": round(df["charges"].mean(), 2),
        "smoker_ratio": round(df["smoker"].value_counts(normalize=True).to_dict().get("yes", 0) * 100, 2)
    }