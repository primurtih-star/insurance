import pandas as pd
import os

DATASET_PATH = "data/dataset/dataset.csv"

def load_summary():
    if not os.path.exists(DATASET_PATH):
        return {"error": "Dataset tidak ditemukan"}

    df = pd.read_csv(DATASET_PATH)

    return {
        "total_records": len(df),
        "avg_age": round(df["age"].mean(), 2),
        "avg_bmi": round(df["bmi"].mean(), 2),
        "avg_charges": round(df["charges"].mean(), 2),
        "smoker_rate": round((df["smoker"].eq("yes").mean()) * 100, 2)
    }
