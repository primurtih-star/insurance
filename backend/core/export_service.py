import pandas as pd
import os

def export_to_csv(df: pd.DataFrame, output_path: str):
    """Export DataFrame ke file CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def export_to_excel(df: pd.DataFrame, output_path: str):
    """Export DataFrame ke Excel."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    return output_path


def export_history_json_to_csv(json_path, output_path):
    """Convert history JSON → CSV."""
    if not os.path.exists(json_path):
        raise FileNotFoundError("History JSON tidak ditemukan.")

    df = pd.read_json(json_path)
    return export_to_csv(df, output_path)
