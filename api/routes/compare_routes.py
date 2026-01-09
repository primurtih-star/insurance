from flask import Blueprint, jsonify
import pandas as pd
import time
import os
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Blueprint Configuration
bp = Blueprint("compare_routes", __name__)

# --- CONFIG PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "uploaded_dataset.csv")
METRICS_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "latest_metrics.json") # Path untuk Dashboard

def load_and_prep_dataset():
    if not os.path.exists(DATASET_PATH):
        return None, f"Dataset tidak ditemukan di: {DATASET_PATH}"

    try:
        df = pd.read_csv(DATASET_PATH)
        df.columns = df.columns.str.lower().str.strip()
        
        if 'charges' not in df.columns:
            return None, "Kolom target 'charges' tidak ditemukan dalam CSV."

        if 'sex' in df.columns:
            df['sex'] = df['sex'].astype(str).str.lower().map({'male': 1, 'female': 0}).fillna(0)
        if 'smoker' in df.columns:
            df['smoker'] = df['smoker'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)
        if 'region' in df.columns:
            df['region'] = df['region'].astype('category').cat.codes

        df_numeric = df.select_dtypes(include=[np.number])
        df_clean = df_numeric.dropna()

        if df_clean.empty:
            return None, "Dataset kosong setelah pembersihan data."
            
        return df_clean, None

    except Exception as exc:
        return None, f"Error sistem saat loading data: {exc}"

def save_winner_metrics(winner):
    """
    Menyimpan skor pemenang ke file JSON agar bisa dibaca Dashboard.
    """
    try:
        # Pastikan folder ada
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        
        # Konversi r2 (persen) kembali ke desimal (0.86) untuk konsistensi
        accuracy_decimal = winner['r2'] / 100.0
        
        data = {
            "accuracy": accuracy_decimal,
            "algorithm": winner['model'],
            "rmse": winner['rmse'],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(METRICS_PATH, "w") as f:
            json.dump(data, f, indent=4)
            
        print(f"💾 Metrics Juara disimpan: {accuracy_decimal*100}% ({winner['model']})")
    except Exception as e:
        print(f"⚠️ Gagal menyimpan metrics: {e}")

# --- ROUTE UTAMA ---
@bp.route("/run", methods=["GET"])
def run_arena():
    df, err = load_and_prep_dataset()
    if err:
        return jsonify({"error": err}), 400

    try:
        y = df["charges"]
        X = df.drop(columns=["charges"])
    except KeyError:
        return jsonify({"error": "Kolom 'charges' hilang."}), 400

    if len(df) < 10:
        return jsonify({"error": "Data terlalu sedikit."}), 400

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ("Decision Tree", DecisionTreeRegressor(random_state=42)),
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=50, random_state=42)),
    ]

    leaderboard = []
    errors_log = []

    for name, model in models:
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            duration = time.time() - start_time

            r2 = r2_score(y_test, pred)
            
            # Hitung RMSE Manual (Safe)
            mse = mean_squared_error(y_test, pred)
            rmse = np.sqrt(mse)

            if np.isnan(r2) or np.isinf(r2): r2 = 0.0
            if np.isnan(rmse) or np.isinf(rmse): rmse = 999999.0

            leaderboard.append({
                "model": name,
                "r2": round(float(r2) * 100, 2),
                "rmse": round(float(rmse), 2),
                "time": round(duration, 4)
            })

        except Exception as e:
            errors_log.append(f"{name}: {str(e)}")
            continue

    leaderboard = sorted(leaderboard, key=lambda x: x["r2"], reverse=True)

    if not leaderboard:
        return jsonify({"error": "Semua model gagal dilatih."}), 500

    # --- FITUR BARU: SIMPAN JUARA KE DASHBOARD ---
    save_winner_metrics(leaderboard[0])
    # ---------------------------------------------

    return jsonify({
        "status": "success",
        "rows_used": len(df),
        "leaderboard": leaderboard
    })