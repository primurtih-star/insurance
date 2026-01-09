# backend/core/prediction_service.py

import json
import os
import joblib
import pandas as pd
from datetime import datetime

# --- IMPORT DARI MODUL BARU ---
from ml.preprocess import preprocess_transform

# Paths Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HISTORY_DIR = os.path.join(BASE_DIR, "data", "history")
HISTORY_PATH = os.path.join(HISTORY_DIR, "prediction_history.json")

MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


# ======================================================
# 1. LOAD MODEL ENGINE
# ======================================================
def load_active_model():
    """
    Mencoba memuat model berdasarkan metadata.
    """
    default_model_path = os.path.join(MODEL_DIR, "rf_model.joblib")
    target_model_file = "rf_model.joblib" 

    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                metadata = json.load(f)
                target_model_file = metadata.get("best_model", "rf_model.joblib")
                if not target_model_file: target_model_file = "rf_model.joblib"
        except Exception:
            pass

    model_path = os.path.join(MODEL_DIR, target_model_file)

    if not os.path.exists(model_path):
        if os.path.exists(default_model_path):
            return joblib.load(default_model_path)
        else:
            raise FileNotFoundError(f"CRITICAL: Tidak ada model di {MODEL_DIR}")

    return joblib.load(model_path)


# ======================================================
# 2. SAVE LOG HISTORY
# ======================================================
def save_prediction_log(input_data, prediction_result):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": input_data,
        "prediction": prediction_result
    }

    history = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                history = json.load(f)
                if not isinstance(history, list): history = []
        except (json.JSONDecodeError, ValueError):
            history = []

    history.append(entry)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

    return entry


# ======================================================
# 3. CORE LOGIC (MAIN FUNCTION)
# ======================================================
def predict_cost(data):
    """
    Fungsi logika utama.
    """
    try:
        # 1. Validasi Format Data
        if isinstance(data, dict):
            df_input = pd.DataFrame([data])
        elif isinstance(data, list):
            df_input = pd.DataFrame(data)
        else:
            raise ValueError("Format data harus Dictionary atau List.")

        # 2. Preprocessing (Transform)
        X = preprocess_transform(df_input)

        # 3. Load Model & Predict
        model = load_active_model()
        prediction_val = model.predict(X)[0]
        prediction = float(prediction_val)

        # 4. Simpan Log
        save_prediction_log(data, prediction)

        return prediction

    except Exception as e:
        print(f"ERROR in predict_cost: {str(e)}")
        raise e


# ======================================================
# 4. COMPATIBILITY LAYER (SOLUSI ERROR IMPORT)
# ======================================================

# A. Alias Fungsi (Untuk error 'predict_insurance_cost' not found)
# Ini membuat nama 'predict_insurance_cost' merujuk ke fungsi 'predict_cost' yang sama
predict_insurance_cost = predict_cost

# B. Class Wrapper (Untuk error 'PredictionService' not found)
class PredictionService:
    @staticmethod
    def predict(data):
        return predict_cost(data)