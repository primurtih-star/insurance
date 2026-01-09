import os
import json
import time
import joblib
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL_DIR = "ml/models"
INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")


# ============================================================
# 1. Load Dataset Test untuk Comparison
# ============================================================

def load_test_data():
    """
    Dataset kecil untuk evaluasi model.
    Format harus sama dengan input model.
    """
    test_file = "data/test/test_eval.json"

    if not os.path.exists(test_file):
        raise FileNotFoundError(
            "File test_eval.json tidak ditemukan. "
            "Buat dataset evaluasi di /data/test/"
        )

    with open(test_file, "r") as f:
        data = json.load(f)

    X = np.array([row["X"] for row in data])
    y = np.array([row["y"] for row in data])

    return X, y


# ============================================================
# 2. Evaluate model
# ============================================================

def evaluate_model(model, X, y):
    """Hitung RMSE, MAE, R² dan waktu prediksi."""

    # waktu mulai
    start = time.time()
    y_pred = model.predict(X)
    duration = (time.time() - start) * 1000  # ms

    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "latency_ms": float(duration)
    }


# ============================================================
# 3. Compare all models in model folder
# ============================================================

def compare_all_models():
    """
    Membandingkan performa setiap model .joblib di folder.
    Menghasilkan dictionary untuk UI Model Arena.
    """

    # 1. Ambil test data
    X, y = load_test_data()

    # 2. Ambil semua model
    models = [
        f for f in os.listdir(MODEL_DIR)
        if f.endswith(".joblib")
    ]

    if len(models) == 0:
        return {"error": "Tidak ada model .joblib ditemukan"}

    results = {}

    for model_file in models:
        model_path = os.path.join(MODEL_DIR, model_file)

        try:
            model = joblib.load(model_path)
            metrics = evaluate_model(model, X, y)

            results[model_file] = metrics

        except Exception as e:
            # Kalau model error → tetap dicatat
            results[model_file] = {
                "error": str(e),
                "rmse": None,
                "mae": None,
                "r2": None,
                "latency_ms": None
            }

    # 4. Simpan file informasi model_info.json
    with open(INFO_PATH, "w") as f:
        json.dump(results, f, indent=4)

    return results


# ============================================================
# 4. Load hasil comparison (untuk Model Arena UI)
# ============================================================

def load_compare_results():
    if not os.path.exists(INFO_PATH):
        return {}

    with open(INFO_PATH, "r") as f:
        return json.load(f)
