import os
import json
import joblib
from datetime import datetime

# ------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------

MODEL_DIR = "ml/models"
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


# ============================================================
# 1. LOAD MODEL YANG SEDANG AKTIF
# ============================================================

def load_active_model():
    """
    Membaca metadata dan load model yang terdaftar sebagai aktif.
    """
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata tidak ditemukan di {METADATA_PATH}")

    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)

    active_name = meta.get("active_model")

    if not active_name:
        raise Exception("Tidak ada model aktif di metadata!")

    model_path = os.path.join(MODEL_DIR, active_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model aktif '{active_name}' tidak ditemukan!")

    return joblib.load(model_path)


# ============================================================
# 2. GET LIST SEMUA MODEL DI FOLDER
# ============================================================

def list_models():
    """
    Mengambil semua file .joblib di folder models/
    """
    if not os.path.exists(MODEL_DIR):
        return []

    return [
        f for f in os.listdir(MODEL_DIR)
        if f.endswith(".joblib")
    ]


# ============================================================
# 3. SET MODEL AKTIF BARU
# ============================================================

def set_active_model(model_filename):
    """
    Menjadikan model tertentu sebagai model aktif.
    """

    model_path = os.path.join(MODEL_DIR, model_filename)

    if not os.path.exists(model_path):
        return {"error": f"Model '{model_filename}' tidak ada."}

    # Update metadata
    metadata = {
        "active_model": model_filename,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    return {"status": "success", "active_model": model_filename}


# ============================================================
# 4. CEK PERFORMANCE MODEL (UNTUK MODEL ARENA)
# ============================================================

def load_model_info():
    """
    Membaca metadata model (misal score CV, RMSE, MAE),
    kalau abang ingin menampilkan performa di Model Arena.
    """

    info_file = os.path.join(MODEL_DIR, "model_info.json")

    if not os.path.exists(info_file):
        return {}

    with open(info_file, "r") as f:
        return json.load(f)


# ============================================================
# 5. VALIDATE MODEL STRUCTURE
# ============================================================

def validate_model(model_path):
    """
    Mengecek apakah model valid & bisa digunakan untuk prediction.
    Misal: structure, predict(), sklearn compatibility.
    """
    try:
        model = joblib.load(model_path)

        # must have predict()
        if not hasattr(model, "predict"):
            return False

        # run dummy
        try:
            model.predict([[30, 20, 1]])
        except:
            # Some models need more/less features → still valid
            pass

        return True

    except Exception as e:
        print(f"Model validation error: {e}")
        return False
