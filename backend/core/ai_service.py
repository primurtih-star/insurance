import os
import json
import joblib

from ml.preprocess import preprocess_fit, preprocess_transform
from ml.utils.scaler import load_scaler
from ml.utils.encoding import load_encoder
from ml.utils.validator import validate_input

MODEL_DIR = "ml/models"
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


# =============================================================
# 1. LOAD MODEL AKTIF
# =============================================================
def load_active_model():
    """Load model ML yang sedang aktif berdasarkan metadata."""
    
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("model_metadata.json tidak ditemukan!")

    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)

    active_model = meta.get("active_model")

    if not active_model:
        raise Exception("active_model tidak didefinisikan di metadata!")

    model_path = os.path.join(MODEL_DIR, active_model)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} tidak ditemukan.")

    return joblib.load(model_path)


# =============================================================
# 2. MAIN PREDICTION PROCESS
# =============================================================
def run_prediction(data: dict):
    """
    data = {
        "age": 30,
        "bmi": 23.4,
        "smoker": "yes",
        "region": "southeast"
    }
    """

    # 1. Validasi input
    validate_input(data)

    # 2. Load scaler, encoder, model
    scaler = load_scaler()
    encoder = load_encoder()
    model = load_active_model()

    # 3. Preprocess input → menghasilkan array numerik siap prediksi
    X = preprocess_input(
        data=data,
        scaler=scaler,
        encoder=encoder
    )

    # 4. Prediksi
    y_pred = model.predict([X])[0]

    return float(y_pred)
