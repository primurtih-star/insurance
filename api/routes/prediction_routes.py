# imports
from __future__ import annotations
import json
import logging
import os
import shutil
import tempfile
import threading
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, render_template, request

from backend.core.prediction_service import predict_cost

# config
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("RiskEngineBackend")

bp = Blueprint("prediction_routes", __name__)
HISTORY_FILE = os.path.join("data", "history", "prediction_history.json")
data_lock = threading.Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "insurance_model.joblib")

# helpers
class HistoryManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self._ensure_directory()
        self.memory_cache = self._read_json_safe()

    def _ensure_directory(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            self._write_atomic([])

    def _read_json_safe(self):
        if not os.path.exists(self.filepath):
            return []
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Gagal membaca history: {e}")
            return []

    def _write_atomic(self, data):
        dir_name = os.path.dirname(self.filepath)
        with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_name, suffix='.tmp') as tf:
            json.dump(data, tf, indent=4)
            temp_name = tf.name
        shutil.move(temp_name, self.filepath)
        self.memory_cache = data

    def reload_cache(self):
        with data_lock:
            self.memory_cache = self._read_json_safe()

    def get_all(self, page=None, per_page=None):
        with data_lock:
            self.memory_cache = self._read_json_safe()
            data = self.memory_cache
            if page is not None and per_page is not None:
                start = (page - 1) * per_page
                end = start + per_page
                return {
                    "total_records": len(data),
                    "page": page,
                    "per_page": per_page,
                    "data": data[start:end],
                }
            return data

    def add_entry(self, entry_data):
        with data_lock:
            self.memory_cache = self._read_json_safe()
            self.memory_cache.append(entry_data)
            self._write_atomic(self.memory_cache)
            return entry_data

    def update_entry(self, index, new_data):
        with data_lock:
            self.memory_cache = self._read_json_safe()
            if index < 0 or index >= len(self.memory_cache):
                raise IndexError("Index out of bounds")
            current = self.memory_cache[index]
            updated_record = {**current, **new_data}
            updated_record["last_modified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.memory_cache[index] = updated_record
            self._write_atomic(self.memory_cache)
            return updated_record

    def delete_entry(self, index):
        with data_lock:
            self.memory_cache = self._read_json_safe()
            if index < 0 or index >= len(self.memory_cache):
                raise IndexError("Index out of bounds")
            self.memory_cache.pop(index)
            self._write_atomic(self.memory_cache)

    def clear_all(self):
        with data_lock:
            self._write_atomic([])

manager = HistoryManager(HISTORY_FILE)

def validate_and_parse(data):
    try:
        name = str(data.get("name", "Anonim")).strip()[:50]
        region = str(data.get("region", "unknown")).lower().strip()
        age = int(data.get("age", 0))
        bmi = float(data.get("bmi", 0))
        children = int(data.get("children", 0))
        smoker = str(data.get("smoker", "")).lower().strip()
        sex = str(data.get("sex", "")).lower().strip()

        if age < 0 or age > 120:
            raise ValueError("Usia tidak valid (0-120).")
        if bmi < 10 or bmi > 100:
            raise ValueError("BMI tidak realistis.")
        if smoker not in ["yes", "no"]:
            raise ValueError("Smoker wajib 'yes'/'no'.")
        if children < 0:
            raise ValueError("Jumlah anak tidak boleh negatif.")

        return {
            "name": name,
            "age": age,
            "bmi": bmi,
            "smoker": smoker,
            "children": children,
            "region": region,
            "sex": sex,
        }
    except (TypeError, ValueError) as e:
        logger.error(f"Validation Error: {e}")
        raise ValueError(f"Input Data Invalid: {str(e)}")

def _build_model_input(clean_data: dict) -> pd.DataFrame:
    sex_code = 1 if clean_data.get("sex") == "male" else 0
    smoker_code = 1 if clean_data.get("smoker") == "yes" else 0
    region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
    region_code = region_map.get(clean_data.get("region", "southeast"), 2)
    return pd.DataFrame([{
        "age": clean_data["age"],
        "sex": sex_code,
        "bmi": clean_data["bmi"],
        "children": clean_data["children"],
        "smoker": smoker_code,
        "region": region_code,
    }])

def _predict_with_fallback(model_obj, clean_data: dict) -> float:
    if model_obj is not None:
        try:
            input_df = _build_model_input(clean_data)
            pred = float(model_obj.predict(input_df)[0])
            if np.isnan(pred) or np.isinf(pred):
                raise ValueError("Model menghasilkan nilai tidak valid.")
            return pred
        except Exception as e:
            logger.warning(f"Model gagal, gunakan fallback: {e}")

    base = 2000.0
    pred_age = clean_data["age"] * 250.0
    bmi = clean_data["bmi"]
    pred_bmi = bmi * 300.0 if bmi > 30 else bmi * 50.0
    pred_smoke = 24000.0 if clean_data["smoker"] == "yes" else 0.0
    return base + pred_age + pred_bmi + pred_smoke

# routes
@bp.route("/", methods=["POST"])
def run_prediction():
    try:
        raw_data = request.get_json()
        if not raw_data:
            return jsonify({"error": "No JSON Payload"}), 400

        clean_data = validate_and_parse(raw_data)

        try:
            prediction_result = predict_cost(clean_data)
        except Exception as e:
            logger.warning(f"predict_cost gagal: {e}")
            prediction_result = None

        model_obj = None
        try:
            if os.path.exists(MODEL_PATH):
                model_obj = joblib.load(MODEL_PATH)
        except Exception as e:
            logger.error(f"Gagal load model: {e}")

        final_pred = prediction_result if prediction_result is not None else _predict_with_fallback(model_obj, clean_data)
        final_cost = max(0.0, float(final_pred))

        entry = {
            **clean_data,
            "prediction": final_cost,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "api",
        }
        manager.add_entry(entry)

        return jsonify({"status": "success", "message": "Prediksi berhasil.", "prediction": final_cost, "data": clean_data}), 201

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.critical(f"Server Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@bp.route("/history", methods=["GET"])
def get_history():
    page = request.args.get("page", type=int)
    limit = request.args.get("limit", default=1000, type=int)
    data = manager.get_all(page=page, per_page=limit) if page else manager.get_all()
    return jsonify(data), 200

@bp.route("/history/update/<int:index>", methods=["PUT"])
def update_record(index):
    try:
        clean_data = validate_and_parse(request.get_json())
        try:
            new_prediction = predict_cost(clean_data)
        except Exception as e:
            logger.warning(f"predict_cost gagal saat update: {e}")
            model_obj = None
            try:
                if os.path.exists(MODEL_PATH):
                    model_obj = joblib.load(MODEL_PATH)
            except Exception as le:
                logger.error(f"Gagal load model untuk update: {le}")
            new_prediction = _predict_with_fallback(model_obj, clean_data)

        update_payload = {**clean_data, "prediction": max(0.0, float(new_prediction))}
        updated_entry = manager.update_entry(index, update_payload)
        return jsonify({"status": "success", "data": updated_entry}), 200
    except (IndexError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Update gagal")
        return jsonify({"error": "Internal Server Error"}), 500

@bp.route("/history/delete/<int:index>", methods=["DELETE"])
def delete_single(index):
    try:
        manager.delete_entry(index)
        return jsonify({"status": "success"}), 200
    except IndexError:
        return jsonify({"error": "Data not found"}), 404

@bp.route("/history/clear", methods=["DELETE"])
def clear_history():
    manager.clear_all()
    return jsonify({"status": "success"}), 200

@bp.route("/analysis")
@bp.route("/risk")
def view_analysis():
    return render_template("risk.html")

@bp.route("/admin/add", methods=["GET"])
def view_admin_panel():
    return render_template("admin_add.html", title="Admin Injection Terminal")

@bp.route("/admin/add", methods=["POST"])
def admin_add_data():
    try:
        data = request.get_json()
        clean_data = validate_and_parse(data)
        clean_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clean_data["source"] = "admin_console"
        try:
            clean_data["prediction"] = predict_cost(clean_data)
        except Exception as e:
            logger.warning(f"predict_cost gagal di admin: {e}")
            model_obj = None
            try:
                if os.path.exists(MODEL_PATH):
                    model_obj = joblib.load(MODEL_PATH)
            except Exception as le:
                logger.error(f"Gagal load model untuk admin: {le}")
            clean_data["prediction"] = _predict_with_fallback(model_obj, clean_data)

        manager.add_entry(clean_data)
        return jsonify({"status": "success", "message": "Data admin tersimpan", "data": clean_data}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400