# backend/controllers/prediction_controller.py

from flask import request, jsonify
from flask import jsonify
from ml.preprocess import preprocess_fit
from backend.core.prediction_service import PredictionService
from backend.core.prediction_service import predict_cost

prediction_service = PredictionService()

def predict_handler():
    """Handle request prediksi dari endpoint /api/predict."""
    payload = request.get_json(silent=True) or {}

    # panggil service
    result = prediction_service.predict(payload)

    return jsonify({
        "input": payload,
        "prediction": result
    })

def run_prediction(payload):
    try:
        age = payload["age"]
        bmi = payload["bmi"]
        smoker = payload["smoker"]

        result = predict_cost(age, bmi, smoker)
        return {
            "status": "success",
            "prediction": result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def predict_api():
    """
    Menerima JSON POST:
    {
        "age": 30,
        "bmi": 25.4,
        "smoker": "no",
        "region": "southeast"
    }
    """
    data = request.get_json()

    try:
        result = predict_cost(data)
        return jsonify({
            "status": "success",
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

from backend.core.prediction_service import predict_insurance_cost

def run_prediction_api(payload):
    return predict_insurance_cost(payload)
