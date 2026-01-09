from flask import Blueprint, request, jsonify
import json, os
from datetime import datetime

ADMIN_DATA_PATH = "data/history/admin_entries.json"

bp = Blueprint("admin_routes", __name__)

@bp.route("/add", methods=["POST"])
def add_data():
    data = request.get_json() or {}

    # --- Validate fields ---
    required = ["age", "bmi", "smoker"]
    if not all(key in data for key in required):
        return jsonify({"error": "Missing fields"}), 400

    # Ensure directory exists
    os.makedirs(os.path.dirname(ADMIN_DATA_PATH), exist_ok=True)

    # --- Load existing history safely ---
    history = []
    if os.path.exists(ADMIN_DATA_PATH):
        try:
            with open(ADMIN_DATA_PATH, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []  # Reset if file corrupted

    # --- Add timestamp ---
    entry = {
        "age": int(data["age"]),
        "bmi": float(data["bmi"]),
        "smoker": bool(data["smoker"]),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    history.append(entry)

    # --- Save back to file safely ---
    with open(ADMIN_DATA_PATH, "w") as f:
        json.dump(history, f, indent=4)

    return jsonify({"status": "saved", "saved_data": entry})