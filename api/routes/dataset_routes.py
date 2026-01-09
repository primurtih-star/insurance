from flask import Blueprint, request, jsonify
import pandas as pd
import os

bp = Blueprint("dataset_api", __name__)

DATASET_PATH = "data/raw/uploaded_dataset.csv"


# ------------------------------------------
# 🔧 UTILITY: NORMALISASI + CLEAN DATA
# ------------------------------------------
def clean_dataset(df):
    # pastikan semua kolom jadi lower-case
    df.columns = [c.strip().lower() for c in df.columns]

    # SEX
    if "sex" in df.columns:
        df["sex"] = df["sex"].astype(str).str.strip().str.lower()
    else:
        df["sex"] = "unknown"

    # SMOKER
    if "smoker" in df.columns:
        df["smoker"] = df["smoker"].astype(str).str.strip().str.lower()
    else:
        df["smoker"] = "no"

    # REGION
    if "region" in df.columns:
        df["region"] = df["region"].astype(str).str.strip().str.lower()
    else:
        df["region"] = "unknown"

    # AGE
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    else:
        df["age"] = 0

    # BMI
    if "bmi" in df.columns:
        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    else:
        df["bmi"] = 0.0

    # CHILDREN
    if "children" in df.columns:
        df["children"] = pd.to_numeric(df["children"], errors="coerce")
    else:
        df["children"] = 0

    return df



# ------------------------------------------
# 📌 UPLOAD DATASET CSV
# ------------------------------------------
@bp.route("/upload", methods=["POST"])
def upload_dataset():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400

    # 🔥 CLEAN DATASET (fix smoker, sex, lowercase, trim)
    df = clean_dataset(df)

    # Simpan versi yang sudah dibersihkan
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)

    return jsonify({
        "status": "uploaded",
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records")
    })


# ------------------------------------------
# 📌 GET STORED DATASET
# ------------------------------------------
@bp.route("/", methods=["GET"])
def get_dataset():

    if not os.path.exists(DATASET_PATH):
        return jsonify([])

    df = pd.read_csv(DATASET_PATH)

    # 🔥 CLEAN lagi untuk jaga-jaga
    df = clean_dataset(df)

    return jsonify(df.to_dict(orient="records"))
