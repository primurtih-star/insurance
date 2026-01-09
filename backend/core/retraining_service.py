# imports
from datetime import datetime
import json
import os

# third_party
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "uploaded_dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "ml", "models")
ENCODER_PATH = os.path.join(PROJECT_ROOT, "ml", "encoder.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# helpers
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.lower()
    required = ["age", "bmi", "smoker", "charges"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan dalam dataset!")
    return df

def preprocess(df: pd.DataFrame):
    df = df.copy()
    df["smoker_num"] = df["smoker"].astype(str).str.lower().map({"yes": 1, "no": 0})
    if "region" in df.columns:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        region_encoded = encoder.fit_transform(df[["region"]])
        joblib.dump(encoder, ENCODER_PATH)
        region_names = encoder.get_feature_names_out(["region"])
        df_region = pd.DataFrame(region_encoded, columns=region_names, index=df.index)
    else:
        encoder = None
        df_region = pd.DataFrame()
    X_base = df[["age", "bmi", "children", "smoker_num"]].fillna(0)
    X = pd.concat([X_base, df_region], axis=1)
    y = df["charges"]
    return X, y, encoder

def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

def save_model(model, model_name, metrics):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, model_name)
    joblib.dump(model, model_path)
    metadata = {
        "active_model": model_name,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    return metadata

# main_training_function
def retrain_model():
    print("⏳ Memulai proses training...")
    try:
        df = load_dataset()
    except Exception as e:
        print(f"❌ Error Load Data: {e}")
        return
    df = df.dropna()
    print("⚙️  Preprocessing & Encoding...")
    X, y, encoder = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("🚀 Melatih Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    metrics = calculate_metrics(model, X_test, y_test)
    print(f"📊 Akurasi (R2 Score): {metrics['r2']*100:.2f}%")
    model_name = "insurance_model.joblib"
    save_model(model, model_name, metrics)
    print(f"✅ Model tersimpan di: {MODEL_DIR}/{model_name}")
    print(f"✅ Metadata tersimpan di: {METADATA_PATH}")

if __name__ == "__main__":
    retrain_model()