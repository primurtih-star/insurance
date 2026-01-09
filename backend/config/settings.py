import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "..", "data")
ML_DIR = os.path.join(BASE_DIR, "..", "ml")
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
EXPORT_DIR = os.path.join(BASE_DIR, "..", "export")

# File paths
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
HISTORY_PATH = os.path.join(DATA_DIR, "history", "history.json")
ADMIN_HISTORY_PATH = os.path.join(DATA_DIR, "history", "admin_entries.json")

MODEL_PATH = os.path.join(ML_DIR, "model.pkl")
MODEL_INFO_PATH = os.path.join(ML_DIR, "model_info.json")

# System Configs
SYSTEM_NAME = "INSURE AI"
VERSION = "2.0"
CURRENCY_DEFAULT = "USD"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

AUTO_RETRAIN = False
RISK_THRESHOLD = 0.7