from backend.config.db import read_json
from backend.config.settings import HISTORY_PATH
from backend.core.summary_service import build_summary

def get_dashboard_data():
    history = read_json(HISTORY_PATH)
    summary = build_summary(history)
    return summary