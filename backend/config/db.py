import os, json
from backend.config.settings import DATE_FORMAT
from datetime import datetime

def read_json(file_path):
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except:
        return []

def write_json(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def append_json(file_path, entry: dict):
    data = read_json(file_path)
    entry["timestamp"] = datetime.now().strftime(DATE_FORMAT)
    data.append(entry)
    write_json(file_path, data)