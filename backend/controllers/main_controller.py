from backend.config.settings import SYSTEM_NAME, VERSION
from flask import jsonify

def system_status():
    return {
        "system": SYSTEM_NAME,
        "version": VERSION,
        "status": "online"
    }