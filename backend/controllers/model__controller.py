from backend.core.model_service import get_model_info, reload_model

def model_info():
    return get_model_info()

def refresh_model():
    return reload_model()
