from backend.core.analysis_service import analyze_model

def get_model_analysis():
    """Return model diagnostic or analysis data."""
    result = analyze_model()
    return result
