from backend.core.export_service import export_history

def export_data(format="csv"):
    """Export history data in specific format."""
    return export_history(format)
