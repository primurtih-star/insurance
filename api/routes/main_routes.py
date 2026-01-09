# imports
from __future__ import annotations
import json
import logging
from pathlib import Path

# third party
from flask import Blueprint, current_app, jsonify, render_template, request

# safe imports
try:
    from core.constants import APP_NAME, VERSION
except Exception:
    APP_NAME = "MyApp"
    VERSION = "0.0.0"

try:
    from ml import engine
except Exception:
    engine = None

# blueprint
logger = logging.getLogger(__name__)
main_bp = Blueprint("main", __name__, template_folder="templates", url_prefix="")

# api routes
@main_bp.route("/api/ping")
def api_ping():
    return jsonify({"status": "ok", "message": "API Connected 🚀"})

@main_bp.route("/api/status")
def api_status():
    return jsonify(
        {
            "system": APP_NAME,
            "version": VERSION,
            "status": "online",
            "ai_engine": "active" if engine else "inactive",
        }
    )

# ui pages
@main_bp.route("/admin", endpoint="admin")
def admin_page():
    return render_template("admin.html", title="Admin Control")

@main_bp.route("/admin/add", endpoint="admin_add")
def admin_add_page():
    return render_template("admin_add.html", title="Add Admin")

@main_bp.route("/analysis", endpoint="analysis")
def analysis_page():
    return render_template("analysis.html", title="Model Analysis")

@main_bp.route("/dashboard", endpoint="dashboard")
def dashboard_page():
    metrics_path = Path("ml/models/latest_metrics.json")
    accuracy_display = "N/A"
    model_status = "Not Trained"

    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                data = json.load(f)
                acc = data.get("accuracy", 0)
                accuracy_display = f"{acc * 100:.1f}%"
                model_status = "Trained"
        except Exception as e:
            logger.error(f"Error reading metrics: {e}")
            accuracy_display = "Error"

    return render_template(
        "dashboard.html",
        title="Dashboard",
        accuracy=accuracy_display,
        model_status=model_status,
    )

@main_bp.route("/dataset", endpoint="dataset")
def dataset_page():
    return render_template("dataset.html", title="Dataset Manager")

@main_bp.route("/history", endpoint="history")
def history_page():
    return render_template("history.html", title="History Logs")

@main_bp.route("/home", endpoint="home")
def home_page():
    return render_template("index.html", title="System Overview")

@main_bp.route("/kuitansi", endpoint="invoice")
def view_kuitansi():
    return render_template("pdf_template.html", title="Cetak Kuitansi")

@main_bp.route("/", endpoint="landing")
def landing_page():
    return render_template("intro.html", title="Initializing...")

@main_bp.route("/model-arena", endpoint="model_arena")
def model_arena_page():
    return render_template("model_compare.html", title="Model Arena")

@main_bp.route("/predict", endpoint="prediction_form")
def predict_page():
    return render_template("predict.html", title="AI Prediction Engine")

@main_bp.route("/risk", endpoint="risk")
def risk_page():
    return render_template("risk.html", title="Risk Analysis")

@main_bp.route("/setting", endpoint="setting")
def setting_page():
    return render_template("setting.html", title="Platform Settings")

# error handlers
@main_bp.app_errorhandler(404)
def handle_404(err):
    logger.warning("404: %s %s", request.path, err)
    return render_template("404.html", title="Not Found"), 404

@main_bp.app_errorhandler(500)
def handle_500(err):
    logger.exception("500 error: %s", err)
    return render_template("500.html", title="Server Error"), 500

