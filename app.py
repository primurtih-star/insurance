#!/usr/bin/env python3
"""
Predicture Application Entry Point.

This module initializes the Flask application, configures logging,
security headers, rate limiting, and defines page routing logic.

Usage:
    Run directly for development: python app.py
    Use create_app() for WSGI production servers (Gunicorn/uWSGI).
"""

from __future__ import annotations

import re  # <--- Tambahkan ini di bagian imports
import os
import sys
import time
import socket
import signal
import logging
import logging.handlers
from typing import Optional
from pathlib import Path

# Third-party imports
import psutil  # optional, kept if used elsewhere
from colorama import init as colorama_init, Fore, Style
from flask import (
    Flask, send_from_directory, render_template,
    request, g, jsonify
)
from flask_cors import CORS
from flask_compress import Compress
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

# Import blueprint and optional register function from project
try:
    from api.routes.main_routes import main_bp
except Exception:
    main_bp = None

# Internal imports (soft)
try:
    from core.env_loader import load_environment
except Exception:
    load_environment = None

try:
    from core.config import get_config, BaseConfig as Config
except Exception:
    get_config = None
    Config = None


try:
    from api import register_routes
except Exception:
    register_routes = None

# --- Configuration & Setup ---
colorama_init(autoreset=True)
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "ui" / "templates"
STATIC_DIR = BASE_DIR / "ui"
LOGS_DIR = BASE_DIR / "logs"

# Ensure log directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Global instances
limiter: Optional[Limiter] = None


def setup_logger(name: str) -> logging.Logger:
    """Configures a rotating file logger (clean) and console output (colored)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Pola Regex untuk mendeteksi kode warna ANSI
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

    # --- Custom Formatter untuk File (Pembersih Warna) ---
    class StripColorFormatter(logging.Formatter):
        def format(self, record):
            # Simpan pesan asli agar console tetap berwarna
            original_msg = record.msg
            
            # Hapus kode warna dari pesan jika tipe datanya string
            if isinstance(record.msg, str):
                record.msg = ansi_escape.sub('', record.msg)
            
            # Format pesan (tambahkan timestamp, level, dll)
            formatted_message = super().format(record)
            
            # Kembalikan pesan asli ke record (supaya handler lain tidak kena dampak)
            record.msg = original_msg
            return formatted_message

    if not logger.handlers:
        # Format Log Standar
        log_format_str = "[%(asctime)s] %(levelname)s [%(name)s]: %(message)s"
        date_format_str = "%Y-%m-%d %H:%M:%S"
        
        # 1. Console Handler (Tetap Berwarna)
        # Menggunakan Formatter bawaan, jadi warna dari colorama tetap muncul
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(log_format_str, date_format_str))
        logger.addHandler(ch)

        # 2. File Handler (Bersih tanpa kode warna)
        # Menggunakan StripColorFormatter yang kita buat di atas
        fh = logging.handlers.RotatingFileHandler(
            LOGS_DIR / "system.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3
        )
        # Di sini kuncinya: kita pasang formatter pembersih
        fh.setFormatter(StripColorFormatter(log_format_str, date_format_str))
        fh.setLevel(logging.WARNING) # Atau ubah ke INFO jika ingin mencatat semua request
        logger.addHandler(fh)

    return logger


logger = setup_logger("Predicture")


# --- Request hooks and security (attach to blueprint main_bp) ---
if main_bp is not None:
    @main_bp.before_app_request
    def start_timer():
        """Track request start time for performance monitoring."""
        g.start = time.perf_counter()

    @main_bp.after_app_request
    def security_and_logging(response):
        """Apply security headers and log request duration."""
        # Security Headers
        headers = {
            "X-Frame-Options": "SAMEORIGIN",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https:; "
                "style-src 'self' 'unsafe-inline' https:; "
                "font-src 'self' https: data:; "
                "img-src 'self' data:;"
            )
        }
        for key, value in headers.items():
            response.headers[key] = value

        # Logging (Skip static assets to keep logs clean)
        if not request.path.startswith(("/assets", "/static")):
            duration = time.perf_counter() - getattr(g, "start", time.perf_counter())
            status = response.status_code

            # Color coding for console output only
            color = Fore.GREEN if status < 300 else Fore.YELLOW if status < 500 else Fore.RED
            logger.info(f"{request.method} {request.path} -> {color}{status}{Style.RESET_ALL} ({duration:.4f}s)")

        return response

else:
    logger.warning("main_bp not found: request hooks and page routes will not be attached.")


# --- Page routes
# Note: routes are expected to be defined in api.routes.main_routes via main_bp.
# If you prefer to define routes here, you can import main_bp and add @main_bp.route(...) functions.
# For completeness, if main_bp is missing, we still define minimal local routes on a fallback blueprint.

if main_bp is None:
    # Fallback: create a minimal blueprint to avoid app errors in dev without api.routes
    from flask import Blueprint
    main_bp = Blueprint("main", __name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))

    @main_bp.route("/", endpoint="landing")
    def landing():
        return render_template("intro.html", title="Initializing...")

    @main_bp.route("/home", endpoint="home")
    def home():
        return render_template("index.html", title="System Overview")

    @main_bp.route("/dashboard", endpoint="dashboard")
    def dashboard_page():
        return render_template("dashboard.html", title="Dashboard")

    @main_bp.route("/predict", endpoint="prediction_form")
    def prediction_form():
        return render_template("predict.html", title="AI Prediction Engine")

    @main_bp.route("/dataset", endpoint="dataset")
    def dataset():
        return render_template("dataset.html", title="Dataset Manager")

    @main_bp.route("/model-arena", endpoint="model_arena")
    def model_arena():
        return render_template("model_compare.html", title="Model Arena")

    @main_bp.route("/history", endpoint="history")
    def history():
        return render_template("history.html", title="History Logs")

    @main_bp.route("/risk", endpoint="risk")
    def risk_page():
        return render_template("risk.html", title="Risk Analysis")

    @main_bp.route("/analysis", endpoint="analysis")
    def analysis_page():
        return render_template("analysis.html", title="Analytics Dashboard")

    @main_bp.route("/admin/add", endpoint="admin_add")
    def admin_panel():
        return render_template("admin_add.html", title="Admin Panel")

    @main_bp.route("/setting", endpoint="settings")
    def settings_page():
        return render_template("setting.html", title="Platform Settings")

    @main_bp.route("/kuitansi/view", endpoint="invoice_view")
    def kuitansi_view():
        return render_template("pdf_template.html", title="Digital Receipt View")

    @main_bp.route("/kuitansi/print", endpoint="invoice_print")
    def kuitansi_print():
        return render_template("pdf_template.html", title="Print Preview", mode="print")


# --- Application Factory ---
def create_app(config_class: Optional[object] = None) -> Flask:
    """
    Factory to create and configure the Flask application instance.
    Includes rate limiting, compression, and CORS support.
    """
    global limiter

    # 1. Environment & Config
    if load_environment:
        try:
            load_environment()
        except Exception as e:
            logger.error(f"Environment load failed: {e}")

    app = Flask(__name__, static_folder=str(STATIC_DIR), template_folder=str(TEMPLATE_DIR))

    # Load configuration: priority -> explicit config_class -> get_config() -> Config -> default
    try:
        if config_class:
            app.config.from_object(config_class)
        elif get_config:
            cfg = get_config()
            app.config.from_object(cfg)
        elif Config:
            app.config.from_object(Config)
        else:
            logger.warning("No configuration object found; using Flask defaults.")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")

    # 2. Extensions
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    Compress(app)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

    # 3. Rate Limiting
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["500 per minute", "20 per second"],
        storage_uri="memory://"
    )
    limiter.init_app(app)
    app.limiter = limiter  # type: ignore

    # 4. Register Blueprints / Routes
    app.register_blueprint(main_bp)

    # If project provides additional API route registration, call it
    if register_routes:
        try:
            register_routes(app, limiter)
        except Exception as e:
            logger.error(f"Failed to register API routes: {e}")

    # 5. Error Handlers
    @app.errorhandler(404)
    def not_found(e):
        return render_template("404.html", title="Page Not Found"), 404

    @app.errorhandler(500)
    def server_error(e):
        logger.error(f"Server Error: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500

    @app.errorhandler(429)
    def ratelimit_handler(e):
        return jsonify({"error": "Rate limit exceeded"}), 429

    # 6. Specific Route Limits
    # Attempt to apply a per-view limit to the prediction endpoint.
    # The view key is "<blueprint_name>.<endpoint>" after registration.
    # We assume blueprint name is main_bp.name (commonly "main").
    try:
        bp_name = main_bp.name or "main"
        prediction_key = f"{bp_name}.prediction_form"
        if prediction_key in app.view_functions:
            limiter.limit("100/hour")(app.view_functions[prediction_key])
        else:
            # Try alternative endpoint name used in some definitions
            alt_key = f"{bp_name}.predict"
            if alt_key in app.view_functions:
                limiter.limit("100/hour")(app.view_functions[alt_key])
    except Exception as e:
        logger.debug(f"Could not apply specific route limit: {e}")

    return app


# Utilities & Entry Point
def find_available_port(start_port: int = 5000) -> int:
    """Helper to find the first free port."""
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    raise RuntimeError("No available ports found.")


def _graceful_exit(signum, frame):
    logger.info(f"Signal {signum} received. Shutting down.")
    sys.exit(0)


if __name__ == "__main__":
    app = create_app()

    # Signal handling
    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    port = find_available_port()

    print(f"{Fore.GREEN}SYSTEM ONLINE -> http://localhost:{port}{Style.RESET_ALL}")

    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)