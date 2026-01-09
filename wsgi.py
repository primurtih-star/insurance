"""
WSGI entry point for production deployment.
This file exposes the Flask application object as `application`,
which is required by WSGI servers like Gunicorn or uWSGI.
"""

from app import app as application  # Import the Flask instance from app.py

# Optional: safety check for local debug
if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000)
