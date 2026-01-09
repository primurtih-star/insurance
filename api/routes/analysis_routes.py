from flask import Blueprint, jsonify

bp = Blueprint("analysis_routes", __name__)

@bp.route("/", methods=["GET"])
def analysis_get():
    return jsonify({"status": "ok", "message": "Analysis route connected"})
