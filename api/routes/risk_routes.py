from flask import Blueprint, jsonify

bp = Blueprint("risk_routes", __name__)

@bp.route("/", methods=["GET"])
def export_status():
    return jsonify({"route": "risk_routes", "status": "ok"})