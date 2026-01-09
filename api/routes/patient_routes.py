from flask import Blueprint, jsonify

bp = Blueprint("patients_routes", __name__)

@bp.route("/", methods=["GET"])
def export_status():
    return jsonify({"route": "patients_routes", "status": "ok"})