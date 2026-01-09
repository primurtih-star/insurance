from flask import Blueprint, jsonify

bp = Blueprint("export_routes", __name__)

@bp.route("/", methods=["GET"])
def export_status():
    return jsonify({"route": "export_routes", "status": "ok"})
