# api/__init__.py

def register_routes(app, limiter=None):
    # import API blueprints (lazy import)
    from api.routes.admin_routes import bp as admin_bp
    from api.routes.analysis_routes import bp as analysis_bp
    from api.routes.compare_routes import bp as compare_bp
    from api.routes.dataset_routes import bp as dataset_bp
    from api.routes.export_routes import bp as export_bp
    from api.routes.model_routes import bp as model_bp
    from api.routes.patient_routes import bp as patient_bp
    from api.routes.prediction_routes import bp as prediction_bp
    from api.routes.risk_routes import bp as risk_bp

    # register API blueprints (alphabetical)
    app.register_blueprint(admin_bp, url_prefix="/api/admin")
    app.register_blueprint(analysis_bp, url_prefix="/api/analysis")
    app.register_blueprint(compare_bp, url_prefix="/api/compare")
    app.register_blueprint(dataset_bp, url_prefix="/api/dataset")
    app.register_blueprint(export_bp, url_prefix="/api/export")
    app.register_blueprint(model_bp, url_prefix="/api/model")
    app.register_blueprint(patient_bp, url_prefix="/api/patient")
    app.register_blueprint(prediction_bp, url_prefix="/api/predict")
    app.register_blueprint(risk_bp, url_prefix="/api/risk")


def register_api(app):
    # legacy helper to expose prediction under /api
    from api.routes.prediction_routes import bp as prediction_bp
    app.register_blueprint(prediction_bp, url_prefix="/api")
    return app