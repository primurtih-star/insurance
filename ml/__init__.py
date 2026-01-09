"""
ml package initializer.

Expose a small, stable public API and perform lightweight, safe imports.
"""
from __future__ import annotations

import logging
from typing import Optional

# Prevent "No handler found" warnings when user doesn't configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
logger = logging.getLogger(__name__)

# Package metadata (single source of truth)
__version__ = "1.0.0"

# ---------------------------------------------------------------------
# Optional components imported safely
# ---------------------------------------------------------------------
PredictionEngine: Optional[type] = None
engine = None
train_model = None
create_inference_pipeline = None
predict_single = None
predict_batch = None
extract_importances = None
plot_feature_importance = None
DataPreprocessor = None
run_preprocessing = None
ModelTrainer = None
ModelEvaluator = None
run_retraining_pipeline = None

# Inference engine (singleton) if available
try:
    from .engine import PredictionEngine, engine  # type: ignore
except Exception as exc:  # log any error to help debugging
    logger.debug("ml.engine not available: %s", exc)

# Training orchestrator
try:
    from .train import train_model  # type: ignore
except Exception as exc:
    logger.debug("ml.train not available: %s", exc)

# Pipeline builder
try:
    from .pipeline.pipeline_builder import create_inference_pipeline  # type: ignore
except Exception as exc:
    logger.debug("ml.pipeline.pipeline_builder not available: %s", exc)

# Prediction helpers
try:
    from .predict import predict_single, predict_batch  # type: ignore
except Exception as exc:
    logger.debug("ml.predict not available: %s", exc)

# Feature importance helpers
try:
    from .feature_importance import extract_importances, plot_feature_importance  # type: ignore
except Exception as exc:
    logger.debug("ml.feature_importance not available: %s", exc)

# Preprocessing helpers
try:
    from .preprocess import DataPreprocessor, run_preprocessing  # type: ignore
except Exception as exc:
    logger.debug("ml.preprocess not available: %s", exc)

# Trainer / evaluator wrappers
try:
    from .trainer import ModelTrainer  # type: ignore
except Exception:
    pass

try:
    from .evaluate import ModelEvaluator  # type: ignore
except Exception as exc:
    logger.debug("ml.evaluate not available: %s", exc)

# Retraining pipeline
try:
    from .retrain import run_retraining_pipeline  # type: ignore
except Exception as exc:
    logger.debug("ml.retrain not available: %s", exc)

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
__all__ = (
    "__version__",
    # Inference
    "PredictionEngine",
    "engine",
    "predict_single",
    "predict_batch",
    # Pipeline
    "create_inference_pipeline",
    # Training
    "train_model",
    "ModelTrainer",
    # Evaluation
    "ModelEvaluator",
    # Feature importance
    "extract_importances",
    "plot_feature_importance",
    # Preprocessing / Retraining
    "DataPreprocessor",
    "run_preprocessing",
    "run_retraining_pipeline",
)