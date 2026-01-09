from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ml.utils.cleaner import basic_clean
from ml.utils.encoding import detect_column_types, build_encoder
from ml.utils.metrics import regression_metrics
from ml.utils.scaler import build_scaler
from ml.utils.validator import validate_dataset

# Paths (Pathlib)
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = (ROOT / "logs").resolve()
MODEL_DIR = (ROOT / "models").resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_LOG = LOG_DIR / "training_logs.txt"
RETRAIN_HISTORY = LOG_DIR / "retraining_history.json"

# Lightweight logger (use app logger in real app)
import logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
                                           datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class PipelineBuilder:
    """
    Unified builder for:
      - loading CSV
      - cleaning
      - building preprocessing pipeline (encoding + scaling)
      - training models (DecisionTree, RandomForest, optional GradientBoosting)
      - saving artifacts and training metadata
    """

    def __init__(
        self,
        csv_path: str | Path,
        target_col: str = "risk_score",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.target_col = target_col
        self.test_size = float(test_size)
        self.random_state = int(random_state)

        self.df: Optional[pd.DataFrame] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.scaler = build_scaler()
        self.models: Dict[str, Pipeline] = {}

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

    # -------------------------
    # Data loading & preparation
    # -------------------------
    def load_data(self) -> pd.DataFrame:
        """Load CSV into DataFrame and store it on the instance."""
        df = pd.read_csv(self.csv_path)
        logger.info("Loaded CSV %s rows=%d", self.csv_path.name, len(df))
        self.df = df
        return df

    def prepare(self) -> ColumnTransformer:
        """
        Validate dataset, clean, detect column types, and build encoder/preprocessor.
        Returns the preprocessor (ColumnTransformer).
        """
        if self.df is None:
            self.load_data()

        validate_dataset(self.df, self.target_col)
        self.df = basic_clean(self.df)

        X = self.df.drop(columns=[self.target_col])
        numeric_cols, cat_cols = detect_column_types(X)
        logger.info("Detected numeric_cols=%s, cat_cols=%s", numeric_cols, cat_cols)

        # If a custom encoder builder exists, use it; otherwise fallback to simple ColumnTransformer
        try:
            self.preprocessor = build_encoder(X, cat_cols)
        except Exception:
            # Fallback: simple imputer + onehot for categorical, scaler for numeric
            numeric_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
            )
            categorical_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                       ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
            )
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_cols),
                    ("cat", categorical_transformer, cat_cols),
                ],
                remainder="drop",
            )

        return self.preprocessor

    # -------------------------
    # Pipeline construction
    # -------------------------
    def build_model_pipelines(self) -> Dict[str, Pipeline]:
        """
        Build sklearn Pipelines for supported models.
        Returns a dict mapping short name -> Pipeline.
        """
        if self.preprocessor is None:
            raise RuntimeError("preprocessor not prepared. Call prepare() first.")

        rf_pipe = Pipeline(
            steps=[
                ("preproc", self.preprocessor),
                ("scaler", self.scaler),
                ("est", RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)),
            ]
        )

        dt_pipe = Pipeline(
            steps=[
                ("preproc", self.preprocessor),
                ("scaler", self.scaler),
                ("est", DecisionTreeRegressor(random_state=self.random_state)),
            ]
        )

        gb_pipe = Pipeline(
            steps=[
                ("preproc", self.preprocessor),
                ("scaler", self.scaler),
                ("est", GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)),
            ]
        )

        self.models = {"rf": rf_pipe, "dt": dt_pipe, "gb": gb_pipe}
        return self.models

    # -------------------------
    # Training & persistence
    # -------------------------
    def train(self, cv: bool = False, grid_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train available pipelines and save artifacts.

        Args:
            cv: if True, run a small GridSearchCV for RandomForest (or provided model).
            grid_params: optional dict of param_grid per model name, e.g. {"rf": {...}}

        Returns:
            results dict containing metrics and metadata per model.
        """
        if self.df is None:
            self.load_data()

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        pipes = self.build_model_pipelines()
        results: Dict[str, Any] = {}

        for name, pipe in pipes.items():
            start = time.time()
            model_to_save = None

            if cv and name in (grid_params or {}):
                param_grid = grid_params[name]
                logger.info("Running GridSearchCV for %s with params=%s", name, param_grid)
                gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, scoring="neg_mean_squared_error")
                gs.fit(X_train, y_train)
                model_to_save = gs.best_estimator_
                logger.info("GridSearchCV best_params for %s: %s", name, gs.best_params_)
            else:
                model_to_save = pipe.fit(X_train, y_train)

            y_pred = model_to_save.predict(X_test)
            metrics = regression_metrics(y_test, y_pred)
            took = time.time() - start
            results[name] = {"metrics": metrics, "time_seconds": round(took, 3)}
            logger.info("Trained %s in %.2fs metrics=%s", name, took, metrics)

            # Persist model artifact and metadata
            model_path = MODEL_DIR / f"{name}_model.joblib"
            joblib.dump(model_to_save, model_path)
            logger.info("Saved model %s -> %s", name, model_path)

            meta = {
                "model": name,
                "path": str(model_path),
                "metrics": metrics,
                "trained_at": time.time(),
            }
            meta_path = MODEL_DIR / f"{name}_metadata.json"
            meta_path.write_text(json.dumps(meta, indent=2))
            logger.info("Saved metadata %s", meta_path)

        # Append retraining history
        self._append_retrain_history(results)
        return results

    def _append_retrain_history(self, results: Dict[str, Any]) -> None:
        entry = {"csv": self.csv_path.name, "timestamp": time.time(), "results": results}
        history = []
        if RETRAIN_HISTORY.exists():
            try:
                history = json.loads(RETRAIN_HISTORY.read_text())
            except Exception:
                history = []
        history.append(entry)
        RETRAIN_HISTORY.write_text(json.dumps(history, indent=2))
        logger.info("Appended retrain history for %s", self.csv_path.name)

    # -------------------------
    # Convenience: create inference pipeline
    # -------------------------
    @staticmethod
    def create_inference_pipeline(
        model_type: str = "random_forest", n_estimators: int = 200, random_state: int = 42
    ) -> Pipeline:
        """
        Build a standalone inference pipeline (useful for deployment).
        """
        NUMERIC_FEATURES = ["age", "bmi", "children"]
        CATEGORICAL_FEATURES = ["sex", "smoker", "region"]

        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
        categorical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                   ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        )

        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, NUMERIC_FEATURES), ("cat", categorical_transformer, CATEGORICAL_FEATURES)],
            remainder="drop",
        )

        if model_type == "gradient_boosting":
            model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

        return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])