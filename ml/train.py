# imports
from __future__ import annotations
import argparse
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# third_party
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

# logger
LOG_FMT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("train")

# helpers
def read_csv(path: Path) -> pd.DataFrame:
    """Load CSV into DataFrame or raise FileNotFoundError."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)

def infer_task(y: pd.Series, explicit: Optional[str] = None) -> str:
    """
    Infer task type from target series or use explicit override.
    Heuristic: floats -> regression; ints/objects -> classification unless many unique values.
    """
    if explicit in ("classification", "regression"):
        return explicit
    if pd.api.types.is_float_dtype(y):
        return "regression"
    if pd.api.types.is_integer_dtype(y) or pd.api.types.is_object_dtype(y):
        return "regression" if y.nunique() > 20 else "classification"
    return "regression"

def encode_categoricals(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label-encode object/category columns. Returns encoded DataFrame and dict of encoders.
    Encoding to string first avoids mixed-type issues.
    """
    encoders: Dict[str, LabelEncoder] = {}
    X_out = X.copy()
    cat_cols = X_out.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X_out[col] = le.fit_transform(X_out[col].astype(str))
        encoders[col] = le
        logger.debug("Encoded column %s with LabelEncoder", col)
    return X_out, encoders

def save_model_with_meta(model: object, encoders: Dict[str, LabelEncoder], feature_names, out_path: Path) -> Path:
    """
    Save model artifact and metadata. Returns final model path.
    Model file will be suffixed with timestamp to avoid accidental overwrite.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = out_path.suffix or ".joblib"
    final_model_path = out_path.with_name(f"{out_path.stem}_v{timestamp}{suffix}")

    joblib.dump(model, final_model_path)
    logger.info("Model saved to %s", final_model_path)

    meta = {
        "saved_at": time.time(),
        "timestamp": timestamp,
        "model_file": str(final_model_path.name),
        "feature_names": list(feature_names),
        "encoders": {
            k: {"classes": getattr(v, "classes_", None).tolist() if hasattr(v, "classes_") else None}
            for k, v in encoders.items()
        },
    }
    meta_path = final_model_path.with_name(f"{final_model_path.stem}_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Model metadata saved to %s", meta_path)

    return final_model_path

def prepare_features(df: pd.DataFrame, target_col: str, feature_cols: Optional[list[str]] = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder]]:
    """Validate presence of target/features, encode categoricals, and return X, y, encoders."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset")
    y = df[target_col].copy()
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Feature columns not found: {missing}")
    X = df[feature_cols].copy()
    X_encoded, encoders = encode_categoricals(X)
    return X_encoded, y, encoders

def build_model(task: str, n_estimators: int = 200, max_depth: Optional[int] = None, random_state: int = 42):
    """Return an unfitted sklearn model according to task."""
    if task == "classification":
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

def train_and_evaluate(model, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, cv: bool = False) -> dict:
    """Fit model, compute train/test scores and optional CV. Returns results dict."""
    logger.info("Fitting model...")
    model.fit(X_train, y_train)

    results: dict = {}
    try:
        train_score = model.score(X_train, y_train)
        results["train_score"] = float(train_score)
        logger.info("Train score: %.4f", train_score)
    except Exception:
        logger.debug("Could not compute train score", exc_info=True)

    try:
        test_score = model.score(X_test, y_test)
        results["test_score"] = float(test_score)
        logger.info("Test score: %.4f", test_score)
    except Exception:
        logger.debug("Could not compute test score", exc_info=True)

    if cv:
        try:
            X_all = pd.concat([X_train, X_test], axis=0)
            y_all = pd.concat([y_train, y_test], axis=0)
            cv_scores = cross_val_score(model, X_all, y_all, cv=3, n_jobs=-1)
            results["cv_mean"] = float(np.mean(cv_scores))
            results["cv_std"] = float(np.std(cv_scores))
            logger.info("CV mean: %.4f std: %.4f", results["cv_mean"], results["cv_std"])
        except Exception:
            logger.warning("CV failed; skipping CV", exc_info=True)

    return results

# cli
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RandomForest model (classification or regression).")
    parser.add_argument("--csv", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", default="risk_score", help="Target column name")
    parser.add_argument("--features", nargs="+", help="Optional list of feature columns; default = all except target")
    parser.add_argument("--task", choices=["classification", "regression"], help="Force task type")
    parser.add_argument("--cv", action="store_true", help="Run light cross-validation")
    parser.add_argument("--model-out", default="models/rf_model.joblib", help="Output model path (joblib)")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=None, help="Max depth for trees")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.model_out)

    try:
        df = read_csv(csv_path)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return

    try:
        X, y, encoders = prepare_features(df, args.target, args.features)
    except KeyError as exc:
        logger.error("Preparation error: %s", exc)
        return

    task = infer_task(y, args.task)
    logger.info("Detected task: %s", task)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(task, n_estimators=args.n_estimators, max_depth=args.max_depth)
    results = train_and_evaluate(model, X_train, X_test, y_train, y_test, cv=args.cv)

    saved = save_model_with_meta(model, encoders, X.columns, out_path)
    results["model_path"] = str(saved)
    logger.info("Training finished. Results: %s", results)

    summary_path = out_path.with_suffix(".train_summary.json")
    summary = {"results": results, "trained_at": time.time(), "csv": str(csv_path)}
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info("Training summary saved to %s", summary_path)
    except Exception:
        logger.exception("Failed to write training summary")

    print(json.dumps(results))

if __name__ == "__main__":
    main()