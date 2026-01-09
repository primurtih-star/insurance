#!/usr/bin/env python3
"""
evaluate.py

Evaluate saved models on a CSV (with true target). Outputs metrics JSON and
a CSV comparing predictions from two models (rf and dt by default).

Usage:
  python evaluate.py --csv data/test.csv --out data/eval_out.csv --target charges
"""
from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- Config ----------
LOG_FMT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("evaluate")

DEFAULT_MODEL_DIR = Path("data/model")
ARTIFACTS_DIR = Path("ml/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Utilities ----------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute common regression metrics and return a serializable dict with lowercase keys."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": round(mae, 4), "mse": round(mse, 4), "rmse": round(rmse, 4), "r2": round(r2, 6)}


def load_model(path: Path) -> Any:
    """Load a joblib model, raise FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        model = joblib.load(path)
        logger.info("Loaded model: %s", path)
        return model
    except Exception as exc:
        logger.exception("Failed to load model %s: %s", path, exc)
        raise


# ---------- Core evaluation ----------
def eval_model_file(
    model_path: Path,
    df: pd.DataFrame,
    target_col: str,
    *,
    force_numeric: bool = False,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate a single model file on dataframe df.
    Returns (metrics_dict, y_true, y_pred).
    - If the model is a pipeline, pass raw X and let pipeline handle preprocessing.
    - If force_numeric True, coerce features to numeric (use with caution).
    """
    model = load_model(model_path)

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in input dataframe")

    df_clean = df.copy()
    X = df_clean.drop(columns=[target_col], errors="ignore")
    y_true = df_clean[target_col].values

    # If user explicitly requests numeric coercion, do it; otherwise pass X as-is.
    if force_numeric:
        X_proc = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    else:
        X_proc = X

    try:
        y_pred = model.predict(X_proc)
    except Exception as exc:
        logger.exception("Model prediction failed for %s: %s", model_path, exc)
        raise RuntimeError(f"Prediction failed for model {model_path}: {exc}") from exc

    metrics = regression_metrics(y_true, y_pred)
    return metrics, np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def _atomic_write_json(path: Path, data: Dict) -> None:
    """Write JSON atomically to avoid partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.flush()
    tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def compare_and_save(
    csv_path: Path,
    out_csv: Path,
    target_col: str = "risk_score",
    rf_model: Optional[Path] = None,
    dt_model: Optional[Path] = None,
    *,
    force_numeric: bool = False,
) -> Dict[str, Dict]:
    """
    Compare two models (rf and dt) on csv_path and save a CSV with side-by-side predictions.
    Returns a dict with metrics for both models.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded test CSV: %s (rows=%d)", csv_path, len(df))

    rf_model = rf_model or (DEFAULT_MODEL_DIR / "rf_model.joblib")
    dt_model = dt_model or (DEFAULT_MODEL_DIR / "dt_model.joblib")

    results: Dict[str, Dict] = {}

    # Evaluate RF
    try:
        rf_metrics, y_true, rf_pred = eval_model_file(rf_model, df, target_col, force_numeric=force_numeric)
        results["rf"] = rf_metrics
        logger.info("RF metrics: %s", rf_metrics)
    except Exception as exc:
        logger.warning("RF evaluation failed: %s", exc)
        rf_pred = np.full(len(df), np.nan)
        results["rf"] = {"error": str(exc)}

    # Evaluate DT
    try:
        dt_metrics, _, dt_pred = eval_model_file(dt_model, df, target_col, force_numeric=force_numeric)
        results["dt"] = dt_metrics
        logger.info("DT metrics: %s", dt_metrics)
    except Exception as exc:
        logger.warning("DT evaluation failed: %s", exc)
        dt_pred = np.full(len(df), np.nan)
        results["dt"] = {"error": str(exc)}

    # Write comparison CSV
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = df.copy()
    df_out[f"{target_col}_rf_pred"] = rf_pred
    df_out[f"{target_col}_dt_pred"] = dt_pred
    df_out.to_csv(out_csv, index=False)
    logger.info("Wrote comparison CSV to %s", out_csv)

    # Save aggregated metrics to artifacts for dashboard with timestamp
    ts = time.strftime("%Y%m%d_%H%M%S")
    metrics_out = {
        "timestamp": ts,
        "source_csv": str(csv_path),
        "target_col": target_col,
        "results": results,
    }
    metrics_path = ARTIFACTS_DIR / f"metrics_{ts}.json"
    _atomic_write_json(metrics_path, metrics_out)
    logger.info("Saved aggregated metrics to %s", metrics_path)

    return results


# ---------- CLI ----------
def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate saved models on a CSV and produce comparison CSV + metrics JSON.")
    parser.add_argument("--csv", required=True, help="Path to test CSV (must include target column)")
    parser.add_argument("--out", required=True, help="Path to output CSV with predictions")
    parser.add_argument("--target", default="risk_score", help="Target column name in CSV")
    parser.add_argument("--rf-model", help="Path to RF model joblib (optional)")
    parser.add_argument("--dt-model", help="Path to DT model joblib (optional)")
    parser.add_argument("--force-numeric", action="store_true", help="Coerce features to numeric before prediction (use with caution)")
    return parser.parse_args()


def main():
    args = _parse_args()
    csv_path = Path(args.csv)
    out_csv = Path(args.out)
    rf_model = Path(args.rf_model) if args.rf_model else None
    dt_model = Path(args.dt_model) if args.dt_model else None

    try:
        results = compare_and_save(
            csv_path,
            out_csv,
            target_col=args.target,
            rf_model=rf_model,
            dt_model=dt_model,
            force_numeric=args.force_numeric,
        )
        # Minimal CLI output (JSON) so caller can parse
        print(json.dumps({"status": "ok", "results": results}, indent=2))
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()