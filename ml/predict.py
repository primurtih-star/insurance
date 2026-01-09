#!/usr/bin/env python3
"""
ml/predict.py

Prediction utility for insurance charges.
- Tries to load a trained model (and optional metadata) from disk.
- If model is missing or fails, falls back to a deterministic dummy logic.
- Supports single interactive prediction and batch CSV prediction.

Usage examples:
  python ml/predict.py --test-single --age 30 --bmi 25.5 --smoker yes
  python ml/predict.py --input data/new_data.csv --output data/results.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# ---------- Config ----------
MODEL_PATH_DEFAULT = Path("data/model/rf_model.joblib")
META_SUFFIX = "_meta.joblib"
LOG_FMT = "%(asctime)s %(levelname)s %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("ml.predict")


# ---------- Helpers ----------
def load_trained_model(model_path: Path = MODEL_PATH_DEFAULT) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Try to load a model and optional metadata (encoders, feature_names).
    Returns (model_or_None, meta_or_None).
    """
    if not model_path.exists():
        logger.info("Model file not found at %s", model_path)
        return None, None

    try:
        model = joblib.load(model_path)
        logger.info("Loaded model from %s", model_path)
    except Exception as exc:
        logger.warning("Failed to load model: %s", exc)
        return None, None

    # Try to load metadata saved alongside model (same stem + META_SUFFIX)
    meta_path = model_path.with_name(f"{model_path.stem}{META_SUFFIX}")
    meta = None
    if meta_path.exists():
        try:
            meta = joblib.load(meta_path)
            logger.info("Loaded model metadata from %s", meta_path)
        except Exception as exc:
            logger.warning("Failed to load model metadata: %s", exc)
            meta = None

    return model, meta


def normalize_smoker_column(series: pd.Series) -> pd.Series:
    """
    Normalize smoker column values to 0/1.
    Accepts numeric 0/1, strings 'yes'/'no' (case-insensitive), or booleans.
    Unknown values are mapped to 0.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).clip(0, 1)
    
    mapping = {
        "yes": 1, "y": 1, "true": 1, "1": 1, 
        "no": 0, "n": 0, "false": 0, "0": 0
    }
    s = series.astype(str).str.strip().str.lower().map(mapping)
    return s.fillna(0).astype(int)


def prepare_input_dataframe(age: float, bmi: float, smoker: Any) -> pd.DataFrame:
    """Create a single-row DataFrame with proper dtypes and normalized smoker."""
    df = pd.DataFrame({"age": [age], "bmi": [bmi], "smoker": [smoker]})
    df["smoker"] = normalize_smoker_column(df["smoker"])
    return df


def dummy_logic_predict(age: float, bmi: float, smoker_val: int) -> float:
    """
    Deterministic fallback logic for prediction.
    Keep it simple and reproducible (no randomness).
    """
    base = (age * 12.0) + (bmi * 22.0)
    if int(smoker_val) == 1:
        base *= 1.8
    # Add small deterministic offset based on bmi to avoid identical outputs
    offset = (bmi % 5) * 10.0
    return round(float(base + offset), 2)


def predict_with_model(model: Any, df: pd.DataFrame, meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Predict using the loaded model. If metadata contains encoders, try to apply them.
    Expects df to contain the same feature columns used during training.
    """
    X = df.copy()

    # If metadata contains encoders, attempt to inverse-transform or align features.
    if meta and "encoders" in meta:
        encoders = meta["encoders"]
        for col, enc in encoders.items():
            if col in X.columns:
                try:
                    X[col] = enc.transform(X[col].astype(str))
                except Exception:
                    # fallback: try mapping common yes/no for smoker
                    if col == "smoker":
                        X[col] = normalize_smoker_column(X[col])
                    else:
                        X[col] = X[col].astype(str)
    else:
        # Basic normalization for smoker if present
        if "smoker" in X.columns:
            X["smoker"] = normalize_smoker_column(X["smoker"])

    # Ensure numeric dtype for model input
    for c in X.columns:
        if X[c].dtype == object:
            try:
                X[c] = pd.to_numeric(X[c], errors="coerce")
            except Exception:
                pass
    X = X.fillna(0)

    # Model predict (handle scikit-learn style)
    preds = model.predict(X)
    return np.array(preds, dtype=float)


# ---------- CLI Functions ----------
def predict_single(age: float, bmi: float, smoker: Any, model_path: Path = MODEL_PATH_DEFAULT) -> float:
    """
    Predict a single record. Prefer trained model; fallback to dummy logic.
    """
    model, meta = load_trained_model(model_path)
    input_df = prepare_input_dataframe(age, bmi, smoker)

    if model is not None:
        try:
            preds = predict_with_model(model, input_df, meta)
            logger.info("Prediction using trained model")
            return round(float(preds[0]), 2)
        except Exception as exc:
            logger.warning("Model prediction failed: %s. Falling back to dummy logic.", exc)

    # fallback
    smoker_val = int(normalize_smoker_column(pd.Series([smoker]))[0])
    logger.info("Using dummy logic for prediction")
    return dummy_logic_predict(age, bmi, smoker_val)


def predict_batch(infile: Path, outfile: Path, model_path: Path = MODEL_PATH_DEFAULT) -> None:
    """
    Read CSV infile, predict, and write outfile with new column 'predicted_charges'.
    """
    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    logger.info("Reading input CSV: %s", infile)
    df = pd.read_csv(infile)

    required_cols = ["age", "bmi", "smoker"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Input CSV missing required columns: {missing}")

    model, meta = load_trained_model(model_path)

    if model is not None:
        logger.info("Using trained model for batch prediction")
        # Normalize smoker column first
        df["smoker"] = normalize_smoker_column(df["smoker"])
        # Select only required columns for prediction to avoid shape mismatch if extra cols exist
        preds = predict_with_model(model, df[required_cols], meta)
    else:
        logger.info("No trained model found; using dummy logic for batch prediction")
        df["smoker"] = normalize_smoker_column(df["smoker"])
        preds = df.apply(lambda r: dummy_logic_predict(float(r["age"]), float(r["bmi"]), int(r["smoker"])), axis=1).to_numpy()

    df_out = df.copy()
    df_out["predicted_charges"] = np.round(preds.astype(float), 2)
    
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(outfile, index=False)
    logger.info("Batch prediction finished. Results saved to %s", outfile)


# ---------- CLI Entrypoint ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insurance Prediction Tool")
    
    # Batch mode
    parser.add_argument("--input", help="Path to input CSV for batch predictions")
    parser.add_argument("--output", help="Path to output CSV for batch predictions")
    
    # Single test mode
    parser.add_argument("--test-single", action="store_true", help="Run a single test prediction")
    parser.add_argument("--age", type=float, help="Age for single prediction")
    parser.add_argument("--bmi", type=float, help="BMI for single prediction")
    parser.add_argument("--smoker", help="Smoker flag for single prediction (yes/no/1/0)")
    
    # Optional model path override
    parser.add_argument("--model-path", default=str(MODEL_PATH_DEFAULT), help="Path to trained model joblib")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)

    try:
        # Mode 1: Single Prediction
        if args.test_single:
            if args.age is None or args.bmi is None or args.smoker is None:
                logger.error("For --test-single you must provide --age, --bmi and --smoker")
                raise SystemExit(2)
            
            result = predict_single(args.age, args.bmi, args.smoker, model_path=model_path)
            print(f"Predicted charges: {result}")
            return

        # Mode 2: Batch Prediction
        if args.input and args.output:
            infile = Path(args.input)
            outfile = Path(args.output)
            predict_batch(infile, outfile, model_path=model_path)
            print(f"Batch prediction saved to: {outfile}")
            return

        # If no valid mode provided
        logger.info("No mode selected. Use --test-single or --input/--output for batch.")
        # Print help to guide the user
        parser = argparse.ArgumentParser() 
        parser.print_help()
        raise SystemExit(2)

    except FileNotFoundError as exc:
        logger.error(str(exc))
        raise SystemExit(1)
    except KeyError as exc:
        logger.error(str(exc))
        raise SystemExit(1)
    except Exception as exc:
        logger.exception("Unexpected error during prediction: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()