#!/usr/bin/env python3
"""
ml/feature_importance.py

Extract and visualize feature importance from scikit-learn models and pipelines.

Usage:
  python ml/feature_importance.py --model data/model/rf_model.joblib
  python ml/feature_importance.py --top-n 10 --output ui/img/plots/imp.png
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ---------- Config ----------
LOG_FMT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("ml.feature_importance")

# Use non-interactive backend for servers
plt.switch_backend("Agg")

DEFAULT_MODEL_PATH = Path("data/model/rf_model.joblib")
DEFAULT_OUTPUT_IMG = Path("ui/img/plots/feature_importance.png")


# ---------- Helpers ----------
def _safe_get_feature_names_from_transformer(transformer: Any, input_features: Optional[Sequence[str]] = None) -> List[str]:
    """
    Best-effort extraction of feature names from a fitted transformer.
    Tries get_feature_names_out, get_feature_names, and falls back to input_features.
    """
    if transformer is None:
        return []

    # Try get_feature_names_out (newer sklearn)
    try:
        if hasattr(transformer, "get_feature_names_out"):
            if input_features is not None:
                try:
                    return list(transformer.get_feature_names_out(input_features))
                except TypeError:
                    return list(transformer.get_feature_names_out())
            return list(transformer.get_feature_names_out())
    except Exception:
        logger.debug("get_feature_names_out failed for %s", type(transformer).__name__, exc_info=True)

    # Try older get_feature_names
    try:
        if hasattr(transformer, "get_feature_names"):
            return list(transformer.get_feature_names())
    except Exception:
        logger.debug("get_feature_names failed for %s", type(transformer).__name__, exc_info=True)

    # Fallback to feature_names_in_ or input_features
    try:
        if hasattr(transformer, "feature_names_in_"):
            return list(transformer.feature_names_in_)
    except Exception:
        pass

    return list(input_features or [])


def get_feature_names_from_estimator(model: Any, input_columns: Optional[Sequence[str]] = None) -> List[str]:
    """
    Attempt to extract feature names from a model or pipeline.
    Handles Pipeline -> ColumnTransformer -> encoders, and direct transformers.
    Returns an ordered list of feature names or empty list if not found.
    """
    # If pipeline, try to find a preprocessor step (ColumnTransformer) first
    if isinstance(model, Pipeline):
        # Try to find ColumnTransformer in pipeline steps
        for name, step in model.named_steps.items():
            if isinstance(step, ColumnTransformer):
                # ColumnTransformer: build feature names from its transformers
                feature_names: List[str] = []
                for trans_name, trans, cols in step.transformers_:
                    if trans_name == "remainder":
                        continue
                    # Normalize cols to list of names
                    if isinstance(cols, (list, tuple)):
                        cols_list = list(cols)
                    else:
                        # If cols is slice or array of indices, try to map using input_columns
                        try:
                            cols_list = [input_columns[i] for i in cols] if input_columns is not None else []
                        except Exception:
                            cols_list = [str(cols)]
                    # Get names from transformer
                    names = _safe_get_feature_names_from_transformer(trans, cols_list)
                    if names:
                        feature_names.extend(names)
                    else:
                        # If transformer didn't expand names, keep original column names
                        feature_names.extend(cols_list)
                # handle remainder passthrough
                if getattr(step, "remainder", None) == "passthrough" and input_columns is not None:
                    transformed_cols = [c for c in feature_names if c in (input_columns or [])]
                    passthrough = [c for c in input_columns if c not in transformed_cols]
                    feature_names.extend(passthrough)
                return feature_names

        # If no ColumnTransformer found, try each step's get_feature_names_out
        for _, step in model.named_steps.items():
            names = _safe_get_feature_names_from_transformer(step, input_columns)
            if names:
                return names

        # As last resort, try final estimator's feature_names_in_
        final_est = model.steps[-1][1]
        if hasattr(final_est, "feature_names_in_"):
            return list(getattr(final_est, "feature_names_in_"))

    # If model is a ColumnTransformer directly
    if isinstance(model, ColumnTransformer):
        feature_names: List[str] = []
        for trans_name, trans, cols in model.transformers_:
            if trans_name == "remainder":
                continue
            if isinstance(cols, (list, tuple)):
                cols_list = list(cols)
            else:
                try:
                    cols_list = [input_columns[i] for i in cols] if input_columns is not None else []
                except Exception:
                    cols_list = [str(cols)]
            names = _safe_get_feature_names_from_transformer(trans, cols_list)
            feature_names.extend(names or cols_list)
        if getattr(model, "remainder", None) == "passthrough" and input_columns is not None:
            transformed_cols = [c for c in feature_names if c in (input_columns or [])]
            passthrough = [c for c in input_columns if c not in transformed_cols]
            feature_names.extend(passthrough)
        return feature_names

    # If estimator has feature_names_in_
    if hasattr(model, "feature_names_in_"):
        try:
            return list(getattr(model, "feature_names_in_"))
        except Exception:
            pass

    # Fallback: return input_columns if provided
    return list(input_columns or [])


# ---------- Core Logic ----------
def extract_importances(model_path: Path, input_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Load model and extract feature importances (or coefficients).
    Returns a DataFrame sorted by absolute importance with columns: Feature, Importance/Coefficient.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    logger.info("Loading model from %s...", model_path)
    model = joblib.load(model_path)

    # Identify estimator (if pipeline, last step is estimator)
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]

    # Extract raw importance values
    importances = None
    importance_label = "importance"

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(getattr(estimator, "feature_importances_"), dtype=float)
        importance_label = "importance"
    elif hasattr(estimator, "coef_"):
        coef = getattr(estimator, "coef_")
        coef_arr = np.asarray(coef, dtype=float)
        # If multi-output or multiclass, try to reduce to a single vector (first class) but warn
        if coef_arr.ndim > 1:
            logger.debug("Estimator coef_ is multi-dimensional; taking first row as representative.")
            importances = coef_arr[0]
        else:
            importances = coef_arr
        importance_label = "coefficient"
    else:
        logger.warning("Model does not have 'feature_importances_' or 'coef_'; cannot extract importances.")
        return pd.DataFrame(columns=["Feature", "Value"])

    # Resolve feature names
    feature_names = get_feature_names_from_estimator(model, input_columns=input_columns)

    # If mismatch, generate deterministic generic names
    if len(feature_names) != len(importances):
        logger.warning(
            "Feature name count (%d) != importance count (%d). Falling back to generic names.",
            len(feature_names), len(importances)
        )
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    df = pd.DataFrame({"Feature": feature_names, "Value": importances})
    df["AbsValue"] = np.abs(df["Value"])
    df = df.sort_values(by="AbsValue", ascending=False).drop(columns=["AbsValue"]).reset_index(drop=True)
    df.rename(columns={"Value": importance_label.capitalize()}, inplace=True)
    return df


def plot_feature_importance(df: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    """
    Plot horizontal bar chart of feature importances using Matplotlib.
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame provided to plot_feature_importance; skipping.")
        return

    plot_df = df.head(top_n).copy()
    plot_df = plot_df.iloc[::-1]  # reverse for horizontal bar chart

    val_col = plot_df.columns[1]  # second column is the numeric importance/coefficient

    # Figure size: height depends on number of bars
    fig_height = max(4, 0.5 * len(plot_df) + 1)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Color: positive green, negative red (useful for coefficients)
    values = plot_df[val_col].astype(float).values
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in values]

    bars = ax.barh(plot_df["Feature"], values, color=colors, height=0.7)

    ax.set_title(f"Top {len(plot_df)} Feature {val_col}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(val_col, fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Remove top/right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add labels at end of bars
    max_val = np.max(np.abs(values)) if values.size else 0.0
    offset = max_val * 0.02 if max_val else 0.01
    for bar in bars:
        w = bar.get_width()
        x = w + offset if w >= 0 else w - offset
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=9, color="#333333")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", output_path)


# ---------- CLI Entrypoint ----------
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and Plot Feature Importance")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to trained .joblib model")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_IMG), help="Path to save the plot image")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top features to plot")
    parser.add_argument("--input-cols", type=str, default=None, help="Optional comma-separated input column names (used as fallback)")
    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output)
    input_cols = args.input_cols.split(",") if args.input_cols else None

    try:
        df_imp = extract_importances(model_path, input_columns=input_cols)
        if df_imp.empty:
            logger.error("No importances extracted from model %s", model_path)
            raise SystemExit(1)

        # Print top-n to stdout for quick inspection
        print("-" * 40)
        print(f"Top {args.top_n} Features:")
        print("-" * 40)
        print(df_imp.head(args.top_n).to_string(index=False))
        print("-" * 40)

        plot_feature_importance(df_imp, output_path, top_n=args.top_n)
    except Exception as exc:
        logger.exception("An error occurred while extracting/plotting feature importances: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()