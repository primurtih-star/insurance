from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)

__all__ = ["regression_metrics", "regression_report"]


def _to_numpy(y: Sequence[Any]) -> np.ndarray:
    """
    Convert array-like to 1-D numpy array; raise on invalid shapes.
    Accepts lists, numpy arrays, pandas Series/DataFrame (squeezes single-column).
    """
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.squeeze()
    arr = np.asarray(y)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.squeeze()
    if arr.ndim != 1:
        raise ValueError("y must be a 1-dimensional array-like")
    return arr


def regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    sample_weight: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Compute common regression metrics and return them as native floats.

    Returned keys:
      - mse: mean squared error
      - rmse: root mean squared error
      - mae: mean absolute error
      - r2: R-squared
      - explained_variance: explained variance score

    sample_weight is forwarded to sklearn metrics where supported.
    """
    y_true_arr = _to_numpy(y_true)
    y_pred_arr = _to_numpy(y_pred)

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    mse = float(mean_squared_error(y_true_arr, y_pred_arr, sample_weight=sample_weight))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr, sample_weight=sample_weight))
    r2 = float(r2_score(y_true_arr, y_pred_arr, sample_weight=sample_weight))
    ev = float(explained_variance_score(y_true_arr, y_pred_arr, sample_weight=sample_weight))

    metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "explained_variance": ev}
    logger.debug("regression_metrics computed: %s", metrics)
    return metrics


def regression_report(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    sample_weight: Optional[Sequence[float]] = None,
    round_digits: Optional[int] = 4,
    prefix: Optional[str] = None,
) -> Dict[str, float]:
    """
    Convenience wrapper that returns (optionally rounded) regression metrics.

    Args:
      - y_true, y_pred: array-like true and predicted values
      - sample_weight: optional sample weights
      - round_digits: number of decimal places to round metrics to (None = no rounding)
      - prefix: optional string to prepend to each metric key (e.g., "val_")

    Returns:
      dict of metrics (possibly rounded and with optional prefix).
    """
    metrics = regression_metrics(y_true, y_pred, sample_weight=sample_weight)
    if round_digits is not None:
        metrics = {k: round(v, int(round_digits)) for k, v in metrics.items()}
    if prefix:
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
    logger.debug("regression_report: %s", metrics)
    return metrics