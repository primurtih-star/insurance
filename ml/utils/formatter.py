from __future__ import annotations

import json
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "to_json_serializable",
    "format_prediction",
    "format_predictions",
    "dataframe_to_records",
    "pretty_print_metrics",
]


def to_json_serializable(value: Any) -> Any:
    """Convert common numpy/pandas/datetime/decimal types to native JSON types."""
    # pandas NA
    if value is pd.NA:
        return None

    # numpy scalar
    if isinstance(value, np.generic):
        try:
            return value.item()
        except Exception:
            return float(value)

    # numpy array
    if isinstance(value, np.ndarray):
        return [to_json_serializable(v) for v in value.tolist()]

    # pandas Series
    if isinstance(value, pd.Series):
        return [to_json_serializable(v) for v in value.tolist()]

    # pandas Timestamp / datetime / date
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()

    # Decimal
    if isinstance(value, Decimal):
        try:
            return float(value)
        except Exception:
            return str(value)

    # Containers
    if isinstance(value, dict):
        return {str(k): to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_serializable(v) for v in value]

    # None or NaN
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None

    # Primitive types that are JSON serializable
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _round_if_numeric(v: Any, digits: Optional[int]) -> Any:
    if digits is None:
        return v
    if isinstance(v, (int, float, np.floating, np.integer)):
        try:
            return round(float(v), int(digits))
        except Exception:
            return v
    return v


def format_prediction(
    pred: Any,
    *,
    id: Optional[Union[str, int]] = None,
    round_digits: Optional[int] = 4,
    value_key: str = "prediction",
) -> Dict[str, Any]:
    """
    Normalize a single prediction into a JSON-serializable dict.

    Accepts scalar, numpy scalar, dict-like, pandas Series, or single-row DataFrame.
    """
    if isinstance(pred, pd.DataFrame):
        if len(pred) != 1:
            raise ValueError("format_prediction expects a single-row DataFrame or scalar/Series/dict")
        pred = pred.iloc[0]

    if isinstance(pred, pd.Series):
        record = pred.to_dict()
    elif isinstance(pred, dict):
        record = dict(pred)
    else:
        record = {value_key: pred}

    # Round numeric values if requested
    if round_digits is not None:
        for k, v in list(record.items()):
            record[k] = _round_if_numeric(v, round_digits)

    # Convert to JSON-serializable types
    record = {str(k): to_json_serializable(v) for k, v in record.items()}

    if id is not None:
        record = {"id": to_json_serializable(id), **record}

    return record


def dataframe_to_records(
    df: pd.DataFrame,
    *,
    round_digits: Optional[int] = 4,
    include_index: bool = False,
    index_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convert DataFrame rows to JSON-serializable records preserving order.

    If include_index True, index is added as a column named index_name or "index".
    """
    if include_index:
        name = index_name or "index"
        df = df.copy()
        df[name] = df.index

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        if round_digits is not None:
            for k, v in list(rec.items()):
                rec[k] = _round_if_numeric(v, round_digits)
        rec = {str(k): to_json_serializable(v) for k, v in rec.items()}
        records.append(rec)
    return records


def format_predictions(
    batch: Union[pd.DataFrame, Iterable[Any]],
    *,
    id_col: Optional[str] = None,
    round_digits: Optional[int] = 4,
    value_key: str = "prediction",
) -> List[Dict[str, Any]]:
    """
    Normalize a batch of predictions.

    - DataFrame: each row -> record; if id_col provided, that column becomes 'id'.
    - Iterable: each element passed to format_prediction.
    """
    if isinstance(batch, pd.DataFrame):
        df = batch.copy()
        ids = None
        if id_col is not None:
            if id_col not in df.columns:
                raise KeyError(f"id_col '{id_col}' not found in DataFrame")
            ids = df[id_col].tolist()
            df = df.drop(columns=[id_col])
        records = dataframe_to_records(df, round_digits=round_digits)
        if ids is not None:
            for rec, idv in zip(records, ids):
                rec["id"] = to_json_serializable(idv)
        return records

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(batch):
        rec = format_prediction(item, id=None if id_col is None else i, round_digits=round_digits, value_key=value_key)
        out.append(rec)
    return out


def pretty_print_metrics(metrics: Dict[str, Any], *, order: Optional[Sequence[str]] = None) -> str:
    """
    Compact one-line summary of metrics. Floats formatted to 4 decimals by default.
    """
    if order is None:
        order = list(metrics.keys())
    parts: List[str] = []
    for k in order:
        if k not in metrics:
            continue
        v = metrics[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)