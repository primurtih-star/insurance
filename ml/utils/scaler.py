from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

__all__ = [
    "build_scaler",
    "numeric_columns",
    "fit_scaler",
    "transform_with_scaler",
    "fit_transform_scaler",
]


# -------------------------
# Utilities
# -------------------------
def numeric_columns(df: pd.DataFrame, include_bool: bool = False) -> Sequence[str]:
    """
    Return numeric column names from DataFrame.
    If include_bool is True, boolean dtype is considered numeric.
    """
    if include_bool:
        return df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    return df.select_dtypes(include=[np.number]).columns.tolist()


# -------------------------
# Scaler factory
# -------------------------
def build_scaler(strategy: str = "standard", **kwargs) -> TransformerMixin:
    """
    Build and return an unfitted scaler.

    strategy:
      - "standard": StandardScaler (Z-score)
      - "robust": RobustScaler (outlier resistant)
      - "minmax": MinMaxScaler (0-1 range)

    kwargs are forwarded to the scaler constructor.
    """
    s = (strategy or "standard").strip().lower()
    if s == "standard":
        return StandardScaler(**kwargs)
    if s == "robust":
        return RobustScaler(**kwargs)
    if s == "minmax":
        return MinMaxScaler(**kwargs)

    logger.warning("Unknown scaler strategy '%s', defaulting to StandardScaler.", strategy)
    return StandardScaler(**kwargs)


# -------------------------
# Fit / Transform helpers
# -------------------------
def fit_scaler(
    scaler: TransformerMixin,
    X: pd.DataFrame,
    *,
    columns: Optional[Iterable[str]] = None,
    copy: bool = True,
) -> Tuple[TransformerMixin, pd.DataFrame]:
    """
    Fit scaler on numeric columns and return (fitted_scaler, scaled_df).

    - scaler: unfitted scaler instance (e.g., StandardScaler)
    - X: DataFrame to fit on
    - columns: optional iterable of column names to scale; if None, numeric columns are used
    - copy: operate on a copy by default
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    df = X.copy() if copy else X

    cols = list(columns) if columns is not None else numeric_columns(df)
    cols = [c for c in cols if c in df.columns]

    if not cols:
        logger.warning("fit_scaler: no numeric columns found to scale")
        return scaler, df

    arr = df[cols].to_numpy(dtype=float, copy=True)
    try:
        scaler.fit(arr)
        transformed = scaler.transform(arr)
    except Exception as exc:
        raise RuntimeError(f"Failed to fit/transform scaler: {exc}") from exc

    df_scaled = df.copy()
    df_scaled.loc[:, cols] = pd.DataFrame(transformed, columns=cols, index=df.index)
    logger.debug("fit_scaler: scaled columns %s", cols)
    return scaler, df_scaled


def transform_with_scaler(
    scaler: TransformerMixin,
    X: pd.DataFrame,
    *,
    columns: Optional[Iterable[str]] = None,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Transform DataFrame X using a previously fitted scaler.

    - scaler: fitted scaler
    - X: DataFrame to transform
    - columns: optional iterable of column names to scale; if None, numeric columns are used
    - copy: operate on a copy by default
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    df = X.copy() if copy else X

    cols = list(columns) if columns is not None else numeric_columns(df)
    cols = [c for c in cols if c in df.columns]

    if not cols:
        logger.warning("transform_with_scaler: no numeric columns found to scale")
        return df

    arr = df[cols].to_numpy(dtype=float, copy=True)
    try:
        transformed = scaler.transform(arr)
    except Exception as exc:
        raise RuntimeError("Failed to transform with provided scaler: " + str(exc)) from exc

    df_transformed = df.copy()
    df_transformed.loc[:, cols] = pd.DataFrame(transformed, columns=cols, index=df.index)
    logger.debug("transform_with_scaler: transformed columns %s", cols)
    return df_transformed


def fit_transform_scaler(
    scaler: Optional[TransformerMixin],
    X: pd.DataFrame,
    *,
    columns: Optional[Iterable[str]] = None,
    copy: bool = True,
    scaler_kwargs: Optional[dict] = None,
) -> Tuple[TransformerMixin, pd.DataFrame]:
    """
    Convenience: build (if None), fit, and transform in one call.

    - scaler: if None, a new scaler is created using build_scaler(**scaler_kwargs)
    - scaler_kwargs: forwarded to build_scaler when scaler is None
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    if scaler is None:
        scaler = build_scaler(**(scaler_kwargs or {}))

    fitted_scaler, df_scaled = fit_scaler(scaler, X, columns=columns, copy=copy)
    return fitted_scaler, df_scaled