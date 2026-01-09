from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "drop_duplicates",
    "normalize_column_names",
    "trim_strings",
    "drop_high_missing",
    "fill_missing_simple",
    "basic_clean",
]

# --- Helpers -----------------------------------------------------------------


def _trim_series_preserve_dtype(series: pd.Series) -> pd.Series:
    """Trim whitespace for a Series while attempting to preserve dtype."""
    if pd.api.types.is_categorical_dtype(series):
        new_vals = series.astype("string").str.strip()
        new_cat = pd.Categorical(new_vals, categories=new_vals.dropna().unique())
        return pd.Series(new_cat, index=series.index, name=series.name)

    mask = series.notna()
    if not mask.any():
        return series.copy()

    trimmed_vals = series.astype("string").str.strip()
    result = series.copy()
    try:
        cast_back = trimmed_vals.astype(series.dtype)
        result.loc[mask] = cast_back.loc[mask]
    except Exception:
        result.loc[mask] = trimmed_vals.loc[mask]
    return result


# --- Public API --------------------------------------------------------------


def drop_duplicates(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    """Return a copy of df with exact duplicate rows removed."""
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(keep=keep)
    after = len(df)
    logger.debug("drop_duplicates: %d -> %d", before, after)
    return df


def normalize_column_names(df: pd.DataFrame, strip: bool = True, lower: bool = False) -> pd.DataFrame:
    """Normalize column names: optionally strip whitespace and lowercase."""
    df = df.copy()
    cols = []
    for c in df.columns:
        name = str(c)
        if strip:
            name = name.strip()
        if lower:
            name = name.lower()
        cols.append(name)
    df.columns = cols
    return df


def trim_strings(df: pd.DataFrame, inplace: bool = False, include_categories: bool = True) -> pd.DataFrame:
    """
    Trim whitespace from string-like columns.

    Handles pandas 'string' dtype, object, and optionally category.
    Returns a new DataFrame unless inplace=True.
    """
    if not inplace:
        df = df.copy()

    str_types = ["object", "string"]
    cols = df.select_dtypes(include=str_types).columns.tolist()

    if include_categories:
        cat_cols = df.select_dtypes(include=["category"]).columns.tolist()
        cols = list(dict.fromkeys(cols + cat_cols))

    for col in cols:
        series = df[col]
        try:
            df[col] = _trim_series_preserve_dtype(series)
        except Exception:
            logger.debug("trim_strings: fallback trimming for column %s", col)
            df[col] = series.astype("string").str.strip().replace({pd.NA: None}).astype("object", errors="ignore")

    return df


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.6, subset: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Drop columns with fraction of missing values greater than threshold.

    threshold: float in [0, 1]
    subset: optional iterable of column names to consider; if None, consider all columns
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    df = df.copy()
    cols_to_check = list(subset) if subset is not None else df.columns.tolist()
    if not cols_to_check:
        return df

    missing_frac = df[cols_to_check].isna().mean()
    to_drop = missing_frac[missing_frac > threshold].index.tolist()
    if to_drop:
        logger.info("drop_high_missing -> dropping columns: %s", to_drop)
        df = df.drop(columns=to_drop)
    return df


def fill_missing_simple(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    numeric_constant: Union[int, float] = 0,
    categorical_strategy: str = "mode",
    categorical_constant: str = "",
    exclude_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Fill missing values with simple strategies.

    numeric_strategy: "median", "mean", or "constant"
    categorical_strategy: "mode" or "constant"
    exclude_columns: columns to skip
    """
    df = df.copy()
    exclude = set(exclude_columns or [])

    if numeric_strategy not in {"median", "mean", "constant"}:
        raise ValueError("numeric_strategy must be 'median', 'mean', or 'constant'")
    if categorical_strategy not in {"mode", "constant"}:
        raise ValueError("categorical_strategy must be 'mode' or 'constant'")

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    cat_cols = [c for c in df.select_dtypes(include=["object", "string", "category"]).columns if c not in exclude]

    for c in num_cols:
        if numeric_strategy == "median":
            val = df[c].median()
        elif numeric_strategy == "mean":
            val = df[c].mean()
        else:
            val = numeric_constant

        if pd.isna(val):
            val = numeric_constant

        df[c] = df[c].fillna(val)

    for c in cat_cols:
        if categorical_strategy == "mode":
            mode_series = df[c].mode(dropna=True)
            fillv = mode_series.iloc[0] if not mode_series.empty else categorical_constant
        else:
            fillv = categorical_constant

        if pd.api.types.is_categorical_dtype(df[c]):
            if fillv not in df[c].cat.categories:
                try:
                    df[c] = df[c].cat.add_categories([fillv])
                except Exception:
                    df[c] = df[c].astype("object")
                    df[c] = df[c].fillna(fillv)
                    continue
            df[c] = df[c].fillna(fillv)
        else:
            df[c] = df[c].fillna(fillv)

    return df


def basic_clean(
    df: pd.DataFrame,
    drop_thresh: Optional[float] = 0.6,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    inplace_trim: bool = False,
    numeric_constant: Union[int, float] = 0,
    categorical_constant: str = "",
    exclude_fill: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Run a standard cleaning pipeline:
      - normalize column names (strip)
      - drop exact duplicates
      - drop fully empty columns
      - trim string columns
      - optionally drop columns with high missing fraction
      - fill missing values with simple strategies
    """
    df = df.copy()
    df = normalize_column_names(df, strip=True)
    df = drop_duplicates(df)
    df = df.dropna(axis=1, how="all")
    df = trim_strings(df, inplace=inplace_trim)

    if drop_thresh is not None:
        df = drop_high_missing(df, threshold=drop_thresh)

    df = fill_missing_simple(
        df,
        numeric_strategy=numeric_strategy,
        numeric_constant=numeric_constant,
        categorical_strategy=categorical_strategy,
        categorical_constant=categorical_constant,
        exclude_columns=exclude_fill,
    )
    return df