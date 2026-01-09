from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

import pandas as pd

from core.constants import CSV_REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

__all__ = ["check_required_columns", "validate_dataset"]


def check_required_columns(df: pd.DataFrame, required: Sequence[str], *, context: str = "dataset") -> None:
    """
    Raise ValueError if any column in `required` is missing from `df`.

    Args:
        df: pandas DataFrame to check.
        required: sequence of required column names.
        context: short label used in the error/log message.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        msg = f"[{context}] Missing required columns: {missing}"
        logger.error(msg)
        raise ValueError(msg)


def validate_dataset(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    *,
    min_rows: int = 2,
    min_feature_cols: int = 1,
    require_target_non_null: bool = True,
    allowed_target_dtypes: Optional[Sequence[str]] = None,
) -> None:
    """
    Validate a DataFrame for ML readiness.

    Checks performed:
      - df is a pandas DataFrame and not empty
      - dataset has at least `min_rows` rows
      - required feature columns (from CSV_REQUIRED_COLUMNS, excluding the canonical target 'charges')
        are present
      - if `target_col` provided: target exists, optional non-null requirement, and optional dtype check
      - dataset has at least `min_feature_cols` feature columns besides the target

    Raises:
      - TypeError, ValueError on invalid inputs
    """
    # Basic type and emptiness checks
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("Dataset is empty.")

    # Row count
    n_rows = int(df.shape[0])
    if n_rows < max(1, int(min_rows)):
        raise ValueError(f"Dataset must contain at least {min_rows} rows; got {n_rows}")

    # Required feature columns (SSOT from core.constants)
    # Exclude canonical target 'charges' from required features check
    feature_required = [c for c in CSV_REQUIRED_COLUMNS if c != "charges"]
    check_required_columns(df, feature_required, context="Input Features")

    # Feature count (exclude target if present)
    n_features = df.shape[1] - (1 if target_col and target_col in df.columns else 0)
    if n_features < max(0, int(min_feature_cols)):
        raise ValueError(
            f"Dataset must have at least {min_feature_cols} feature column(s) besides the target; found {n_features}"
        )

    # Target checks (only when target_col provided)
    if target_col:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found for training.")
        if require_target_non_null:
            non_null_count = int(df[target_col].notna().sum())
            if non_null_count == 0:
                raise ValueError(f"Target column '{target_col}' contains only null values.")
            # warn if many target nulls
            if non_null_count < n_rows * 0.5:
                logger.warning(
                    "Target column '%s' has a high fraction of nulls: %d/%d non-null",
                    target_col,
                    non_null_count,
                    n_rows,
                )

        # Optional dtype whitelist for target
        if allowed_target_dtypes is not None:
            dtype_name = str(df[target_col].dtype)
            if dtype_name not in set(allowed_target_dtypes):
                raise TypeError(
                    f"Target column '{target_col}' has dtype '{dtype_name}' which is not in allowed dtypes: "
                    f"{list(allowed_target_dtypes)}"
                )

    logger.info("Dataset validation passed. shape=%s, target=%s", df.shape, target_col or "None (inference)")