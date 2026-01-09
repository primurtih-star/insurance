from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)

__all__ = [
    "detect_column_types",
    "build_encoder",
    "fit_transform_encoder",
    "transform_with_encoder",
    "get_feature_names_from_column_transformer",
]


def detect_column_types(df: pd.DataFrame, *, cardinality_thresh: int = 15) -> Tuple[List[str], List[str]]:
    """Return numeric_cols and categorical_cols based on dtypes."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    other_non_numeric = [c for c in df.columns if c not in numeric_cols and c not in cat_cols]
    cat_cols = list(dict.fromkeys(cat_cols + other_non_numeric))
    return numeric_cols, cat_cols


def _onehot_default_kwargs() -> Dict:
    return {"handle_unknown": "ignore", "sparse": False}


def _ordinal_default_kwargs() -> Dict:
    return {"handle_unknown": "use_encoded_value", "unknown_value": -1}


def build_encoder(
    df: pd.DataFrame,
    cat_cols: Optional[Iterable[str]] = None,
    *,
    cardinality_thresh: int = 15,
    onehot_kwargs: Optional[Dict] = None,
    ordinal_kwargs: Optional[Dict] = None,
    passthrough_remainder: bool = True,
) -> ColumnTransformer:
    """Build ColumnTransformer for categorical encoding (OneHot for low cardinality, Ordinal for high)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    onehot_kwargs = dict(onehot_kwargs or {})
    ordinal_kwargs = dict(ordinal_kwargs or {})

    for k, v in _onehot_default_kwargs().items():
        onehot_kwargs.setdefault(k, v)
    for k, v in _ordinal_default_kwargs().items():
        ordinal_kwargs.setdefault(k, v)

    if cat_cols is None:
        _, cat_cols = detect_column_types(df)

    cat_cols = [c for c in list(cat_cols) if c in df.columns]
    low_card = [c for c in cat_cols if df[c].nunique(dropna=True) <= cardinality_thresh]
    high_card = [c for c in cat_cols if df[c].nunique(dropna=True) > cardinality_thresh]

    transformers = []
    if low_card:
        try:
            ohe = OneHotEncoder(**onehot_kwargs)
        except TypeError:
            onehot_kwargs.pop("sparse", None)
            ohe = OneHotEncoder(**onehot_kwargs)
        transformers.append(("onehot", ohe, low_card))
        logger.debug("build_encoder: onehot -> %s", low_card)

    if high_card:
        ord_enc = OrdinalEncoder(**ordinal_kwargs)
        transformers.append(("ordinal", ord_enc, high_card))
        logger.debug("build_encoder: ordinal -> %s", high_card)

    remainder = "passthrough" if passthrough_remainder else "drop"
    return ColumnTransformer(transformers=transformers, remainder=remainder)


def _ensure_dataframe_like(X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    if isinstance(X, pd.Series):
        return X.to_frame()
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame or Series")
    return X


def fit_transform_encoder(
    encoder: ColumnTransformer,
    X: Union[pd.DataFrame, pd.Series],
    *,
    return_transformer: bool = True,
) -> Union[pd.DataFrame, Tuple[ColumnTransformer, pd.DataFrame]]:
    """Fit encoder on X and return transformed DataFrame (optionally with the fitted encoder)."""
    df = _ensure_dataframe_like(X).copy()
    fitted = encoder.fit(df)
    arr = fitted.transform(df)
    out_cols = get_feature_names_from_column_transformer(fitted, list(df.columns))
    out_df = pd.DataFrame(arr, columns=out_cols, index=df.index)
    return (fitted, out_df) if return_transformer else out_df


def transform_with_encoder(encoder: ColumnTransformer, X: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """Transform X using fitted encoder and return DataFrame."""
    df = _ensure_dataframe_like(X).copy()
    arr = encoder.transform(df)
    out_cols = get_feature_names_from_column_transformer(encoder, list(df.columns))
    return pd.DataFrame(arr, columns=out_cols, index=df.index)


def _get_names_from_transformer(trans, col_names: List[str]) -> List[str]:
    if hasattr(trans, "get_feature_names_out"):
        try:
            return list(trans.get_feature_names_out(col_names))
        except TypeError:
            return list(trans.get_feature_names_out())
        except Exception:
            pass

    if isinstance(trans, OrdinalEncoder) or hasattr(trans, "categories_"):
        return list(col_names)

    return [f"{type(trans).__name__}__{c}" for c in col_names]


def get_feature_names_from_column_transformer(ct: ColumnTransformer, input_columns: Optional[Sequence[str]] = None) -> List[str]:
    """Extract output feature names from a ColumnTransformer."""
    input_columns = list(input_columns or [])
    feature_names: List[str] = []

    if not hasattr(ct, "transformers_"):
        return input_columns

    transformed_input_cols: List[str] = []

    for name, trans, cols in ct.transformers_:
        if name == "remainder" or trans == "drop" or trans == "passthrough":
            continue

        if isinstance(cols, (list, tuple, np.ndarray)):
            col_names = list(cols)
        else:
            try:
                col_names = [input_columns[i] for i in cols]
            except Exception:
                col_names = [str(cols)]

        transformed_input_cols.extend([c for c in col_names if isinstance(c, str)])

        try:
            names = _get_names_from_transformer(trans, col_names)
            feature_names.extend(names)
        except Exception:
            for c in col_names:
                feature_names.append(f"{name}__{c}")

    passthrough_cols = [c for c in input_columns if c not in transformed_input_cols]
    if getattr(ct, "remainder", None) == "passthrough":
        feature_names.extend(passthrough_cols)

    return feature_names if feature_names else input_columns