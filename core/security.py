from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional

from core.constants import (
    ML_CONSTRAINTS,
    REGION_OPTIONS,
    GENDER_OPTIONS,
    SMOKER_OPTIONS,
)

logger = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]*>")
_SAFE_CHARS_RE = re.compile(r"[^\w\s\-\.\@]")

_GENDER_SET = {g.lower() for g in GENDER_OPTIONS}
_SMOKER_SET = {s.lower() for s in SMOKER_OPTIONS}
_REGION_SET = {r.lower() for r in REGION_OPTIONS}

_GENDER_MAP = {g.lower(): i for i, g in enumerate(GENDER_OPTIONS)}
_SMOKER_MAP = {s.lower(): i for i, s in enumerate(SMOKER_OPTIONS)}
_REGION_MAP = {r.lower(): i for i, r in enumerate(REGION_OPTIONS)}


@dataclass(frozen=True)
class ValidationResult:
    errors: Dict[str, str]
    cleaned: Dict[str, Any]

    def is_valid(self) -> bool:
        return not bool(self.errors)


def sanitize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = _HTML_TAG_RE.sub("", text)
    text = _SAFE_CHARS_RE.sub("", text)
    return text.strip()


def _to_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
        if v != v:  # NaN
            return None
        return v
    except (TypeError, ValueError):
        logger.debug("safe float conversion failed for %r", value)
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        logger.debug("safe int conversion failed for %r", value)
        return None


def _check_ml_constraints() -> None:
    required = {"age_min", "age_max", "bmi_min", "bmi_max", "children_max"}
    missing = required - set(ML_CONSTRAINTS.keys())
    if missing:
        logger.error("ML_CONSTRAINTS missing keys: %s", missing)


def validate_prediction_input(raw: Mapping[str, Any]) -> ValidationResult:
    _check_ml_constraints()

    errors: MutableMapping[str, str] = {}
    cleaned: MutableMapping[str, Any] = {}

    age = _to_int(raw.get("age"))
    if age is None:
        errors["age"] = "Age is required and must be a number."
    else:
        if not (ML_CONSTRAINTS["age_min"] <= age <= ML_CONSTRAINTS["age_max"]):
            errors["age"] = f"Age must be between {ML_CONSTRAINTS['age_min']} and {ML_CONSTRAINTS['age_max']}."
            logger.warning("Validation failed for age: %s", age)
        cleaned["age"] = age

    bmi = _to_float(raw.get("bmi"))
    if bmi is None:
        errors["bmi"] = "BMI is required and must be a number."
    else:
        if not (ML_CONSTRAINTS["bmi_min"] <= bmi <= ML_CONSTRAINTS["bmi_max"]):
            errors["bmi"] = f"BMI must be between {ML_CONSTRAINTS['bmi_min']} and {ML_CONSTRAINTS['bmi_max']}."
            logger.warning("Validation failed for bmi: %s", bmi)
        cleaned["bmi"] = bmi

    children = _to_int(raw.get("children"))
    if children is None:
        children = 0
    if children < 0 or children > ML_CONSTRAINTS["children_max"]:
        errors["children"] = f"Number of children must be 0..{ML_CONSTRAINTS['children_max']}."
        logger.warning("Validation failed for children: %s", children)
    cleaned["children"] = children

    sex_raw = str(raw.get("sex") or "").strip().lower()
    if sex_raw not in _GENDER_SET:
        errors["sex"] = "Invalid gender selection."
    else:
        cleaned["sex"] = sex_raw
        cleaned["sex_code"] = _GENDER_MAP.get(sex_raw)

    smoker_raw = str(raw.get("smoker") or "").strip().lower()
    if smoker_raw not in _SMOKER_SET:
        errors["smoker"] = "Invalid smoker status."
    else:
        cleaned["smoker"] = smoker_raw
        cleaned["smoker_code"] = _SMOKER_MAP.get(smoker_raw)

    region_raw = str(raw.get("region") or "").strip().lower()
    if region_raw not in _REGION_SET:
        errors["region"] = "Invalid region selected."
    else:
        cleaned["region"] = region_raw
        cleaned["region_code"] = _REGION_MAP.get(region_raw)

    if "name" in raw:
        cleaned["name"] = sanitize_text(raw.get("name"))

    return ValidationResult(errors=dict(errors), cleaned=dict(cleaned))