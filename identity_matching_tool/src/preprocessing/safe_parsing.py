from __future__ import annotations

import ast
from datetime import date, datetime
from typing import Any

import pandas as pd


def none_if_missing(value: Any) -> Any | None:
    """Normalize pandas/empty missing values to None."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict, set)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and not value.strip():
        return None
    return value


def parse_optional_float(value: Any) -> float | None:
    """Parse a float value while tolerating missing or malformed inputs."""
    value = none_if_missing(value)
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def parse_optional_date(value: Any) -> date | datetime | None:
    """Parse a date-like value while tolerating missing or malformed inputs."""
    value = none_if_missing(value)
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime().date()


def parse_name_tokens(value: Any) -> list[str]:
    """Parse the name_tokens field into a list of string tokens."""
    value = none_if_missing(value)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(token).strip() for token in value if str(token).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return [str(token).strip() for token in parsed if str(token).strip()]
        return [token.strip() for token in stripped.replace(";", ",").split(",") if token.strip()]
    return [str(value).strip()]
