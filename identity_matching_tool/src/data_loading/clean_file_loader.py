from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.models import MissingPersonRecord
from src.preprocessing.safe_parsing import (
    none_if_missing,
    parse_name_tokens,
    parse_optional_date,
    parse_optional_float,
)


class CleanFileLoadingError(ValueError):
    """Raised when a clean CSV cannot be loaded with a user-readable message."""


def load_clean_csv(uploaded_file_or_path: Any) -> pd.DataFrame:
    """Load a clean CSV file into a DataFrame with user-readable errors."""
    try:
        if isinstance(uploaded_file_or_path, (str, Path)):
            return pd.read_csv(uploaded_file_or_path)
        return pd.read_csv(uploaded_file_or_path)
    except UnicodeDecodeError as exc:
        raise CleanFileLoadingError(
            "The CSV file cannot be read with the expected encoding. "
            "Check that the file exported by step 1 is a readable CSV."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise CleanFileLoadingError("The CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise CleanFileLoadingError(
            "The CSV file appears to be malformed. Check the separator, quotes, and step 1 export."
        ) from exc
    except OSError as exc:
        raise CleanFileLoadingError(f"The CSV file cannot be found or accessed: {exc}") from exc
    except Exception as exc:
        raise CleanFileLoadingError(f"Unable to load the CSV file: {exc}") from exc


def dataframe_to_records(df: pd.DataFrame) -> list[MissingPersonRecord]:
    """Convert validated DataFrame rows into internal missing person records."""
    records: list[MissingPersonRecord] = []
    for _, row in df.iterrows():
        raw_data = row.where(pd.notna(row), None).to_dict()
        records.append(
            MissingPersonRecord(
                case_id=str(none_if_missing(row.get("case_id")) or ""),
                full_name_raw=str(none_if_missing(row.get("full_name_raw")) or ""),
                full_name_normalized=str(none_if_missing(row.get("full_name_normalized")) or ""),
                first_name_raw=_optional_str(row.get("first_name_raw")),
                first_name_normalized=_optional_str(row.get("first_name_normalized")),
                last_name_raw=_optional_str(row.get("last_name_raw")),
                last_name_normalized=_optional_str(row.get("last_name_normalized")),
                name_tokens=parse_name_tokens(row.get("name_tokens")),
                disappearance_city_verified=str(none_if_missing(row.get("disappearance_city_verified")) or ""),
                disappearance_department_verified=_optional_str(row.get("disappearance_department_verified")),
                latitude=parse_optional_float(row.get("latitude")),
                longitude=parse_optional_float(row.get("longitude")),
                disappearance_date_parsed=parse_optional_date(row.get("disappearance_date_parsed")),
                raw_data=raw_data,
            )
        )
    return records


def _optional_str(value: Any) -> str | None:
    """Return a stripped string value or None for missing values."""
    value = none_if_missing(value)
    if value is None:
        return None
    return str(value)
