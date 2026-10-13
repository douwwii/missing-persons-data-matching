from __future__ import annotations

from typing import Any

import pandas as pd

from identity_matching.data_loading.case_ids import case_id_issue_counts, normalized_case_ids
from identity_matching.models import MissingPersonRecord
from identity_matching.preprocessing.safe_parsing import (
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
        # Converters retain literal IDs such as NA/NULL and leading zeros,
        # while pandas still handles missing values normally in other fields.
        return pd.read_csv(uploaded_file_or_path, converters={"case_id": str, "source_case_id": str})
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
    if "case_id" not in df.columns:
        raise CleanFileLoadingError("The case_id column is required for matching.")
    missing, duplicated = case_id_issue_counts(df["case_id"])
    if missing or duplicated:
        raise CleanFileLoadingError("Case IDs must be nonempty and unique before converting records.")
    ids = normalized_case_ids(df["case_id"])
    records: list[MissingPersonRecord] = []
    for position, (_, row) in enumerate(df.iterrows()):
        raw_data = row.where(pd.notna(row), None).to_dict()
        records.append(
            MissingPersonRecord(
                case_id=str(ids.iloc[position]),
                full_name_raw=str(none_if_missing(row.get("full_name_raw")) or ""),
                full_name_normalized=str(none_if_missing(row.get("full_name_normalized")) or ""),
                first_name_raw=_optional_str(row.get("first_name_raw")),
                first_name_normalized=_optional_str(row.get("first_name_normalized")),
                last_name_raw=_optional_str(row.get("last_name_raw")),
                last_name_normalized=_optional_str(row.get("last_name_normalized")),
                name_tokens=parse_name_tokens(row.get("name_tokens")),
                disappearance_city_verified=str(
                    none_if_missing(row.get("disappearance_city_verified")) or ""
                ),
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
