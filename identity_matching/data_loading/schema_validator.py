from __future__ import annotations

import pandas as pd

from identity_matching.config import RECOMMENDED_COLUMNS, REQUIRED_COLUMNS
from identity_matching.models import DatasetValidationResult


def validate_clean_dataset(df: pd.DataFrame) -> DatasetValidationResult:
    """Validate required/recommended schema columns and basic dataset quality counts."""
    columns = set(df.columns)
    missing_required = [column for column in REQUIRED_COLUMNS if column not in columns]
    missing_recommended = [column for column in RECOMMENDED_COLUMNS if column not in columns]

    total_rows = len(df)
    missing_case_ids_count = _missing_count(df, "case_id")
    duplicated_case_ids_count = _duplicated_case_ids_count(df)
    valid_coordinates_count, missing_coordinates_count = _coordinates_counts(df)
    valid_dates_count, missing_dates_count = _date_counts(df)

    is_valid = not missing_required and not missing_case_ids_count and not duplicated_case_ids_count
    if is_valid:
        message = "The clean file contains the required columns and can be configured."
    elif missing_required:
        message = f"Missing required columns: {', '.join(missing_required)}."
    else:
        message = "Case IDs must be nonempty and unique before matching."

    return DatasetValidationResult(
        is_valid=is_valid,
        missing_required_columns=missing_required,
        missing_recommended_columns=missing_recommended,
        total_rows=total_rows,
        valid_coordinates_count=valid_coordinates_count,
        missing_coordinates_count=missing_coordinates_count,
        valid_dates_count=valid_dates_count,
        missing_dates_count=missing_dates_count,
        duplicated_case_ids_count=duplicated_case_ids_count,
        missing_case_ids_count=missing_case_ids_count,
        message=message,
    )


def _missing_count(df: pd.DataFrame, column: str) -> int:
    """Count null or blank values in a DataFrame column."""
    if column not in df.columns:
        return len(df)
    return int(df[column].isna().sum() + (df[column].astype(str).str.strip() == "").sum())


def _duplicated_case_ids_count(df: pd.DataFrame) -> int:
    """Count case IDs that appear more than once."""
    if "case_id" not in df.columns:
        return 0
    case_ids = df["case_id"].dropna().astype(str).str.strip()
    case_ids = case_ids[case_ids != ""]
    return int(case_ids.duplicated(keep=False).sum())


def _coordinates_counts(df: pd.DataFrame) -> tuple[int, int]:
    """Count valid and invalid GPS coordinate rows."""
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return 0, len(df)
    latitude = pd.to_numeric(df["latitude"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    longitude = pd.to_numeric(df["longitude"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    valid = latitude.between(-90, 90) & longitude.between(-180, 180)
    valid_count = int(valid.sum())
    return valid_count, int(len(df) - valid_count)


def _date_counts(df: pd.DataFrame) -> tuple[int, int]:
    """Count valid and invalid parsed disappearance dates."""
    if "disappearance_date_parsed" not in df.columns:
        return 0, len(df)
    dates = pd.to_datetime(df["disappearance_date_parsed"], errors="coerce")
    valid_count = int(dates.notna().sum())
    return valid_count, int(len(df) - valid_count)
