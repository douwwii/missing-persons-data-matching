from __future__ import annotations

import pandas as pd

from src.config import RECOMMENDED_COLUMNS, REQUIRED_COLUMNS
from src.data_loading.schema_validator import validate_clean_dataset


def valid_df() -> pd.DataFrame:
    data = {
        "case_id": ["MP-001", "MP-002", "MP-003"],
        "full_name_raw": ["Marie Dupont", "Marc Dupond", "Anne Martin"],
        "full_name_normalized": ["marie dupont", "marc dupond", "anne martin"],
        "name_tokens": ["['marie', 'dupont']", "['marc', 'dupond']", "['anne', 'martin']"],
        "disappearance_city_verified": ["Lyon", "Villeurbanne", "Grenoble"],
        "latitude": [45.764, "45,7719", None],
        "longitude": [4.8357, "4,8902", None],
        "disappearance_date_parsed": ["2024-02-18", "2024-02-21", ""],
    }
    for column in RECOMMENDED_COLUMNS:
        data[column] = ["value", "value", "value"]
    return pd.DataFrame(data)


def test_valid_file_with_all_required_columns() -> None:
    result = validate_clean_dataset(valid_df())

    assert result.is_valid is True
    assert result.missing_required_columns == []
    assert result.total_rows == 3


def test_invalid_without_case_id() -> None:
    df = valid_df().drop(columns=["case_id"])

    result = validate_clean_dataset(df)

    assert result.is_valid is False
    assert "case_id" in result.missing_required_columns


def test_invalid_without_full_name_normalized() -> None:
    df = valid_df().drop(columns=["full_name_normalized"])

    result = validate_clean_dataset(df)

    assert result.is_valid is False
    assert "full_name_normalized" in result.missing_required_columns


def test_invalid_without_name_tokens() -> None:
    df = valid_df().drop(columns=["name_tokens"])

    result = validate_clean_dataset(df)

    assert result.is_valid is False
    assert "name_tokens" in result.missing_required_columns


def test_invalid_without_disappearance_city_verified() -> None:
    df = valid_df().drop(columns=["disappearance_city_verified"])

    result = validate_clean_dataset(df)

    assert result.is_valid is False
    assert "disappearance_city_verified" in result.missing_required_columns


def test_invalid_without_latitude() -> None:
    df = valid_df().drop(columns=["latitude"])

    result = validate_clean_dataset(df)

    assert result.is_valid is False
    assert "latitude" in result.missing_required_columns


def test_invalid_without_longitude() -> None:
    df = valid_df().drop(columns=["longitude"])

    result = validate_clean_dataset(df)

    assert result.is_valid is False
    assert "longitude" in result.missing_required_columns


def test_invalid_without_disappearance_date_parsed() -> None:
    df = valid_df().drop(columns=["disappearance_date_parsed"])

    result = validate_clean_dataset(df)

    assert result.is_valid is False
    assert "disappearance_date_parsed" in result.missing_required_columns


def test_missing_recommended_columns_do_not_block() -> None:
    df = valid_df().drop(columns=RECOMMENDED_COLUMNS)

    result = validate_clean_dataset(df)

    assert result.is_valid is True
    assert set(result.missing_recommended_columns) == set(RECOMMENDED_COLUMNS)


def test_duplicate_case_ids_are_counted() -> None:
    df = valid_df()
    df.loc[1, "case_id"] = "MP-001"

    result = validate_clean_dataset(df)

    assert result.duplicated_case_ids_count == 2


def test_valid_coordinates_are_counted() -> None:
    result = validate_clean_dataset(valid_df())

    assert result.valid_coordinates_count == 2
    assert result.missing_coordinates_count == 1


def test_valid_dates_are_counted() -> None:
    df = valid_df()
    df.loc[1, "disappearance_date_parsed"] = "not-a-date"

    result = validate_clean_dataset(df)

    assert result.valid_dates_count == 1
    assert result.missing_dates_count == 2


def test_required_columns_constant_matches_expected_contract() -> None:
    assert REQUIRED_COLUMNS == [
        "case_id",
        "full_name_raw",
        "full_name_normalized",
        "name_tokens",
        "disappearance_city_verified",
        "latitude",
        "longitude",
        "disappearance_date_parsed",
    ]

