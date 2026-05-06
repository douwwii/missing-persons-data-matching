from __future__ import annotations

from io import StringIO

import pandas as pd
import pytest

from src.data_loading.clean_file_loader import (
    CleanFileLoadingError,
    dataframe_to_records,
    load_clean_csv,
)
from src.models import MissingPersonRecord


CSV_CONTENT = """case_id,full_name_raw,full_name_normalized,name_tokens,disappearance_city_verified,latitude,longitude,disappearance_date_parsed
MP-001,Marie Dupont,marie dupont,"['marie', 'dupont']",Lyon,45.7640,4.8357,2024-02-18
"""


def test_load_valid_csv() -> None:
    df = load_clean_csv(StringIO(CSV_CONTENT))

    assert len(df) == 1
    assert df.loc[0, "case_id"] == "MP-001"


def test_latitude_longitude_are_converted_to_float_in_records() -> None:
    df = pd.DataFrame(
        {
            "case_id": ["MP-001"],
            "full_name_raw": ["Marie Dupont"],
            "full_name_normalized": ["marie dupont"],
            "name_tokens": ["['marie', 'dupont']"],
            "disappearance_city_verified": ["Lyon"],
            "latitude": ["45,7640"],
            "longitude": ["4,8357"],
            "disappearance_date_parsed": ["2024-02-18"],
        }
    )

    record = dataframe_to_records(df)[0]

    assert record.latitude == pytest.approx(45.7640)
    assert record.longitude == pytest.approx(4.8357)


def test_raw_data_is_preserved() -> None:
    df = load_clean_csv(StringIO(CSV_CONTENT))

    record = dataframe_to_records(df)[0]

    assert record.raw_data["case_id"] == "MP-001"
    assert record.raw_data["full_name_raw"] == "Marie Dupont"


def test_missing_person_record_is_created() -> None:
    df = load_clean_csv(StringIO(CSV_CONTENT))

    records = dataframe_to_records(df)

    assert len(records) == 1
    assert isinstance(records[0], MissingPersonRecord)
    assert records[0].name_tokens == ["marie", "dupont"]


def test_unreadable_file_raises_user_readable_error() -> None:
    with pytest.raises(CleanFileLoadingError) as exc_info:
        load_clean_csv("does-not-exist.csv")

    assert "cannot be found or accessed" in str(exc_info.value)
