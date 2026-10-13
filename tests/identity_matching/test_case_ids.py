from io import StringIO

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from examples.demo import EXAMPLES, clean_synthetic_records
from identity_matching.data_loading.case_ids import prepare_unique_case_ids
from identity_matching.data_loading.clean_file_loader import (
    CleanFileLoadingError,
    dataframe_to_records,
    load_clean_csv,
)
from identity_matching.data_loading.schema_validator import validate_clean_dataset
from identity_matching.matching.matching_engine import run_matching
from identity_matching.models import AnalysisConfig, ParallelConfig


def test_duplicate_ids_report_the_actual_problem():
    df = clean_synthetic_records()
    df.loc[1, "case_id"] = df.loc[0, "case_id"]
    validation = validate_clean_dataset(df)
    assert validation.missing_required_columns == []
    assert not validation.is_valid
    assert validation.duplicated_case_ids_count == 2
    assert "2" in validation.message
    assert "duplicate" in validation.message


def test_row_id_repair_preserves_source_data_and_all_matching_pairs():
    original = clean_synthetic_records()
    original["case_id"] = ["same", "same", "0003", ""]
    before = original.copy(deep=True)
    repaired, changed = prepare_unique_case_ids(original)
    pd.testing.assert_frame_equal(original, before)
    pd.testing.assert_frame_equal(
        repaired.drop(columns=["case_id", "source_case_id"]), original.drop(columns="case_id")
    )
    assert repaired["source_case_id"].tolist() == ["same", "same", "0003", ""]
    assert repaired.loc[2, "case_id"] == "0003"
    assert changed == 3
    assert validate_clean_dataset(repaired).is_valid
    loaded = load_clean_csv(StringIO(repaired.to_csv(index=False)))
    assert loaded["source_case_id"].tolist() == ["same", "same", "0003", ""]
    records = dataframe_to_records(loaded)
    assert len(records) == len(original) == len({record.case_id for record in records})
    result = run_matching(
        records, AnalysisConfig(investigation_mode="global"), parallel_config=ParallelConfig(num_workers=1)
    )
    assert result.compared_pairs_count == 6


def test_generated_ids_avoid_existing_ids_and_duplicate_dataframe_indexes():
    df = pd.DataFrame(
        {"case_id": ["duplicate", "duplicate", "record-000001", "record-000002", " ", None]},
        index=[0, 0, 1, 1, 2, 2],
    )
    repaired, changed = prepare_unique_case_ids(df)
    assert changed == 4
    assert repaired.index.tolist() == df.index.tolist()
    assert repaired["case_id"].is_unique
    assert repaired["case_id"].iloc[2:4].tolist() == ["record-000001", "record-000002"]
    assert repaired["case_id"].iloc[0] == "record-000001-2"
    assert repaired["case_id"].iloc[1] == "record-000002-2"
    again, changed_again = prepare_unique_case_ids(repaired)
    pd.testing.assert_frame_equal(repaired, again)
    assert changed_again == 0


def test_existing_source_identifiers_are_not_overwritten():
    df = pd.DataFrame({"case_id": ["x", "x"], "source_case_id": ["original-a", "original-b"]})
    repaired, _ = prepare_unique_case_ids(df)
    assert repaired["source_case_id"].tolist() == ["original-a", "original-b"]
    assert repaired["input_case_id"].tolist() == ["x", "x"]


def test_ten_thousand_records_are_retained_without_scoring_them():
    df = pd.DataFrame({"case_id": ["shared"] * 10290, "row_value": range(10290)})
    repaired, changed = prepare_unique_case_ids(df)
    assert changed == len(df) == len(repaired)
    assert repaired["case_id"].is_unique
    assert repaired["row_value"].tolist() == df["row_value"].tolist()


def test_csv_loading_preserves_literal_and_zero_padded_ids():
    csv = "case_id,source_case_id,full_name_raw\nNA,NULL,\nNULL,0001,Name\n0000,NA,Name\n"
    df = load_clean_csv(StringIO(csv))
    assert df["case_id"].tolist() == ["NA", "NULL", "0000"]
    assert df["source_case_id"].tolist() == ["NULL", "0001", "NA"]
    assert pd.isna(df.loc[0, "full_name_raw"])
    assert [record.case_id for record in dataframe_to_records(df)] == ["NA", "NULL", "0000"]


@pytest.mark.parametrize("ids", [["same", " same ", "3", "4"], ["", "2", "3", "4"]])
def test_record_conversion_rejects_ambiguous_identifiers(ids):
    df = clean_synthetic_records()
    df["case_id"] = ids
    with pytest.raises(CleanFileLoadingError, match="unique"):
        dataframe_to_records(df)


def test_matching_ui_repairs_ids_only_after_user_action(monkeypatch):
    from identity_matching.ui import upload_section

    original = clean_synthetic_records()
    original["case_id"] = ["same", "same", "0003", "0004"]

    def uploaded_dataset():
        st.session_state["uploaded_file_signature"] = "synthetic-upload"
        return st.session_state.get("loaded_clean_dataset", original)

    monkeypatch.setattr(upload_section, "render_upload_section", uploaded_dataset)
    app = AppTest.from_file(str(EXAMPLES.parent / "identity_matching/app.py")).run(timeout=20)
    assert not app.exception
    assert any("duplicate" in message.value for message in app.error)
    assert not any("required columns" in message.value for message in app.error)
    assert "loaded_clean_dataset" not in app.session_state
    app.button(key="repair_case_ids").click().run(timeout=20)
    assert not app.exception
    assert not app.error
    assert len(app.session_state["missing_person_records"]) == 4
    assert (
        app.session_state["loaded_clean_dataset"]["source_case_id"].tolist() == original["case_id"].tolist()
    )
    assert app.session_state["dataset_validation"].is_valid
    assert any("All 4 rows are retained" in message.value for message in app.success)


def test_missing_schema_cannot_be_bypassed_by_id_repair(monkeypatch):
    from identity_matching.ui import upload_section

    df = clean_synthetic_records().drop(columns="name_tokens")
    monkeypatch.setattr(upload_section, "render_upload_section", lambda: df)
    app = AppTest.from_file(str(EXAMPLES.parent / "identity_matching/app.py")).run(timeout=20)
    assert not app.exception
    assert any("name_tokens" in message.value for message in app.error)
    assert not any(button.key == "repair_case_ids" for button in app.button)


def test_existing_input_id_backup_is_retained():
    df = pd.DataFrame(
        {"case_id": ["x", "x"], "source_case_id": ["a", "b"], "input_case_id": ["previous-a", "previous-b"]}
    )
    repaired, _ = prepare_unique_case_ids(df)
    assert repaired["input_case_id"].tolist() == ["previous-a", "previous-b"]
    assert repaired["input_case_id_2"].tolist() == ["x", "x"]


def test_zero_and_trimmed_ids_stay_usable_as_engine_keys():
    df = clean_synthetic_records()
    df["case_id"] = [0, " second ", "3", "4"]
    records = dataframe_to_records(df)
    assert [record.case_id for record in records] == ["0", "second", "3", "4"]


def test_new_upload_discards_previous_repair_and_cached_records(monkeypatch):

    class UploadedCSV:
        name = "synthetic.csv"

        def __init__(self, df):
            self.data = df.to_csv(index=False).encode("utf-8-sig")

        def getvalue(self):
            return self.data

    first = clean_synthetic_records()
    first["case_id"] = ["shared", "shared", "0003", "0004"]
    uploaded = [UploadedCSV(first)]
    monkeypatch.setattr(st, "file_uploader", lambda *_, **__: uploaded[0])
    app = AppTest.from_file(str(EXAMPLES.parent / "identity_matching/app.py")).run(timeout=20)
    assert not app.exception
    app.button(key="repair_case_ids").click().run(timeout=20)
    assert not app.exception
    assert app.session_state["case_id_repair_count"] == 2
    assert len(app.session_state["missing_person_records"]) == 4
    second = clean_synthetic_records().iloc[:2].copy()
    second["case_id"] = ["fresh-1", "fresh-2"]
    uploaded[0] = UploadedCSV(second)
    app.run(timeout=20)
    assert not app.exception
    assert not app.error
    assert "case_id_repair_count" not in app.session_state
    assert app.session_state["dataset_validation"].total_rows == 2
    assert [record.case_id for record in app.session_state["missing_person_records"]] == [
        "fresh-1",
        "fresh-2",
    ]
    assert "source_case_id" not in app.session_state["loaded_clean_dataset"]
