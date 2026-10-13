import sqlite3
from io import BytesIO, StringIO

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from data_cleaning.cities import build_city_references
from data_cleaning.io import read_table
from examples.demo import EXAMPLES, clean_synthetic_records, run_demo
from identity_matching.cache import sqlite_cache
from identity_matching.data_loading.clean_file_loader import dataframe_to_records, load_clean_csv
from identity_matching.data_loading.schema_validator import validate_clean_dataset
from identity_matching.models import MissingPersonRecord
from identity_matching.preprocessing.safe_parsing import parse_name_tokens
from identity_matching.similarity import phonetic_similarity


def test_cleaning_export_can_be_consumed_without_losing_ids_or_tokens(tmp_path):
    clean = clean_synthetic_records()
    loaded = load_clean_csv(StringIO(clean.to_csv(index=False)))
    assert validate_clean_dataset(loaded).is_valid
    records = dataframe_to_records(loaded)
    assert records[0].case_id == "0001"
    assert records[0].name_tokens == ["aline", "fiction"]
    assert clean.loc[3, "city_resolution_status"] == "human_validated_city"
    result = run_demo(tmp_path)
    assert result.status == "ok"
    assert result.compared_pairs_count == 6
    assert any({r.id_A, r.id_B} == {"0001", "0002"} for r in result.results)


@pytest.mark.parametrize("ids", [["same", "same", "3", "4"], ["", "2", "3", "4"]])
def test_ambiguous_case_ids_block_matching(ids):
    clean = clean_synthetic_records()
    clean["case_id"] = ids
    validation = validate_clean_dataset(clean)
    assert not validation.is_valid
    assert "unique" in validation.message


@pytest.mark.parametrize("value", ["aline fiction", "aline;fiction", "['aline', 'fiction']"])
def test_cleaner_and_legacy_token_encodings_are_supported(value):
    assert parse_name_tokens(value) == ["aline", "fiction"]


def test_csv_and_xlsx_loading_preserve_identifiers():
    assert read_table(EXAMPLES / "synthetic_persons.csv").iloc[0]["id"] == "0001"
    buffer = BytesIO()
    pd.DataFrame({"id": ["0001"]}).to_excel(buffer, index=False)
    buffer.seek(0)
    buffer.name = "synthetic.xlsx"
    assert read_table(buffer).iloc[0]["id"] == "0001"


def test_invalid_reference_coordinates_are_discarded():
    refs = pd.DataFrame(
        {
            "city": ["Invalid", "Also Invalid", "Valid"],
            "dept": ["Demo"] * 3,
            "lat": [999, "inf", "10,2"],
            "lon": [0, 0, "20,3"],
        }
    )
    built = build_city_references(refs, "city", "dept", "lat", "lon")
    assert len(built) == 1
    assert built[0].latitude == pytest.approx(10.2)


def test_sqlite_commits_rolls_back_and_closes_connections():
    with sqlite_cache.connect() as connection:
        connection.execute("CREATE TABLE demo (value TEXT)")
        connection.execute("INSERT INTO demo VALUES ('synthetic')")
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    with pytest.raises(RuntimeError):
        with sqlite_cache.connect() as other:
            other.execute("INSERT INTO demo VALUES ('rollback')")
            raise RuntimeError("simulated failure")
    with sqlite_cache.connect() as check:
        rows = check.execute("SELECT value FROM demo").fetchall()
        assert [row["value"] for row in rows] == ["synthetic"]


@pytest.mark.parametrize("app", ["data_cleaning/app.py", "identity_matching/app.py"])
def test_streamlit_apps_start_without_exceptions(app):
    tested = AppTest.from_file(str(EXAMPLES.parent / app)).run(timeout=20)
    assert not tested.exception
    assert tested.title


def test_real_phonetic_libraries_on_fabricated_names(monkeypatch):
    pytest.importorskip("epitran")
    pytest.importorskip("panphon")
    monkeypatch.setattr(phonetic_similarity, "_PANPHON_DISTANCE", None)
    first = MissingPersonRecord(
        case_id="SYNTH-A",
        full_name_raw="Aline Fiction",
        full_name_normalized="aline fiction",
        name_tokens=["aline", "fiction"],
        disappearance_city_verified="Testville",
    )
    other = MissingPersonRecord(
        case_id="SYNTH-B",
        full_name_raw="Zoran Demonstration",
        full_name_normalized="zoran demonstration",
        name_tokens=["zoran", "demonstration"],
        disappearance_city_verified="Exampleton",
    )
    same = phonetic_similarity.compute_phonetic_similarity(first, first)
    different = phonetic_similarity.compute_phonetic_similarity(first, other)
    assert same.status == "ok"
    assert same.score_final == pytest.approx(1.0)
    assert same.ipa_a
    assert different.status == "ok"
    assert 0 <= different.score_final < 1.0
