import pandas as pd
import pytest

from identity_cleaning.cities import build_city_references, resolve_city
from identity_cleaning.mapping import ColumnMapping
from identity_cleaning.normalization import compact_text, name_tokens, normalize_text, parse_disappearance_date
from identity_cleaning.pipeline import HumanDecision, build_clean_database


def sample_refs():
    df = pd.DataFrame(
        [
            {"city": "Saint-Etienne", "department": "Loire", "lat": 45.4397, "lon": 4.3872},
            {"city": "San Jose", "department": "San Jose", "lat": 9.9281, "lon": -84.0907},
            {"city": "Bogota", "department": "Bogota D.C.", "lat": 4.711, "lon": -74.0721},
            {"city": "Bogota, Distrito Capital", "department": "Bogota, D.C.", "lat": 4.6502, "lon": -74.1074},
            {"city": "Medellin", "department": "Antioquia", "lat": 6.2442, "lon": -75.5812},
            {"city": "Calido", "department": "Valle", "lat": 5.0, "lon": -76.0},
            {"city": "Santiago de Cali", "department": "Valle del Cauca", "lat": 3.4516, "lon": -76.532},
            {"city": "Springfield", "department": "A", "lat": 1.0, "lon": 2.0},
            {"city": "Springfield", "department": "B", "lat": 3.0, "lon": 4.0},
        ]
    )
    return build_city_references(df, "city", "department", "lat", "lon")


def person_mapping():
    return ColumnMapping(
        {
            "id": "case_id",
            "full": "full_name",
            "first": "first_name",
            "last": "last_name",
            "city": "disappearance_city",
            "dept": "disappearance_department",
            "date": "disappearance_date",
        }
    )


def test_text_normalization_and_tokens():
    assert normalize_text(" Saint-Etienne  ") == "saint etienne"
    assert normalize_text("Bogota") == "bogota"
    assert compact_text("San Jose") == "sanjose"
    assert name_tokens("Jean Jean Dupont") == ["dupont", "jean"]


def test_date_parsing():
    assert parse_disappearance_date("18/02/2005") == "2005-02-18"
    assert parse_disappearance_date("date inconnue") == ""


def test_manual_mapping_rejects_duplicate_fields():
    mapping = ColumnMapping({"a": "case_id", "b": "case_id", "c": "ignore"})
    with pytest.raises(ValueError):
        mapping.require_unique_fields()


def test_exact_city_department_match():
    result = resolve_city("Saint Etienne", "Loire", sample_refs())
    assert result.status == "exact_city_department_match"
    assert result.selected.latitude == 45.4397


def test_exact_city_only_match():
    result = resolve_city("Bogota", "", sample_refs())
    assert result.status == "exact_city_match"
    assert result.selected.city_raw == "Bogota"


def test_city_with_accent_and_hyphen():
    result = resolve_city("Saint-Etienne", "Loire", sample_refs())
    assert result.status == "exact_city_department_match"
    result = resolve_city("Bogota", "Bogota D.C.", sample_refs())
    assert result.selected.city_raw == "Bogota"


def test_abbreviation_and_extra_words_have_limited_impact():
    result = resolve_city("BOGOTA DC", "BOGOTA D C", sample_refs())
    assert result.status == "fuzzy_city_candidates"
    assert result.selected is None
    assert result.candidates[0].reference.city_raw == "Bogota, Distrito Capital"
    assert result.candidates[0].score > 90


def test_city_compacted_vs_words():
    result = resolve_city("SanJose", "San Jose", sample_refs())
    assert result.status == "exact_city_department_match"
    assert result.selected.city_raw == "San Jose"


def test_light_typo_proposes_candidate():
    result = resolve_city("Medelin", "Antioquia", sample_refs())
    assert result.status == "fuzzy_city_candidates"
    assert result.selected is None
    assert result.needs_human_validation
    assert result.candidates[0].reference.city_raw == "Medellin"


def test_department_similarity_counts_in_score():
    result = resolve_city("Medelin", "Antioqia", sample_refs())
    assert result.candidates[0].reference.city_raw == "Medellin"
    assert "departement similaire" in result.candidates[0].explanation


def test_city_substring_is_not_enough_for_match():
    result = resolve_city("Cali", "Valle", sample_refs())
    assert result.status == "fuzzy_city_candidates"
    assert result.selected is None
    assert result.needs_human_validation
    assert {candidate.reference.city_raw for candidate in result.candidates} >= {"Calido", "Santiago de Cali"}


def test_composed_city_with_matching_department_is_proposed_first():
    result = resolve_city("Cali", "Valle del Cauca", sample_refs())
    assert result.status == "fuzzy_city_candidates"
    assert result.selected is None
    assert result.candidates[0].reference.city_raw == "Santiago de Cali"
    assert "Calido" in {candidate.reference.city_raw for candidate in result.candidates}


def test_exact_city_with_wrong_department_requires_validation():
    result = resolve_city("Medellin", "Loire", sample_refs())
    assert result.status == "city_match_department_mismatch"
    assert result.selected is None
    assert result.needs_human_validation


def test_ambiguous_city():
    result = resolve_city("Springfield", "", sample_refs())
    assert result.status == "ambiguous_city_match"
    assert result.needs_human_validation


def test_missing_and_unknown_city():
    assert resolve_city("", "", sample_refs()).status == "missing_city"
    assert resolve_city("No City Here", "", sample_refs()).status == "unresolved_city"


def test_human_correction_and_final_export_columns():
    refs = sample_refs()
    persons = pd.DataFrame(
        [{"id": "1", "full": "Duvan Banol Bueno", "first": "Duvan", "last": "Banol Bueno", "city": "Springfield", "dept": "", "date": "2005-02-18"}]
    )
    decisions = {"1": HumanDecision("accept_candidate", reference_id=refs[-2].ref_id)}
    clean = build_clean_database(persons, person_mapping(), refs, decisions)
    assert clean.loc[0, "city_resolution_status"] == "human_validated_city"
    assert clean.loc[0, "human_validation_decision"] == "accept_candidate"
    csv_text = clean.to_csv(index=False)
    assert "case_id" in csv_text
    assert "full_name_normalized" in clean.columns
