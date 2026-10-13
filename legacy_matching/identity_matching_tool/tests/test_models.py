from __future__ import annotations

from datetime import date

from src.models import AnalysisConfig, MissingPersonRecord


def test_missing_person_record_creation() -> None:
    record = MissingPersonRecord(
        case_id="MP-001",
        full_name_raw="Marie Dupont",
        full_name_normalized="marie dupont",
        name_tokens=["marie", "dupont"],
        disappearance_city_verified="Lyon",
        latitude=45.764,
        longitude=4.8357,
        disappearance_date_parsed=date(2024, 2, 18),
        raw_data={"case_id": "MP-001"},
    )

    assert record.case_id == "MP-001"
    assert record.name_tokens == ["marie", "dupont"]
    assert record.raw_data["case_id"] == "MP-001"


def test_analysis_config_creation() -> None:
    config = AnalysisConfig(
        investigation_mode="global",
        enabled_filters={"name": True, "phonetic": False, "geographic": True, "temporal": True},
        filter_parameters={
            "geographic_distance_min_km": 5.0,
            "geographic_distance_max_km": 50.0,
            "temporal_max_days": 365,
            "max_results": 50,
            "min_score_threshold": 0.7,
        },
    )

    assert config.investigation_mode == "global"
    assert config.enabled_filters["geographic"] is True


def test_analysis_config_defaults_are_coherent() -> None:
    config = AnalysisConfig()

    assert config.investigation_mode == "single_case"
    assert config.enabled_filters == {
        "name": True,
        "phonetic": False,
        "geographic": False,
        "temporal": False,
    }
    assert config.filter_parameters["geographic_distance_min_km"] < config.filter_parameters[
        "geographic_distance_max_km"
    ]
    assert 0 <= config.filter_parameters["min_score_threshold"] <= 1
    assert config.filter_parameters["max_results"] > 0

