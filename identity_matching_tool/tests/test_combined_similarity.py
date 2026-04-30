from __future__ import annotations

import pytest

from src.models import AnalysisConfig, MissingPersonRecord
from src.similarity.combined_similarity import compute_combined_similarity


def record(
    full_name: str,
    lat: float | None = 0,
    lon: float | None = 0,
    date_value: str | None = "2024-01-01",
) -> MissingPersonRecord:
    return MissingPersonRecord(
        case_id="x",
        full_name_raw=full_name,
        full_name_normalized=full_name,
        name_tokens=full_name.split(),
        disappearance_city_verified="city",
        latitude=lat,
        longitude=lon,
        disappearance_date_parsed=date_value,
    )


def config(enabled: dict[str, bool], parameters: dict[str, float | int] | None = None) -> AnalysisConfig:
    base_parameters = {
        "geographic_distance_min_km": 10.0,
        "geographic_distance_max_km": 100.0,
        "temporal_max_days": 365,
        "name_weight": 1.0,
        "geographic_weight": 1.0,
        "temporal_weight": 1.0,
        "phonetic_weight": 1.0,
    }
    if parameters:
        base_parameters.update(parameters)
    return AnalysisConfig(enabled_filters=enabled, filter_parameters=base_parameters)


def test_combines_multiple_scores() -> None:
    result = compute_combined_similarity(
        record("juan pablo", date_value="2024-01-01"),
        record("juan pablo", date_value="2024-07-01"),
        config({"name": True, "phonetic": False, "geographic": True, "temporal": True}),
    )

    assert result.status == "ok"
    assert result.score_final is not None
    assert 0 <= result.score_final <= 1
    assert result.name_result is not None
    assert result.geo_result is not None
    assert result.temporal_result is not None
    assert result.explanation


def test_disabled_filter_is_ignored() -> None:
    result = compute_combined_similarity(
        record("juan"),
        record("maria"),
        config({"name": False, "phonetic": False, "geographic": True, "temporal": False}),
    )

    assert result.name_result is None
    assert result.geo_result is not None
    assert result.score_final == 1.0


def test_non_calculable_score_is_ignored() -> None:
    result = compute_combined_similarity(
        record("juan", lat=None, lon=None),
        record("juan", lat=0, lon=0),
        config({"name": True, "phonetic": False, "geographic": True, "temporal": False}),
    )

    assert result.geo_result is not None
    assert result.geo_result.status == "missing_coordinates"
    assert result.score_final == pytest.approx(result.name_result.score_final)
    assert "ignorees" in result.explanation


def test_different_weights_change_score() -> None:
    record_a = record("juan", lat=0, lon=0)
    record_b = record("maria", lat=0, lon=0)

    equal_weight = compute_combined_similarity(
        record_a,
        record_b,
        config({"name": True, "phonetic": False, "geographic": True, "temporal": False}),
    )
    geo_heavy = compute_combined_similarity(
        record_a,
        record_b,
        config(
            {"name": True, "phonetic": False, "geographic": True, "temporal": False},
            {"geographic_weight": 5.0},
        ),
    )

    assert equal_weight.score_final is not None
    assert geo_heavy.score_final is not None
    assert geo_heavy.score_final > equal_weight.score_final


def test_global_explanation_mentions_order_mismatch() -> None:
    result = compute_combined_similarity(
        record("juan pablo payar"),
        record("payar juan pablo"),
        config({"name": True, "phonetic": False, "geographic": False, "temporal": False}),
    )

    assert result.name_result is not None
    assert result.name_result.order_mismatch_suspected is True
    assert "ordre different" in result.explanation

