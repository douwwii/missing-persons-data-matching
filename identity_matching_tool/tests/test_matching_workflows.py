from __future__ import annotations

from src.matching.global_matching import find_global_matches
from src.matching.single_case import find_single_case_matches
from src.models import AnalysisConfig, MissingPersonRecord


def record(case_id: str, full_name: str, lat: float = 0, lon: float = 0) -> MissingPersonRecord:
    return MissingPersonRecord(
        case_id=case_id,
        full_name_raw=full_name,
        full_name_normalized=full_name,
        name_tokens=full_name.split(),
        disappearance_city_verified="city",
        latitude=lat,
        longitude=lon,
        disappearance_date_parsed="2024-01-01",
    )


def config(mode: str = "single_case", selected_case_id: str | None = "A") -> AnalysisConfig:
    return AnalysisConfig(
        investigation_mode=mode,
        selected_case_id=selected_case_id,
        enabled_filters={"name": True, "phonetic": False, "geographic": False, "temporal": False},
        filter_parameters={"min_score_threshold": 0.5, "max_results": 10},
    )


def test_single_case_matches_return_candidates() -> None:
    records = [record("A", "juan pablo payar"), record("B", "payar juan pablo"), record("C", "maria gomez")]

    results = find_single_case_matches(records, "A", config())

    assert results
    assert results[0][0].case_id == "B"
    assert results[0][1].score_final is not None


def test_global_matches_return_pairs() -> None:
    records = [record("A", "juan pablo payar"), record("B", "payar juan pablo"), record("C", "maria gomez")]

    results = find_global_matches(records, config(mode="global", selected_case_id=None))

    assert results
    assert {results[0][0].case_id, results[0][1].case_id} == {"A", "B"}

