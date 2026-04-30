from __future__ import annotations

from src.matching.matching_engine import run_matching
from src.models import AnalysisConfig, BlockingConfig, MissingPersonRecord


def record(
    case_id: str,
    full_name: str,
    date_value: str = "2024-01-01",
    lat: float = 0.0,
    lon: float = 0.0,
) -> MissingPersonRecord:
    return MissingPersonRecord(
        case_id=case_id,
        full_name_raw=full_name,
        full_name_normalized=full_name,
        name_tokens=full_name.split(),
        disappearance_city_verified="city",
        latitude=lat,
        longitude=lon,
        disappearance_date_parsed=date_value,
    )


def config(
    analysis_mode: str,
    investigation_mode: str = "global",
    selected_case_id: str | None = None,
    threshold: float = 0.0,
    max_results: int = 50,
) -> AnalysisConfig:
    return AnalysisConfig(
        investigation_mode=investigation_mode,
        analysis_mode=analysis_mode,
        selected_case_id=selected_case_id,
        enabled_filters={"name": True, "phonetic": False, "geographic": False, "temporal": False},
        filter_parameters={"min_score_threshold": threshold, "max_results": max_results},
    )


def records() -> list[MissingPersonRecord]:
    return [
        record("A", "juan pablo payar", "2024-01-01", 0, 0),
        record("B", "payar juan pablo", "2024-01-02", 0, 0),
        record("C", "maria gomez", "2025-01-01", 50, 50),
        record("D", "carlos ruiz", "2025-05-01", 60, 60),
    ]


def test_exhaustive_compares_all_pairs() -> None:
    result = run_matching(records(), config("exhaustive"))

    assert result.status == "ok"
    assert result.theoretical_pairs_count == 6
    assert result.compared_pairs_count == 6


def test_standard_blocking_compares_less_or_equal_than_exhaustive() -> None:
    exhaustive = run_matching(records(), config("exhaustive"))
    blocking = run_matching(records(), config("standard_blocking"), BlockingConfig())

    assert blocking.compared_pairs_count <= exhaustive.compared_pairs_count
    assert blocking.blocking_result is not None


def test_custom_placeholder_returns_not_implemented() -> None:
    result = run_matching(records(), config("custom_placeholder"))

    assert result.status == "not_implemented"
    assert "prochaine version" in result.explanation


def test_single_case_exhaustive_compares_n_minus_one_pairs() -> None:
    result = run_matching(records(), config("exhaustive", "single_case", "A"))

    assert result.theoretical_pairs_count == 3
    assert result.compared_pairs_count == 3


def test_single_case_blocking_only_uses_target_pairs() -> None:
    result = run_matching(records(), config("standard_blocking", "single_case", "A"), BlockingConfig())

    assert result.blocking_result is not None
    assert result.theoretical_pairs_count == 3
    assert all("A" in {pair.id_A, pair.id_B} for pair in result.results)


def test_results_are_sorted_descending() -> None:
    result = run_matching(records(), config("exhaustive"))
    scores = [pair.score_final for pair in result.results if pair.score_final is not None]

    assert scores == sorted(scores, reverse=True)


def test_min_score_threshold_filters_results() -> None:
    result = run_matching(records(), config("exhaustive", threshold=0.99))

    assert all((pair.score_final or 0) >= 0.99 for pair in result.results)


def test_max_results_limits_results() -> None:
    result = run_matching(records(), config("exhaustive", max_results=2))

    assert len(result.results) <= 2

