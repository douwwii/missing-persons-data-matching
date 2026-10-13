from __future__ import annotations

from identity_matching.matching.matching_engine import run_matching, run_matching_on_pair_results
from identity_matching.models import (
    AnalysisConfig,
    BlockingConfig,
    MissingPersonRecord,
    PairMatchingResult,
    ParallelConfig,
)


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
    assert "future version" in result.explanation


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


def source_pair(id_a: str, id_b: str) -> PairMatchingResult:
    record_by_id = {record.case_id: record for record in records()}
    left = record_by_id[id_a]
    right = record_by_id[id_b]
    return PairMatchingResult(
        id_A=id_a,
        id_B=id_b,
        name_A=left.full_name_raw,
        name_B=right.full_name_raw,
        score_final=0.0,
    )


def test_refinement_scores_only_visible_source_pairs() -> None:
    source_results = [source_pair("A", "B"), source_pair("C", "D")]

    result = run_matching_on_pair_results(
        records(),
        source_results,
        config("exhaustive"),
        ParallelConfig(enable_parallelism=False),
    )

    compared_pairs = {frozenset((pair.id_A, pair.id_B)) for pair in result.results}
    assert result.status == "ok"
    assert result.analysis_mode == "refinement"
    assert result.theoretical_pairs_count == 2
    assert result.compared_pairs_count == 2
    assert compared_pairs <= {frozenset(("A", "B")), frozenset(("C", "D"))}


def test_refinement_applies_sorting_and_max_results_after_recalculation() -> None:
    source_results = [source_pair("C", "D"), source_pair("A", "B")]

    result = run_matching_on_pair_results(
        records(),
        source_results,
        config("exhaustive", max_results=1),
        ParallelConfig(enable_parallelism=False),
    )

    assert len(result.results) == 1
    assert {result.results[0].id_A, result.results[0].id_B} == {"A", "B"}
    assert result.retained_results_count == 1


def test_refinement_skips_pairs_with_missing_records() -> None:
    source_results = [
        source_pair("A", "B"),
        PairMatchingResult("A", "Z", "juan pablo payar", "missing", 0.0),
    ]

    result = run_matching_on_pair_results(
        records(),
        source_results,
        config("exhaustive"),
        ParallelConfig(enable_parallelism=False),
    )

    assert result.theoretical_pairs_count == 2
    assert result.compared_pairs_count == 1
    assert all("Z" not in {pair.id_A, pair.id_B} for pair in result.results)
