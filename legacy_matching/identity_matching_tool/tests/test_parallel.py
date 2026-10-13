from __future__ import annotations

import pytest

from src.cache import sqlite_cache
from src.matching import pairwise_cache
from src.matching.pair_generation import generate_all_pairs, generate_single_case_pairs
from src.matching.parallel import parallel_compute_pair_scores, should_use_parallelism
from src.models import AnalysisConfig, MissingPersonRecord
from src.similarity.combined_similarity import compute_combined_similarity


def record(case_id: str, full_name: str) -> MissingPersonRecord:
    return MissingPersonRecord(
        case_id=case_id,
        full_name_raw=full_name,
        full_name_normalized=full_name,
        name_tokens=full_name.split(),
        disappearance_city_verified="city",
        latitude=0.0,
        longitude=0.0,
        disappearance_date_parsed="2024-01-01",
    )


def records() -> list[MissingPersonRecord]:
    return [
        record("A", "juan pablo payar"),
        record("B", "payar juan pablo"),
        record("C", "jose perez"),
        record("D", "maria gomez"),
        record("E", "carlos ruiz"),
    ]


def config() -> AnalysisConfig:
    return AnalysisConfig(
        enabled_filters={"name": True, "phonetic": False, "geographic": False, "temporal": False},
        filter_parameters={"min_score_threshold": 0.0, "max_results": 50},
    )


def normalize_results(results):
    return {
        tuple(sorted([result.id_A, result.id_B])): result.score_final
        for result in results
    }


def test_should_use_parallelism_rules() -> None:
    assert not should_use_parallelism(1000, 1, False)
    assert not should_use_parallelism(100, 2, False)
    assert should_use_parallelism(500, 2, False)
    assert not should_use_parallelism(99, 2, True)
    assert should_use_parallelism(100, 2, True)


def test_parallel_results_match_sequential_results() -> None:
    data = records()
    pairs = generate_all_pairs(data)
    records_by_id = {item.case_id: item for item in data}

    sequential = parallel_compute_pair_scores(pairs, records_by_id, config(), num_workers=1, chunk_size=3, use_cache=False)
    parallel = parallel_compute_pair_scores(pairs, records_by_id, config(), num_workers=2, chunk_size=3, use_cache=False)

    assert normalize_results(parallel).keys() == normalize_results(sequential).keys()
    for pair_key, score in normalize_results(sequential).items():
        assert normalize_results(parallel)[pair_key] == pytest.approx(score)


def test_parallel_does_not_create_duplicates() -> None:
    data = records()
    pairs = generate_all_pairs(data)
    results = parallel_compute_pair_scores(
        pairs,
        {item.case_id: item for item in data},
        config(),
        num_workers=2,
        chunk_size=2,
        use_cache=False,
    )

    keys = [tuple(sorted([result.id_A, result.id_B])) for result in results]
    assert len(keys) == len(set(keys))


def test_single_case_parallel_matches_sequential() -> None:
    data = records()
    pairs = generate_single_case_pairs(data, "A")
    records_by_id = {item.case_id: item for item in data}

    sequential = parallel_compute_pair_scores(pairs, records_by_id, config(), num_workers=1, chunk_size=2, use_cache=False)
    parallel = parallel_compute_pair_scores(pairs, records_by_id, config(), num_workers=2, chunk_size=2, use_cache=False)

    assert normalize_results(parallel) == pytest.approx(normalize_results(sequential))


def test_cache_skips_existing_pairs_and_adds_new_pairs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sqlite_cache, "CACHE_DIR", tmp_path)
    data = records()[:3]
    records_by_id = {item.case_id: item for item in data}
    analysis_config = config()
    cached_similarity = compute_combined_similarity(records_by_id["A"], records_by_id["B"], analysis_config)
    pairwise_cache.insert_component_rows(
        [
            pairwise_cache.component_row_for_records(
                "A",
                "B",
                records_by_id["A"],
                records_by_id["B"],
                analysis_config,
                cached_similarity,
            )
        ]
    )

    results = parallel_compute_pair_scores(
        [("A", "B"), ("A", "C")],
        records_by_id,
        analysis_config,
        num_workers=2,
        chunk_size=1,
        use_cache=True,
    )

    assert sum(result.from_cache for result in results) == 1
    cache = pairwise_cache.load_cache()
    assert len(cache) == 2


def test_component_cache_reuses_stable_scores_and_recomputes_threshold_scores(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sqlite_cache, "CACHE_DIR", tmp_path)
    record_a = record("A", "juan pablo")
    record_b = record("B", "juan pablo")
    record_b.latitude = 0.1
    record_b.longitude = 0.0
    record_b.disappearance_date_parsed = "2024-01-21"
    records_by_id = {"A": record_a, "B": record_b}
    first_config = AnalysisConfig(
        enabled_filters={"name": True, "phonetic": False, "geographic": True, "temporal": True},
        filter_parameters={
            "geographic_distance_min_km": 0.0,
            "geographic_distance_max_km": 50.0,
            "temporal_max_days": 40,
            "max_results": 50,
            "min_score_threshold": 0.0,
        },
    )
    second_config = AnalysisConfig(
        enabled_filters={"name": True, "phonetic": False, "geographic": True, "temporal": True},
        filter_parameters={
            "geographic_distance_min_km": 0.0,
            "geographic_distance_max_km": 10.0,
            "temporal_max_days": 20,
            "max_results": 50,
            "min_score_threshold": 0.0,
        },
    )

    first = parallel_compute_pair_scores(
        [("A", "B")],
        records_by_id,
        first_config,
        num_workers=1,
        chunk_size=1,
        use_cache=True,
    )[0]

    def fail_if_recomputed(*args, **kwargs):
        raise AssertionError("Stable pair components should have been reused from cache")

    monkeypatch.setattr("src.matching.parallel.compute_combined_similarity", fail_if_recomputed)
    second = parallel_compute_pair_scores(
        [("A", "B")],
        records_by_id,
        second_config,
        num_workers=1,
        chunk_size=1,
        use_cache=True,
    )[0]

    assert second.from_cache
    assert second.score_name == pytest.approx(first.score_name)
    assert second.distance_km == pytest.approx(first.distance_km)
    assert second.distance_days == first.distance_days
    assert second.score_geographic != pytest.approx(first.score_geographic)
    assert second.score_temporal != pytest.approx(first.score_temporal)
    assert second.score_final != pytest.approx(first.score_final)


def test_num_workers_one_uses_sequential_path() -> None:
    data = records()
    pairs = generate_all_pairs(data)
    results = parallel_compute_pair_scores(
        pairs,
        {item.case_id: item for item in data},
        config(),
        num_workers=1,
        chunk_size=2,
        use_cache=False,
    )

    assert results
    assert all(result.pair_num_workers == 1 for result in results)


def test_chunking_processes_all_pairs_when_not_divisible() -> None:
    data = records()
    pairs = generate_all_pairs(data)
    results = parallel_compute_pair_scores(
        pairs,
        {item.case_id: item for item in data},
        config(),
        num_workers=2,
        chunk_size=4,
        use_cache=False,
    )

    assert len(results) == len(pairs)


def test_progress_callback_reports_chunks_in_sequential_mode() -> None:
    data = records()
    progress_events: list[tuple[str, int, int]] = []

    parallel_compute_pair_scores(
        generate_all_pairs(data),
        {item.case_id: item for item in data},
        config(),
        num_workers=1,
        chunk_size=4,
        use_cache=False,
        progress_callback=lambda label, current, total: progress_events.append((label, current, total)),
    )

    assert progress_events
    assert any("chunk" in label for label, _, _ in progress_events)
    assert progress_events[-1][1] == progress_events[-1][2]
