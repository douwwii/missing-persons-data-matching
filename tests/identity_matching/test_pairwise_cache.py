from __future__ import annotations

from identity_matching.cache import sqlite_cache
from identity_matching.matching import pairwise_cache
from identity_matching.models import AnalysisConfig, MissingPersonRecord
from identity_matching.similarity.combined_similarity import compute_combined_similarity


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


def test_make_pair_key_is_order_independent() -> None:
    assert pairwise_cache.make_pair_key("A", "B") == pairwise_cache.make_pair_key("B", "A")


def test_component_hash_ignores_recomputable_threshold_parameters() -> None:
    record_a = record("A", "juan")
    record_b = record("B", "juan")
    config_a = AnalysisConfig(filter_parameters={"temporal_max_days": 30, "geographic_distance_max_km": 20.0})
    config_b = AnalysisConfig(
        filter_parameters={"temporal_max_days": 365, "geographic_distance_max_km": 50.0}
    )

    assert pairwise_cache.make_component_hash(
        record_a, record_b, config_a
    ) == pairwise_cache.make_component_hash(
        record_a,
        record_b,
        config_b,
    )


def test_store_then_get_works(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sqlite_cache, "CACHE_DIR", tmp_path)
    config = AnalysisConfig(
        enabled_filters={"name": True, "phonetic": False, "geographic": False, "temporal": False}
    )
    record_a = record("A", "juan")
    record_b = record("B", "juan")
    records_by_id = {"A": record_a, "B": record_b}
    result = compute_combined_similarity(record_a, record_b, config)
    row = pairwise_cache.component_row_for_records("A", "B", record_a, record_b, config, result)

    pairwise_cache.insert_component_rows([row])
    cached = pairwise_cache.get_cached_result("B", "A", records_by_id, config)

    assert cached is not None
    assert cached["cache_key"] == "A__B"
    assert "name_direct_full_name_score" in cached


def test_insert_component_rows_and_lookup(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sqlite_cache, "CACHE_DIR", tmp_path)
    config = AnalysisConfig(
        enabled_filters={"name": True, "phonetic": False, "geographic": False, "temporal": False}
    )
    record_a = record("A", "juan")
    record_b = record("B", "juan")
    record_c = record("C", "maria")
    records_by_id = {"A": record_a, "B": record_b, "C": record_c}
    result_ab = compute_combined_similarity(record_a, record_b, config)
    result_ac = compute_combined_similarity(record_a, record_c, config)
    rows = [
        pairwise_cache.component_row_for_records("A", "B", record_a, record_b, config, result_ab),
        pairwise_cache.component_row_for_records("A", "C", record_a, record_c, config, result_ac),
    ]

    pairwise_cache.insert_component_rows(rows)
    lookup = pairwise_cache.build_component_cache_lookup(records_by_id, config)

    assert set(lookup) == {"A__B", "A__C"}
    assert lookup["A__B"]["name_reordered_full_name_score"] is not None


def test_component_cache_is_selective(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sqlite_cache, "CACHE_DIR", tmp_path)
    config = AnalysisConfig(
        enabled_filters={"name": True, "phonetic": False, "geographic": False, "temporal": False}
    )
    record_a = record("A", "juan")
    record_b = record("B", "juan")
    record_b.latitude = 40.0
    record_b.longitude = 40.0
    record_b.disappearance_date_parsed = "2025-01-01"
    result = compute_combined_similarity(record_a, record_b, config)
    row = pairwise_cache.component_row_for_records("A", "B", record_a, record_b, config, result)

    pairwise_cache.insert_component_rows([row])

    assert pairwise_cache.get_cached_result("A", "B", {"A": record_a, "B": record_b}, config) is None
    assert pairwise_cache.load_cache().empty


def test_component_cache_policy_version_for_stored_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sqlite_cache, "CACHE_DIR", tmp_path)
    config = AnalysisConfig(
        enabled_filters={"name": True, "phonetic": False, "geographic": False, "temporal": False}
    )
    record_a = record("A", "juan")
    record_b = record("B", "juan")
    result = compute_combined_similarity(record_a, record_b, config)
    row = pairwise_cache.component_row_for_records("A", "B", record_a, record_b, config, result)

    pairwise_cache.insert_component_rows([row])

    assert pairwise_cache.load_cache().iloc[0]["cache_policy_version"] == "two_scores_over_0_5_components_v2"


def test_clear_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sqlite_cache, "CACHE_DIR", tmp_path)
    pairwise_cache.save_cache(pairwise_cache.load_cache())

    pairwise_cache.clear_cache()

    assert pairwise_cache.load_cache().empty
