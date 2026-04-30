from __future__ import annotations

from src.matching import pairwise_cache
from src.models import AnalysisConfig, MissingPersonRecord
from src.similarity.combined_similarity import compute_combined_similarity


def record(case_id: str, full_name: str) -> MissingPersonRecord:
    return MissingPersonRecord(
        case_id=case_id,
        full_name_raw=full_name,
        full_name_normalized=full_name,
        name_tokens=full_name.split(),
        disappearance_city_verified="city",
    )


def test_make_pair_key_is_order_independent() -> None:
    assert pairwise_cache.make_pair_key("A", "B") == pairwise_cache.make_pair_key("B", "A")


def test_make_config_hash_changes_when_parameters_change() -> None:
    config_a = AnalysisConfig(filter_parameters={"min_score_threshold": 0.5})
    config_b = AnalysisConfig(filter_parameters={"min_score_threshold": 0.8})

    assert pairwise_cache.make_config_hash(config_a) != pairwise_cache.make_config_hash(config_b)


def test_store_then_get_works(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pairwise_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pairwise_cache, "CACHE_FILE", tmp_path / "pairwise_scores.csv")
    config = AnalysisConfig(enabled_filters={"name": True, "phonetic": False, "geographic": False, "temporal": False})
    result = compute_combined_similarity(record("A", "juan"), record("B", "juan"), config)

    pairwise_cache.store_pair_result("A", "B", config, result)
    cached = pairwise_cache.get_cached_result("B", "A", config)

    assert cached is not None
    assert cached["cache_key"] == "A__B"
    assert "name_direct_full_name_score" in cached


def test_store_pair_results_batch_and_lookup(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pairwise_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pairwise_cache, "CACHE_FILE", tmp_path / "pairwise_scores.csv")
    config = AnalysisConfig(enabled_filters={"name": True, "phonetic": False, "geographic": False, "temporal": False})
    result_ab = compute_combined_similarity(record("A", "juan"), record("B", "juan"), config)
    result_ac = compute_combined_similarity(record("A", "juan"), record("C", "maria"), config)

    pairwise_cache.store_pair_results_batch([("A", "B", result_ab), ("A", "C", result_ac)], config)
    lookup = pairwise_cache.build_cache_lookup(config)

    assert set(lookup) == {"A__B", "A__C"}
    assert lookup["A__B"]["name_reordered_full_name_score"] is not None


def test_clear_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pairwise_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(pairwise_cache, "CACHE_FILE", tmp_path / "pairwise_scores.csv")
    pairwise_cache.save_cache(pairwise_cache.load_cache())

    pairwise_cache.clear_cache()

    assert not pairwise_cache.CACHE_FILE.exists()
