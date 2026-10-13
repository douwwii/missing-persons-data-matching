from __future__ import annotations

import sys
import types

import pytest

from identity_matching.cache import sqlite_cache
from identity_matching.models import MissingPersonRecord
from identity_matching.similarity import phonetic_similarity
from identity_matching.similarity.phonetic_similarity import (
    compute_phonetic_similarity,
    precompute_ipa_for_records,
)


def record(full_name: str) -> MissingPersonRecord:
    return MissingPersonRecord(
        case_id="x",
        full_name_raw=full_name,
        full_name_normalized=full_name,
        name_tokens=full_name.split(),
        disappearance_city_verified="city",
    )


def install_fake_phonetic_modules(monkeypatch, distance_value: float = 0.0) -> None:
    monkeypatch.setattr(phonetic_similarity, "_PANPHON_DISTANCE", None)
    epitran_module = types.ModuleType("epitran")

    class FakeEpitran:
        def __init__(self, lang: str) -> None:
            if lang == "invalid-lang":
                raise ValueError("invalid")

        def transliterate(self, value: str) -> str:
            return value

    class FakeDistance:
        def feature_edit_distance(self, ipa_a: str, ipa_b: str) -> float:
            if ipa_a == ipa_b:
                return 0.0
            return distance_value

    panphon_module = types.ModuleType("panphon")
    distance_module = types.ModuleType("panphon.distance")
    distance_module.Distance = FakeDistance
    panphon_module.distance = distance_module
    epitran_module.Epitran = FakeEpitran

    monkeypatch.setitem(sys.modules, "epitran", epitran_module)
    monkeypatch.setitem(sys.modules, "panphon", panphon_module)
    monkeypatch.setitem(sys.modules, "panphon.distance", distance_module)


def test_identical_name_when_dependencies_available(monkeypatch) -> None:
    install_fake_phonetic_modules(monkeypatch)

    result = compute_phonetic_similarity(record("juan pablo"), record("juan pablo"))

    assert result.status == "ok"
    assert result.score_final == 1.0
    assert result.ipa_a == "juan pablo"
    assert result.explanation


def test_phonetically_close_names_when_dependencies_available(monkeypatch) -> None:
    install_fake_phonetic_modules(monkeypatch, distance_value=1.0)

    result = compute_phonetic_similarity(record("jose perez"), record("jose peres"))

    assert result.status == "ok"
    assert result.score_final is not None
    assert 0 <= result.score_final <= 1
    # One edit over ten IPA characters gives 0.9 before the documented power calibration.
    assert result.score_final == pytest.approx(0.7684334714209162)


def test_reordered_phonetic_comparison_handles_inverted_tokens(monkeypatch) -> None:
    install_fake_phonetic_modules(monkeypatch)

    result = compute_phonetic_similarity(record("juan pablo payar"), record("payar juan pablo"))

    assert result.status == "ok"
    assert result.score_final == 1.0
    assert result.ipa_a == result.ipa_b
    assert "sorted by IPA order" in result.explanation


def test_missing_name() -> None:
    result = compute_phonetic_similarity(record(""), record("juan"))

    assert result.status == "missing_name"
    assert result.score_final is None


def test_invalid_language(monkeypatch) -> None:
    install_fake_phonetic_modules(monkeypatch)

    result = compute_phonetic_similarity(record("juan"), record("juan"), lang="invalid-lang")

    assert result.status == "invalid_language"
    assert result.score_final is None


def test_unavailable_when_epitran_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "epitran", None)

    result = compute_phonetic_similarity(record("juan"), record("juan"))

    assert result.status == "unavailable"
    assert result.score_final is None


def test_precompute_ipa_cache_is_used_for_pair_comparison(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sqlite_cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(phonetic_similarity, "_PANPHON_DISTANCE", None)
    install_fake_phonetic_modules(monkeypatch, distance_value=1.0)
    record_a = record("jose perez")
    record_b = record("jose peres")

    stats = precompute_ipa_for_records([record_a, record_b])
    result = compute_phonetic_similarity(record_a, record_b)

    assert stats["computed"] == 2
    assert sqlite_cache.cache_db_path().exists()
    assert result.status == "ok"
    assert result.score_final is not None
    assert result.ipa_a == "jose perez"
    assert result.ipa_b == "jose peres"
