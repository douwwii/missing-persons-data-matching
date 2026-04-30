from __future__ import annotations

from collections import Counter
import csv
import json

import pytest

from src.matching.blocking import (
    are_counters_compatible,
    build_blocking_signature,
    clear_blocking_index_cache,
    date_diff_days,
    export_blocking_index_to_disk,
    generate_standard_blocking_pairs,
    get_or_build_blocking_index,
    haversine_km,
    phonetic_onset,
)
from src.models import BlockingConfig, MissingPersonRecord


def record(
    case_id: str,
    name: str,
    date_value: str = "2024-01-01",
    lat: float = 0.0,
    lon: float = 0.0,
) -> MissingPersonRecord:
    return MissingPersonRecord(
        case_id=case_id,
        full_name_raw=name,
        full_name_normalized=name,
        name_tokens=name.split(),
        disappearance_city_verified="city",
        disappearance_department_verified="dept",
        latitude=lat,
        longitude=lon,
        disappearance_date_parsed=date_value,
    )


def test_initials_bag_is_correct() -> None:
    signature = build_blocking_signature(record("A", "juan pablo payar"))

    assert signature.initials_bag == {"J": 1, "P": 2}


def test_blocking_signature_falls_back_to_full_name_tokens_when_name_tokens_is_single_string() -> None:
    item = record("A", "juan pablo payar")
    item.name_tokens = ["juan pablo payar"]

    signature = build_blocking_signature(item)

    assert signature.tokens == ["juan", "pablo", "payar"]
    assert signature.initials_bag == {"J": 1, "P": 2}


def test_blocking_signature_prefers_full_name_normalized_over_incomplete_name_tokens() -> None:
    item = record("A", "juan pablo payar")
    item.name_tokens = ["juan"]

    signature = build_blocking_signature(item)

    assert signature.tokens == ["juan", "pablo", "payar"]
    assert signature.initials_bag == {"J": 1, "P": 2}


def test_are_counters_compatible() -> None:
    assert are_counters_compatible(Counter({"J": 1, "P": 2}), Counter({"J": 1, "P": 2, "E": 1}))
    assert not are_counters_compatible(Counter({"J": 1, "P": 2}), Counter({"J": 1, "R": 1}))


def test_r1_true_with_compatible_initials_and_near_date() -> None:
    result = generate_standard_blocking_pairs(
        [record("A", "juan pablo payar"), record("B", "juan pablo exposito payar", "2024-01-15")]
    )

    assert result.candidate_pairs_count == 1
    assert "R1" in result.candidate_pairs[0].matched_rules


def test_r1_false_when_date_too_far() -> None:
    result = generate_standard_blocking_pairs(
        [
            record("A", "juan pablo payar", "2024-01-01", 0, 0),
            record("B", "juan pablo payar", "2024-05-01", 10, 10),
        ],
        BlockingConfig(date_threshold_name_rules=60, geo_threshold_km=1, date_threshold_geo_rule=1),
    )

    assert not result.candidate_pairs


def test_phonetic_onset_rules() -> None:
    assert phonetic_onset("carlos") == "K"
    assert phonetic_onset("kilo") == "K"
    assert phonetic_onset("queso") == "K"
    assert phonetic_onset("victor") == "B"
    assert phonetic_onset("zapato") == "S"
    assert phonetic_onset("yanez") == "I"
    assert phonetic_onset("juan") == "J"
    assert phonetic_onset("gerardo") == "J"
    assert phonetic_onset("gomez") == "G"


def test_r2_true_with_compatible_onsets_and_near_date() -> None:
    result = generate_standard_blocking_pairs([record("A", "carlos"), record("B", "karlos", "2024-01-02")])

    assert result.candidate_pairs_count == 1
    assert "R2" in result.candidate_pairs[0].matched_rules


def test_r4_true_with_close_space_and_time() -> None:
    result = generate_standard_blocking_pairs(
        [record("A", "abc", "2024-01-01", 48.8566, 2.3522), record("B", "xyz", "2024-01-05", 48.857, 2.353)]
    )

    assert result.candidate_pairs_count == 1
    assert "R4" in result.candidate_pairs[0].matched_rules


def test_r4_false_when_distance_too_large() -> None:
    result = generate_standard_blocking_pairs(
        [
            record("A", "abc", "2024-01-01", 48.8566, 2.3522),
            record("B", "xyz", "2024-01-05", 43.6047, 1.4442),
        ]
    )

    assert not result.candidate_pairs


def test_blocking_keeps_pair_if_any_rule_matches_and_reports_rules() -> None:
    result = generate_standard_blocking_pairs(
        [record("A", "juan pablo payar"), record("B", "juan pablo exposito payar", "2024-01-05")]
    )

    assert result.candidate_pairs_count == 1
    assert set(result.candidate_pairs[0].matched_rules) >= {"R1", "R2", "R4"}


def test_reduction_percentage() -> None:
    records = [
        record("A", "juan pablo", "2024-01-01", 0, 0),
        record("B", "juan pablo", "2024-01-02", 0, 0),
        record("C", "maria gomez", "2025-01-01", 50, 50),
    ]

    result = generate_standard_blocking_pairs(records)

    assert result.theoretical_pairs_count == 3
    assert result.candidate_pairs_count == 1
    assert result.reduction_percentage == pytest.approx(2 / 3)


def test_indexed_blocking_matches_naive_rules_on_small_dataset() -> None:
    data = [
        record("A", "juan pablo payar", "2024-01-01", 0, 0),
        record("B", "juan pablo exposito payar", "2024-01-05", 0, 0),
        record("C", "carlos", "2024-02-01", 1, 1),
        record("D", "karlos", "2024-02-02", 1, 1),
        record("E", "maria gomez", "2025-01-01", 50, 50),
    ]
    config = BlockingConfig()

    indexed = generate_standard_blocking_pairs(data, config)
    indexed_pairs = {tuple(sorted([pair.id_A, pair.id_B])) for pair in indexed.candidate_pairs}
    naive_pairs = _naive_blocking_pairs(data, config)

    assert indexed_pairs == naive_pairs


def test_blocking_index_is_reused_for_same_dataset_and_config(monkeypatch) -> None:
    import src.matching.blocking as blocking

    clear_blocking_index_cache()
    calls = {"count": 0}
    original_builder = blocking.build_standard_blocking_index

    def counting_builder(*args, **kwargs):
        calls["count"] += 1
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(blocking, "build_standard_blocking_index", counting_builder)
    data = [record("A", "juan pablo"), record("B", "juan pablo", "2024-01-02")]
    config = BlockingConfig()

    generate_standard_blocking_pairs(data, config)
    generate_standard_blocking_pairs(data, config)

    assert calls["count"] == 1


def test_blocking_index_is_exported_to_disk(tmp_path) -> None:
    clear_blocking_index_cache()
    data = [record("A", "juan pablo"), record("B", "juan pablo", "2024-01-02")]
    config = BlockingConfig()
    index = get_or_build_blocking_index(data, config)

    export_dir = export_blocking_index_to_disk(index, config, base_dir=tmp_path)

    assert (export_dir / "metadata.json").exists()
    assert (export_dir / "bucket_summary.csv").exists()
    assert (export_dir / "r1_initials_buckets.csv").exists()
    assert (export_dir / "r2_phonetic_onset_buckets.csv").exists()
    assert (export_dir / "r4_geo_time_buckets.csv").exists()
    assert (export_dir / "signatures.csv").exists()


def test_export_contains_multi_initial_buckets_when_full_name_has_multiple_tokens(tmp_path) -> None:
    clear_blocking_index_cache()
    data = [record("A", "juan pablo payar"), record("B", "payar juan pablo")]
    config = BlockingConfig()
    index = get_or_build_blocking_index(data, config)

    export_dir = export_blocking_index_to_disk(index, config, base_dir=tmp_path)
    with (export_dir / "r1_initials_buckets.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert any(json.loads(row["bucket_key"])[0] == ["J", "P", "P"] for row in rows)


def test_blocking_result_exposes_bucket_cache_directory() -> None:
    clear_blocking_index_cache()
    result = generate_standard_blocking_pairs([record("A", "juan pablo"), record("B", "juan pablo")])

    assert result.blocking_index_cache_dir is not None


def _naive_blocking_pairs(records: list[MissingPersonRecord], config: BlockingConfig) -> set[tuple[str, str]]:
    signatures = {item.case_id: build_blocking_signature(item) for item in records}
    pairs: set[tuple[str, str]] = set()
    for index_a, record_a in enumerate(records):
        for record_b in records[index_a + 1 :]:
            sig_a = signatures[record_a.case_id]
            sig_b = signatures[record_b.case_id]
            diff = date_diff_days(sig_a.parsed_date, sig_b.parsed_date)
            distance = haversine_km(sig_a.latitude, sig_a.longitude, sig_b.latitude, sig_b.longitude)
            r1 = (
                diff is not None
                and diff <= config.date_threshold_name_rules
                and are_counters_compatible(sig_a.initials_bag, sig_b.initials_bag)
            )
            r2 = (
                diff is not None
                and diff <= config.date_threshold_name_rules
                and are_counters_compatible(sig_a.phonetic_onset_bag, sig_b.phonetic_onset_bag)
            )
            r4 = (
                distance is not None
                and diff is not None
                and distance <= config.geo_threshold_km
                and diff <= config.date_threshold_geo_rule
            )
            if r1 or r2 or r4:
                pairs.add(tuple(sorted([record_a.case_id, record_b.case_id])))
    return pairs
