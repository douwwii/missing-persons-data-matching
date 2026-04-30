from __future__ import annotations

from collections import Counter
import csv
from datetime import date, datetime
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

from src.models import BlockingConfig, BlockingIndex, BlockingResult, BlockingSignature, CandidatePair, MissingPersonRecord
from src.preprocessing.safe_parsing import none_if_missing, parse_name_tokens, parse_optional_date, parse_optional_float
from src.similarity.geo_similarity import haversine_distance_km
from src.matching.pair_generation import generate_single_case_pairs, theoretical_pair_count

MAX_SUBSIGNATURE_SIZE = 6
MAX_SUBSIGNATURE_KEYS = 256
GEO_NEIGHBOR_RADIUS = 2
BLOCKING_INDEX_CACHE_DIR = Path("cache") / "blocking_indexes"

_BLOCKING_INDEX_CACHE: dict[str, BlockingIndex] = {}


def build_blocking_signature(record: MissingPersonRecord) -> BlockingSignature:
    tokens = _blocking_tokens(record)
    initials = Counter(token[0].upper() for token in tokens if token)
    onsets = Counter(phonetic_onset(token) for token in tokens if phonetic_onset(token))
    parsed_date = _as_date(record.disappearance_date_parsed)
    latitude = parse_optional_float(record.latitude)
    longitude = parse_optional_float(record.longitude)
    return BlockingSignature(
        case_id=str(record.case_id),
        tokens=tokens,
        initials_bag=dict(initials),
        phonetic_onset_bag=dict(onsets),
        parsed_date=parsed_date,
        latitude=latitude,
        longitude=longitude,
        city=_optional_str(record.disappearance_city_verified),
        department=_optional_str(record.disappearance_department_verified),
        full_name_normalized=str(record.full_name_normalized or ""),
    )


def _blocking_tokens(record: MissingPersonRecord) -> list[str]:
    full_name_tokens = [
        token.strip()
        for token in str(record.full_name_normalized or "").split()
        if token.strip()
    ]
    if full_name_tokens:
        return full_name_tokens
    return parse_name_tokens(record.name_tokens)


def generate_standard_blocking_pairs(
    records: list[MissingPersonRecord],
    blocking_config: BlockingConfig | None = None,
) -> BlockingResult:
    started_at = perf_counter()
    blocking_config = blocking_config or BlockingConfig()
    blocking_index = get_or_build_blocking_index(records, blocking_config)
    signatures = blocking_index.signatures
    if blocking_config.selected_case_id:
        pairs = [
            (record_a.case_id, record_b.case_id)
            for record_a, record_b in generate_single_case_pairs(records, blocking_config.selected_case_id)
        ]
    else:
        pairs = sorted(_candidate_pair_ids_from_index(blocking_index, blocking_config))

    candidate_pairs: list[CandidatePair] = []
    r1_count = 0
    r2_count = 0
    r4_count = 0

    for id_a, id_b in pairs:
        sig_a = signatures[id_a]
        sig_b = signatures[id_b]
        diff_days = date_diff_days(sig_a.parsed_date, sig_b.parsed_date)
        distance_km = haversine_km(sig_a.latitude, sig_a.longitude, sig_b.latitude, sig_b.longitude)

        matched_rules: list[str] = []
        if _rule_r1(sig_a, sig_b, diff_days, blocking_config):
            matched_rules.append("R1")
            r1_count += 1
        if _rule_r2(sig_a, sig_b, diff_days, blocking_config):
            matched_rules.append("R2")
            r2_count += 1
        if _rule_r4(distance_km, diff_days, blocking_config):
            matched_rules.append("R4")
            r4_count += 1

        if not matched_rules:
            continue
        candidate_pairs.append(_candidate_pair(sig_a, sig_b, diff_days, distance_km, matched_rules))

    theoretical = theoretical_pair_count(len(records), blocking_config.selected_case_id)
    count = len(candidate_pairs)
    reduction = 0.0 if theoretical == 0 else 1 - count / theoretical
    elapsed = perf_counter() - started_at
    return BlockingResult(
        candidate_pairs=candidate_pairs,
        total_records=len(records),
        theoretical_pairs_count=theoretical,
        candidate_pairs_count=count,
        reduction_percentage=reduction,
        r1_count=r1_count,
        r2_count=r2_count,
        r4_count=r4_count,
        elapsed_time=elapsed,
        blocking_index_cache_dir=blocking_index.export_dir,
        explanation=(
            "Blocking standard indexe applique : R1 initiales/date, R2 debuts phonetiques simples/date, "
            "R4 proximite spatio-temporelle. Les buckets proposent les paires, puis les regles sont reverifiees."
        ),
    )


def get_or_build_blocking_index(
    records: list[MissingPersonRecord],
    blocking_config: BlockingConfig,
) -> BlockingIndex:
    cache_key = make_blocking_index_cache_key(records, blocking_config)
    cached = _BLOCKING_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    index = build_standard_blocking_index(records, blocking_config, cache_key=cache_key)
    export_blocking_index_to_disk(index, blocking_config)
    _BLOCKING_INDEX_CACHE.clear()
    _BLOCKING_INDEX_CACHE[cache_key] = index
    return index


def build_standard_blocking_index(
    records: list[MissingPersonRecord],
    blocking_config: BlockingConfig,
    cache_key: str | None = None,
) -> BlockingIndex:
    signatures = {record.case_id: build_blocking_signature(record) for record in records}
    index = BlockingIndex(cache_key=cache_key or make_blocking_index_cache_key(records, blocking_config), signatures=signatures)

    for signature in signatures.values():
        for date_bucket in _neighbor_date_buckets(signature.parsed_date, blocking_config.date_threshold_name_rules):
            if date_bucket is None:
                continue
            for key in _subsignature_keys(signature.initials_bag):
                index.initials_index.setdefault((key, date_bucket), []).append(signature.case_id)
            for key in _subsignature_keys(signature.phonetic_onset_bag):
                index.phonetic_onset_index.setdefault((key, date_bucket), []).append(signature.case_id)

        geo_cell = _geo_cell(signature.latitude, signature.longitude, blocking_config.geo_threshold_km)
        date_bucket = _date_bucket(signature.parsed_date, blocking_config.date_threshold_geo_rule)
        if geo_cell is not None and date_bucket is not None:
            cell_lat, cell_lon = geo_cell
            index.geo_time_index.setdefault((cell_lat, cell_lon, date_bucket), []).append(signature.case_id)

    return index


def make_blocking_index_cache_key(records: list[MissingPersonRecord], blocking_config: BlockingConfig) -> str:
    payload = {
        "records": [
            (
                record.case_id,
                record.name_tokens,
                str(record.disappearance_date_parsed),
                record.latitude,
                record.longitude,
            )
            for record in records
        ],
        "date_threshold_name_rules": blocking_config.date_threshold_name_rules,
        "geo_threshold_km": blocking_config.geo_threshold_km,
        "date_threshold_geo_rule": blocking_config.date_threshold_geo_rule,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def clear_blocking_index_cache() -> None:
    _BLOCKING_INDEX_CACHE.clear()


def export_blocking_index_to_disk(
    blocking_index: BlockingIndex,
    blocking_config: BlockingConfig,
    base_dir: Path | None = None,
) -> Path:
    export_dir = (base_dir or BLOCKING_INDEX_CACHE_DIR) / blocking_index.cache_key[:16]
    export_dir.mkdir(parents=True, exist_ok=True)

    blocking_index.export_dir = str(export_dir)
    try:
        _write_metadata(export_dir, blocking_index, blocking_config)
        _write_signatures(export_dir / "signatures.csv", blocking_index)
        _write_bucket_index(export_dir / "r1_initials_buckets.csv", "R1", blocking_index.initials_index)
        _write_bucket_index(export_dir / "r2_phonetic_onset_buckets.csv", "R2", blocking_index.phonetic_onset_index)
        _write_bucket_index(export_dir / "r4_geo_time_buckets.csv", "R4", blocking_index.geo_time_index)
        _write_bucket_summary(export_dir / "bucket_summary.csv", blocking_index)
    except OSError:
        # Diagnostic export must never block matching if another process keeps a CSV open.
        return export_dir
    return export_dir


def are_counters_compatible(counter_a: Counter | dict[str, int], counter_b: Counter | dict[str, int]) -> bool:
    a = Counter(counter_a)
    b = Counter(counter_b)
    shorter, longer = (a, b) if sum(a.values()) <= sum(b.values()) else (b, a)
    if not shorter:
        return False
    return all(longer[key] >= value for key, value in shorter.items())


def date_diff_days(date_a: object, date_b: object) -> int | None:
    parsed_a = _as_date(date_a)
    parsed_b = _as_date(date_b)
    if parsed_a is None or parsed_b is None:
        return None
    return abs((parsed_a - parsed_b).days)


def haversine_km(lat1: object, lon1: object, lat2: object, lon2: object) -> float | None:
    lat1_float = parse_optional_float(lat1)
    lon1_float = parse_optional_float(lon1)
    lat2_float = parse_optional_float(lat2)
    lon2_float = parse_optional_float(lon2)
    if not _valid_coordinates(lat1_float, lon1_float) or not _valid_coordinates(lat2_float, lon2_float):
        return None
    return haversine_distance_km(lat1_float, lon1_float, lat2_float, lon2_float)


def phonetic_onset(token: str) -> str:
    if not token:
        return ""
    text = str(token).strip().upper()
    if not text:
        return ""
    first = text[0]
    second = text[1] if len(text) > 1 else ""
    if first in {"C", "K", "Q"}:
        return "K"
    if first in {"B", "V"}:
        return "B"
    if first in {"S", "Z"}:
        return "S"
    if first in {"I", "Y"}:
        return "I"
    if first == "J":
        return "J"
    if first == "G" and second in {"E", "I"}:
        return "J"
    return first


def _write_metadata(export_dir: Path, blocking_index: BlockingIndex, blocking_config: BlockingConfig) -> None:
    metadata = {
        "cache_key": blocking_index.cache_key,
        "total_signatures": len(blocking_index.signatures),
        "r1_bucket_count": len(blocking_index.initials_index),
        "r2_bucket_count": len(blocking_index.phonetic_onset_index),
        "r4_bucket_count": len(blocking_index.geo_time_index),
        "date_threshold_name_rules": blocking_config.date_threshold_name_rules,
        "geo_threshold_km": blocking_config.geo_threshold_km,
        "date_threshold_geo_rule": blocking_config.date_threshold_geo_rule,
    }
    (export_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_signatures(path: Path, blocking_index: BlockingIndex) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "full_name_normalized",
                "tokens",
                "initials_bag",
                "phonetic_onset_bag",
                "date",
                "latitude",
                "longitude",
                "city",
                "department",
            ],
        )
        writer.writeheader()
        for signature in blocking_index.signatures.values():
            writer.writerow(
                {
                    "case_id": signature.case_id,
                    "full_name_normalized": signature.full_name_normalized,
                    "tokens": json.dumps(signature.tokens, ensure_ascii=False),
                    "initials_bag": json.dumps(signature.initials_bag, ensure_ascii=False, sort_keys=True),
                    "phonetic_onset_bag": json.dumps(signature.phonetic_onset_bag, ensure_ascii=False, sort_keys=True),
                    "date": signature.parsed_date.isoformat() if signature.parsed_date else "",
                    "latitude": signature.latitude,
                    "longitude": signature.longitude,
                    "city": signature.city,
                    "department": signature.department,
                }
            )


def _write_bucket_index(
    path: Path,
    rule: str,
    bucket_index: dict[object, list[str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rule", "bucket_key", "bucket_size", "case_ids"],
        )
        writer.writeheader()
        for bucket_key, case_ids in sorted(bucket_index.items(), key=lambda item: len(item[1]), reverse=True):
            writer.writerow(
                {
                    "rule": rule,
                    "bucket_key": json.dumps(_jsonable_bucket_key(bucket_key), ensure_ascii=False),
                    "bucket_size": len(case_ids),
                    "case_ids": "|".join(case_ids),
                }
            )


def _write_bucket_summary(path: Path, blocking_index: BlockingIndex) -> None:
    rows = []
    for rule, bucket_index in [
        ("R1", blocking_index.initials_index),
        ("R2", blocking_index.phonetic_onset_index),
        ("R4", blocking_index.geo_time_index),
    ]:
        sizes = [len(case_ids) for case_ids in bucket_index.values()]
        rows.append(
            {
                "rule": rule,
                "bucket_count": len(sizes),
                "max_bucket_size": max(sizes) if sizes else 0,
                "average_bucket_size": round(sum(sizes) / len(sizes), 3) if sizes else 0,
                "buckets_over_100": sum(1 for size in sizes if size > 100),
                "buckets_over_1000": sum(1 for size in sizes if size > 1000),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rule",
                "bucket_count",
                "max_bucket_size",
                "average_bucket_size",
                "buckets_over_100",
                "buckets_over_1000",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _jsonable_bucket_key(bucket_key: object) -> object:
    if isinstance(bucket_key, tuple):
        return [_jsonable_bucket_key(value) for value in bucket_key]
    return bucket_key


def _candidate_pair_ids_from_index(
    blocking_index: BlockingIndex,
    blocking_config: BlockingConfig,
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for signature in blocking_index.signatures.values():
        query_keys = _query_subsignature_keys(signature.initials_bag)
        for date_bucket in _neighbor_date_buckets(signature.parsed_date, blocking_config.date_threshold_name_rules):
            if date_bucket is None:
                continue
            for key in query_keys:
                _add_pairs_for_ids(signature.case_id, blocking_index.initials_index.get((key, date_bucket), []), pairs)

        query_keys = _query_subsignature_keys(signature.phonetic_onset_bag)
        for date_bucket in _neighbor_date_buckets(signature.parsed_date, blocking_config.date_threshold_name_rules):
            if date_bucket is None:
                continue
            for key in query_keys:
                _add_pairs_for_ids(
                    signature.case_id,
                    blocking_index.phonetic_onset_index.get((key, date_bucket), []),
                    pairs,
                )

        geo_cell = _geo_cell(signature.latitude, signature.longitude, blocking_config.geo_threshold_km)
        date_bucket = _date_bucket(signature.parsed_date, blocking_config.date_threshold_geo_rule)
        if geo_cell is not None and date_bucket is not None:
            cell_lat, cell_lon = geo_cell
            for neighbor_lat in range(cell_lat - GEO_NEIGHBOR_RADIUS, cell_lat + GEO_NEIGHBOR_RADIUS + 1):
                for neighbor_lon in range(cell_lon - GEO_NEIGHBOR_RADIUS, cell_lon + GEO_NEIGHBOR_RADIUS + 1):
                    for neighbor_date in (date_bucket - 1, date_bucket, date_bucket + 1):
                        _add_pairs_for_ids(
                            signature.case_id,
                            blocking_index.geo_time_index.get((neighbor_lat, neighbor_lon, neighbor_date), []),
                            pairs,
                        )
    return pairs


def _add_pairs_for_ids(anchor_id: str, candidate_ids: list[str], pairs: set[tuple[str, str]]) -> None:
    for candidate_id in candidate_ids:
        if candidate_id == anchor_id:
            continue
        pairs.add(tuple(sorted((anchor_id, candidate_id))))


def _subsignature_keys(counter: Counter | dict[str, int]) -> list[tuple[str, ...]]:
    elements = tuple(sorted(Counter(counter).elements()))
    if not elements:
        return []
    keys: set[tuple[str, ...]] = {elements}
    max_size = min(len(elements), MAX_SUBSIGNATURE_SIZE)
    for size in range(1, max_size + 1):
        for combo in combinations(elements, size):
            keys.add(combo)
            if len(keys) >= MAX_SUBSIGNATURE_KEYS:
                return sorted(keys)
    return sorted(keys)


def _query_subsignature_keys(counter: Counter | dict[str, int]) -> list[tuple[str, ...]]:
    return _subsignature_keys(counter)


def _date_bucket(value: date | None, threshold_days: int) -> int | None:
    if value is None:
        return None
    bucket_size = max(int(threshold_days), 1)
    return value.toordinal() // bucket_size


def _neighbor_date_buckets(value: date | None, threshold_days: int) -> tuple[int | None, ...]:
    bucket = _date_bucket(value, threshold_days)
    if bucket is None:
        return (None,)
    return (bucket - 1, bucket, bucket + 1)


def _geo_cell(latitude: float | None, longitude: float | None, threshold_km: float) -> tuple[int, int] | None:
    if not _valid_coordinates(latitude, longitude):
        return None
    cell_size_deg = max(float(threshold_km) / 111.0, 0.05)
    return (math.floor(latitude / cell_size_deg), math.floor(longitude / cell_size_deg))


def _rule_r1(
    sig_a: BlockingSignature,
    sig_b: BlockingSignature,
    diff_days: int | None,
    config: BlockingConfig,
) -> bool:
    return (
        diff_days is not None
        and diff_days <= config.date_threshold_name_rules
        and are_counters_compatible(sig_a.initials_bag, sig_b.initials_bag)
    )


def _rule_r2(
    sig_a: BlockingSignature,
    sig_b: BlockingSignature,
    diff_days: int | None,
    config: BlockingConfig,
) -> bool:
    return (
        diff_days is not None
        and diff_days <= config.date_threshold_name_rules
        and are_counters_compatible(sig_a.phonetic_onset_bag, sig_b.phonetic_onset_bag)
    )


def _rule_r4(distance_km: float | None, diff_days: int | None, config: BlockingConfig) -> bool:
    return (
        distance_km is not None
        and diff_days is not None
        and distance_km <= config.geo_threshold_km
        and diff_days <= config.date_threshold_geo_rule
    )


def _candidate_pair(
    sig_a: BlockingSignature,
    sig_b: BlockingSignature,
    diff_days: int | None,
    distance_km: float | None,
    matched_rules: list[str],
) -> CandidatePair:
    return CandidatePair(
        id_A=sig_a.case_id,
        id_B=sig_b.case_id,
        name_A=sig_a.full_name_normalized,
        name_B=sig_b.full_name_normalized,
        date_A=sig_a.parsed_date,
        date_B=sig_b.parsed_date,
        date_diff_days=diff_days,
        city_A=sig_a.city,
        city_B=sig_b.city,
        department_A=sig_a.department,
        department_B=sig_b.department,
        geo_distance_km=distance_km,
        matched_rules=matched_rules,
    )


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = parse_optional_date(value)
    if isinstance(parsed, datetime):
        return parsed.date()
    return parsed


def _optional_str(value: object) -> str | None:
    value = none_if_missing(value)
    if value is None:
        return None
    return str(value)


def _valid_coordinates(latitude: float | None, longitude: float | None) -> bool:
    if latitude is None or longitude is None:
        return False
    return -90 <= latitude <= 90 and -180 <= longitude <= 180
