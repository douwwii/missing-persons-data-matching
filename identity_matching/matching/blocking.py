from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import date, datetime
from itertools import combinations
from pathlib import Path
from time import perf_counter

from identity_matching.cache import sqlite_cache
from identity_matching.matching.pair_generation import generate_single_case_pairs, theoretical_pair_count
from identity_matching.models import (
    BlockingConfig,
    BlockingIndex,
    BlockingResult,
    BlockingSignature,
    CandidatePair,
    MissingPersonRecord,
)
from identity_matching.preprocessing.safe_parsing import (
    none_if_missing,
    parse_name_tokens,
    parse_optional_date,
    parse_optional_float,
)
from identity_matching.similarity.geo_similarity import haversine_distance_km

MAX_SUBSIGNATURE_SIZE = 6
MAX_SUBSIGNATURE_KEYS = 256
GEO_NEIGHBOR_RADIUS = 2
BLOCKING_INDEX_CACHE_DIR = sqlite_cache.CACHE_DIR / "blocking_indexes"

_BLOCKING_INDEX_CACHE: dict[str, BlockingIndex] = {}


def build_blocking_signature(record: MissingPersonRecord) -> BlockingSignature:
    """Build the lightweight per-record data used by standard blocking rules."""
    tokens = _blocking_tokens(record)
    initials = Counter(token[0].upper() for token in tokens if token)
    onsets = Counter(onset for token in tokens if (onset := phonetic_onset(token)))
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
    """Return blocking tokens from full_name_normalized, falling back to name_tokens."""
    full_name_tokens = [
        token.strip() for token in str(record.full_name_normalized or "").split() if token.strip()
    ]
    if full_name_tokens:
        return full_name_tokens
    return parse_name_tokens(record.name_tokens)


def generate_standard_blocking_pairs(
    records: list[MissingPersonRecord],
    blocking_config: BlockingConfig | None = None,
) -> BlockingResult:
    """Generate candidate pairs kept by standard blocking rules R1, R2, or R3."""
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
    r3_count = 0

    for id_a, id_b in pairs:
        sig_a = signatures[id_a]
        sig_b = signatures[id_b]
        diff_days = _signature_date_diff_days(sig_a.parsed_date, sig_b.parsed_date)
        distance_km = _signature_haversine_km(
            sig_a.latitude, sig_a.longitude, sig_b.latitude, sig_b.longitude
        )

        matched_rules: list[str] = []
        if _rule_r1(sig_a, sig_b, diff_days, blocking_config):
            matched_rules.append("R1")
            r1_count += 1
        if _rule_r2(sig_a, sig_b, diff_days, blocking_config):
            matched_rules.append("R2")
            r2_count += 1
        if _rule_r3(distance_km, diff_days, blocking_config):
            matched_rules.append("R3")
            r3_count += 1

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
        r3_count=r3_count,
        elapsed_time=elapsed,
        blocking_index_cache_dir=blocking_index.export_dir,
        explanation=(
            "Indexed standard blocking applied: R1 initials/date, R2 simple phonetic onsets/date, "
            "R3 strong spatio-temporal proximity. Buckets propose pairs, then rules are checked exactly."
        ),
    )


def get_or_build_blocking_index(
    records: list[MissingPersonRecord],
    blocking_config: BlockingConfig,
) -> BlockingIndex:
    """Load a blocking bucket index from memory or SQLite, or build it if absent."""
    cache_key = make_blocking_index_cache_key(records, blocking_config)
    cached = _BLOCKING_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    signatures = {record.case_id: build_blocking_signature(record) for record in records}
    cached_index = _load_blocking_buckets_from_sqlite(cache_key, signatures)
    if cached_index is not None:
        export_blocking_index_to_disk(cached_index, blocking_config)
        _BLOCKING_INDEX_CACHE.clear()
        _BLOCKING_INDEX_CACHE[cache_key] = cached_index
        return cached_index
    index = build_standard_blocking_index(
        records, blocking_config, cache_key=cache_key, signatures=signatures
    )
    _store_blocking_buckets_in_sqlite(index)
    export_blocking_index_to_disk(index, blocking_config)
    _BLOCKING_INDEX_CACHE.clear()
    _BLOCKING_INDEX_CACHE[cache_key] = index
    return index


def build_standard_blocking_index(
    records: list[MissingPersonRecord],
    blocking_config: BlockingConfig,
    cache_key: str | None = None,
    signatures: dict[str, BlockingSignature] | None = None,
) -> BlockingIndex:
    """Build bucket indexes for initials, simple phonetic onsets, and geo-time cells."""
    signatures = signatures or {record.case_id: build_blocking_signature(record) for record in records}
    index = BlockingIndex(
        cache_key=cache_key or make_blocking_index_cache_key(records, blocking_config), signatures=signatures
    )

    for signature in signatures.values():
        for date_bucket in _neighbor_date_buckets(
            signature.parsed_date, blocking_config.date_threshold_name_rules
        ):
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
    """Create a stable hash for the records and blocking thresholds."""
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
    """Clear in-memory and SQLite blocking bucket caches."""
    _BLOCKING_INDEX_CACHE.clear()
    _ensure_blocking_bucket_table()
    with sqlite_cache.connect() as connection:
        connection.execute("DELETE FROM blocking_buckets")
        connection.commit()


def export_blocking_index_to_disk(
    blocking_index: BlockingIndex,
    blocking_config: BlockingConfig,
    base_dir: Path | None = None,
) -> Path:
    """Write human-readable diagnostic CSV files for the current blocking buckets."""
    export_dir = (base_dir or BLOCKING_INDEX_CACHE_DIR) / blocking_index.cache_key[:16]
    export_dir.mkdir(parents=True, exist_ok=True)

    blocking_index.export_dir = str(export_dir)
    try:
        _write_metadata(export_dir, blocking_index, blocking_config)
        _write_signatures(export_dir / "signatures.csv", blocking_index)
        _write_bucket_index(export_dir / "r1_initials_buckets.csv", "R1", blocking_index.initials_index)
        _write_bucket_index(
            export_dir / "r2_phonetic_onset_buckets.csv", "R2", blocking_index.phonetic_onset_index
        )
        _write_bucket_index(export_dir / "r3_geo_time_buckets.csv", "R3", blocking_index.geo_time_index)
        _write_bucket_summary(export_dir / "bucket_summary.csv", blocking_index)
    except OSError:
        # Diagnostic export must never block matching if another process keeps a CSV open.
        return export_dir
    return export_dir


def are_counters_compatible(counter_a: Counter | dict[str, int], counter_b: Counter | dict[str, int]) -> bool:
    """Return True when the smaller multiset is included in the larger multiset."""
    a = Counter(counter_a)
    b = Counter(counter_b)
    shorter, longer = (a, b) if sum(a.values()) <= sum(b.values()) else (b, a)
    if not shorter:
        return False
    return all(longer[key] >= value for key, value in shorter.items())


def date_diff_days(date_a: object, date_b: object) -> int | None:
    """Return the absolute day difference between two parseable dates."""
    parsed_a = _as_date(date_a)
    parsed_b = _as_date(date_b)
    if parsed_a is None or parsed_b is None:
        return None
    return abs((parsed_a - parsed_b).days)


def haversine_km(lat1: object, lon1: object, lat2: object, lon2: object) -> float | None:
    """Return the Haversine distance in kilometers when both coordinates are valid."""
    lat1_float = parse_optional_float(lat1)
    lon1_float = parse_optional_float(lon1)
    lat2_float = parse_optional_float(lat2)
    lon2_float = parse_optional_float(lon2)
    if not _valid_coordinates(lat1_float, lon1_float) or not _valid_coordinates(lat2_float, lon2_float):
        return None
    return haversine_distance_km(lat1_float, lon1_float, lat2_float, lon2_float)


def _signature_date_diff_days(date_a: date | None, date_b: date | None) -> int | None:
    """Return a day difference for dates already normalized in blocking signatures."""
    if date_a is None or date_b is None:
        return None
    return abs((date_a - date_b).days)


def _signature_haversine_km(
    lat1: float | None,
    lon1: float | None,
    lat2: float | None,
    lon2: float | None,
) -> float | None:
    """Return a GPS distance for coordinates already normalized in blocking signatures."""
    if not _valid_coordinates(lat1, lon1) or not _valid_coordinates(lat2, lon2):
        return None
    return haversine_distance_km(lat1, lon1, lat2, lon2)


def phonetic_onset(token: str) -> str:
    """Return the cheap normalized first-sound bucket used by R2 blocking."""
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
    """Write blocking index metadata as JSON for manual inspection."""
    metadata = {
        "cache_key": blocking_index.cache_key,
        "total_signatures": len(blocking_index.signatures),
        "r1_bucket_count": len(blocking_index.initials_index),
        "r2_bucket_count": len(blocking_index.phonetic_onset_index),
        "r3_bucket_count": len(blocking_index.geo_time_index),
        "date_threshold_name_rules": blocking_config.date_threshold_name_rules,
        "geo_threshold_km": blocking_config.geo_threshold_km,
        "date_threshold_geo_rule": blocking_config.date_threshold_geo_rule,
    }
    (export_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _write_signatures(path: Path, blocking_index: BlockingIndex) -> None:
    """Write one CSV row per blocking signature for manual inspection."""
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
                    "phonetic_onset_bag": json.dumps(
                        signature.phonetic_onset_bag, ensure_ascii=False, sort_keys=True
                    ),
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
    """Write one CSV row per bucket for a blocking rule."""
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
    """Write aggregate bucket size statistics for all standard blocking rules."""
    rows = []
    for rule, bucket_index in [
        ("R1", blocking_index.initials_index),
        ("R2", blocking_index.phonetic_onset_index),
        ("R3", blocking_index.geo_time_index),
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
    """Convert tuple bucket keys to JSON-serializable lists."""
    if isinstance(bucket_key, tuple):
        return [_jsonable_bucket_key(value) for value in bucket_key]
    return bucket_key


def _store_blocking_buckets_in_sqlite(blocking_index: BlockingIndex) -> None:
    """Persist only bucket definitions and case IDs for a blocking index in SQLite."""
    _ensure_blocking_bucket_table()
    rows = []
    for rule, bucket_index in [
        ("R1", blocking_index.initials_index),
        ("R2", blocking_index.phonetic_onset_index),
        ("R3", blocking_index.geo_time_index),
    ]:
        for bucket_key, case_ids in bucket_index.items():
            rows.append(
                (
                    blocking_index.cache_key,
                    rule,
                    json.dumps(_jsonable_bucket_key(bucket_key), ensure_ascii=False),
                    json.dumps(sorted(case_ids), ensure_ascii=False),
                    len(case_ids),
                )
            )
    if not rows:
        return
    with sqlite_cache.connect() as connection:
        connection.execute("DELETE FROM blocking_buckets WHERE cache_key = ?", (blocking_index.cache_key,))
        connection.executemany(
            """
            INSERT INTO blocking_buckets (cache_key, rule, bucket_key, case_ids_json, bucket_size)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def _load_blocking_buckets_from_sqlite(
    cache_key: str,
    signatures: dict[str, BlockingSignature],
) -> BlockingIndex | None:
    """Load cached blocking buckets from SQLite for a cache key."""
    _ensure_blocking_bucket_table()
    with sqlite_cache.connect() as connection:
        rows = connection.execute(
            "SELECT rule, bucket_key, case_ids_json FROM blocking_buckets WHERE cache_key = ?",
            (cache_key,),
        ).fetchall()
    if not rows:
        return None
    index = BlockingIndex(cache_key=cache_key, signatures=signatures)
    for row in rows:
        bucket_key = _tuple_bucket_key(json.loads(row["bucket_key"]))
        case_ids = [str(case_id) for case_id in json.loads(row["case_ids_json"])]
        if row["rule"] == "R1":
            index.initials_index[bucket_key] = case_ids
        elif row["rule"] == "R2":
            index.phonetic_onset_index[bucket_key] = case_ids
        elif row["rule"] in {"R3", "R4"}:
            index.geo_time_index[bucket_key] = case_ids
    return index


def _tuple_bucket_key(value: object) -> object:
    """Convert JSON-loaded list bucket keys back to tuples."""
    if isinstance(value, list):
        return tuple(_tuple_bucket_key(item) for item in value)
    return value


def _ensure_blocking_bucket_table() -> None:
    """Create the SQLite table used to persist blocking buckets if needed."""
    with sqlite_cache.connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS blocking_buckets (
                cache_key TEXT NOT NULL,
                rule TEXT NOT NULL,
                bucket_key TEXT NOT NULL,
                case_ids_json TEXT NOT NULL,
                bucket_size INTEGER NOT NULL,
                PRIMARY KEY (cache_key, rule, bucket_key)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_blocking_buckets_cache_key ON blocking_buckets(cache_key)"
        )
        connection.commit()


def _candidate_pair_ids_from_index(
    blocking_index: BlockingIndex,
    blocking_config: BlockingConfig,
) -> set[tuple[str, str]]:
    """Generate candidate ID pairs by crossing compatible cached buckets."""
    pairs: set[tuple[str, str]] = set()
    for signature in blocking_index.signatures.values():
        query_keys = _query_subsignature_keys(signature.initials_bag)
        for date_bucket in _neighbor_date_buckets(
            signature.parsed_date, blocking_config.date_threshold_name_rules
        ):
            if date_bucket is None:
                continue
            for key in query_keys:
                _add_pairs_for_ids(
                    signature.case_id, blocking_index.initials_index.get((key, date_bucket), []), pairs
                )

        query_keys = _query_subsignature_keys(signature.phonetic_onset_bag)
        for date_bucket in _neighbor_date_buckets(
            signature.parsed_date, blocking_config.date_threshold_name_rules
        ):
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
                            blocking_index.geo_time_index.get(
                                (neighbor_lat, neighbor_lon, neighbor_date), []
                            ),
                            pairs,
                        )
    return pairs


def _add_pairs_for_ids(anchor_id: str, candidate_ids: list[str], pairs: set[tuple[str, str]]) -> None:
    """Add normalized pairs between an anchor ID and bucket member IDs."""
    for candidate_id in candidate_ids:
        if candidate_id == anchor_id:
            continue
        pairs.add(tuple(sorted((anchor_id, candidate_id))))


def _subsignature_keys(counter: Counter | dict[str, int]) -> list[tuple[str, ...]]:
    """Generate bounded multiset keys so shorter names can query longer-name buckets."""
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
    """Return the bucket keys used when querying candidate buckets for one signature."""
    return _subsignature_keys(counter)


def _date_bucket(value: date | None, threshold_days: int) -> int | None:
    """Map a date to an integer bucket based on a day threshold."""
    if value is None:
        return None
    bucket_size = max(int(threshold_days), 1)
    return value.toordinal() // bucket_size


def _neighbor_date_buckets(value: date | None, threshold_days: int) -> tuple[int | None, ...]:
    """Return the date bucket and direct neighbors for threshold-boundary safety."""
    bucket = _date_bucket(value, threshold_days)
    if bucket is None:
        return (None,)
    return (bucket - 1, bucket, bucket + 1)


def _geo_cell(latitude: float | None, longitude: float | None, threshold_km: float) -> tuple[int, int] | None:
    """Map coordinates to a coarse geographic cell derived from the kilometer threshold."""
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
    """Evaluate R1: compatible initials and a close disappearance date."""
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
    """Evaluate R2: compatible cheap phonetic onsets and a close disappearance date."""
    return (
        diff_days is not None
        and diff_days <= config.date_threshold_name_rules
        and are_counters_compatible(sig_a.phonetic_onset_bag, sig_b.phonetic_onset_bag)
    )


def _rule_r3(distance_km: float | None, diff_days: int | None, config: BlockingConfig) -> bool:
    """Evaluate R3: strong geographic and temporal proximity."""
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
    """Build a candidate pair object from two blocking signatures and matched rules."""
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
    """Parse a date-like value into a date or return None."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = parse_optional_date(value)
    if isinstance(parsed, datetime):
        return parsed.date()
    return parsed


def _optional_str(value: object) -> str | None:
    """Return a string for present values and None for missing values."""
    value = none_if_missing(value)
    if value is None:
        return None
    return str(value)


def _valid_coordinates(latitude: float | None, longitude: float | None) -> bool:
    """Return True when latitude and longitude are both inside valid GPS ranges."""
    if latitude is None or longitude is None:
        return False
    return -90 <= latitude <= 90 and -180 <= longitude <= 180
