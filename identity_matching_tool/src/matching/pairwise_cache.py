from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.cache import sqlite_cache
from src.models import (
    AnalysisConfig,
    CombinedSimilarityResult,
    GeoSimilarityResult,
    MissingPersonRecord,
    NameSimilarityResult,
    PhoneticSimilarityResult,
    TemporalSimilarityResult,
)

CACHE_THRESHOLD = 0.5
COMPONENT_CACHE_VERSION = "pairwise_components_v1"


def make_pair_key(id_a: str, id_b: str) -> str:
    """Build an order-independent cache key for two case IDs."""
    left, right = sorted([str(id_a), str(id_b)])
    return f"{left}__{right}"


def make_component_hash(
    record_a: MissingPersonRecord,
    record_b: MissingPersonRecord,
    analysis_config: AnalysisConfig,
) -> str:
    """Hash stable record fields and algorithm versions for component cache reuse."""
    ordered_records = sorted([record_a, record_b], key=lambda record: str(record.case_id))
    payload = {
        "version": COMPONENT_CACHE_VERSION,
        "phonetic_lang": str(analysis_config.filter_parameters.get("phonetic_lang", "spa-Latn")),
        "records": [
            _record_component_payload(record)
            for record in ordered_records
        ],
        "name_engine_version": "weighted_name_score_power2_v1",
        "phonetic_engine_version": "reordered_ipa_panphon_power2_5_v1",
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_component_cache_lookup(
    records_by_id: dict[str, MissingPersonRecord],
    analysis_config: AnalysisConfig,
) -> dict[str, dict[str, Any]]:
    """Load reusable component rows valid for the current records and enabled filters."""
    _ensure_component_table()
    with sqlite_cache.connect() as connection:
        rows = connection.execute("SELECT * FROM pairwise_components").fetchall()
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        data = dict(row)
        record_a = records_by_id.get(str(data.get("id_A")))
        record_b = records_by_id.get(str(data.get("id_B")))
        if record_a is None or record_b is None:
            continue
        expected_hash = make_component_hash(record_a, record_b, analysis_config)
        if data.get("component_hash") != expected_hash:
            continue
        if not _component_row_satisfies_config(data, analysis_config):
            continue
        lookup[str(data["cache_key"])] = data
    return lookup


def component_row_to_result(row: dict[str, Any], analysis_config: AnalysisConfig) -> CombinedSimilarityResult:
    """Rebuild a combined result from cached stable components and current thresholds."""
    enabled_filters = dict(analysis_config.enabled_filters)
    parameters = analysis_config.filter_parameters
    cache_context_needed = sum(1 for enabled in enabled_filters.values() if enabled) < 2

    name_result = _name_result_from_component_row(row) if enabled_filters.get("name") else None
    phonetic_result = _phonetic_result_from_component_row(row) if enabled_filters.get("phonetic") else None
    geo_result = (
        _geo_result_from_distance(row, parameters)
        if enabled_filters.get("geographic") or cache_context_needed
        else None
    )
    temporal_result = (
        _temporal_result_from_distance(row, parameters)
        if enabled_filters.get("temporal") or cache_context_needed
        else None
    )
    weighted_scores = _weighted_scores_from_components(
        parameters,
        enabled_filters,
        name_result,
        phonetic_result,
        geo_result,
        temporal_result,
    )
    if weighted_scores:
        score_final = sum(score * weight for score, weight, _ in weighted_scores) / sum(
            weight for _, weight, _ in weighted_scores
        )
        status = "ok"
    else:
        score_final = None
        status = "no_score"
    labels = ", ".join(label for _, _, label in weighted_scores) or "no enabled calculable dimensions"
    return CombinedSimilarityResult(
        score_final=_clamp_optional(score_final),
        name_result=name_result,
        phonetic_result=phonetic_result,
        geo_result=geo_result,
        temporal_result=temporal_result,
        enabled_filters=enabled_filters,
        explanation=f"Combined score rebuilt from pairwise component cache using: {labels}.",
        status=status,
    )


def get_cached_result(
    id_a: str,
    id_b: str,
    records_by_id: dict[str, MissingPersonRecord],
    analysis_config: AnalysisConfig,
) -> dict[str, Any] | None:
    """Fetch one reusable component row for a pair and the current record content."""
    pair_key = make_pair_key(id_a, id_b)
    lookup = build_component_cache_lookup(records_by_id, analysis_config)
    return lookup.get(pair_key)


def load_cache() -> pd.DataFrame:
    """Return the full pairwise component cache as a DataFrame for diagnostics and tests."""
    if not sqlite_cache.cache_db_path().exists():
        return pd.DataFrame(columns=_component_columns())
    _ensure_component_table()
    with sqlite_cache.connect() as connection:
        return pd.read_sql_query("SELECT * FROM pairwise_components", connection)


def save_cache(cache: pd.DataFrame) -> None:
    """Replace the pairwise component cache content from a DataFrame."""
    _ensure_component_table()
    clear_cache()
    rows = [row.dropna().to_dict() for _, row in cache.iterrows()]
    insert_component_rows(rows)


def clear_cache() -> None:
    """Delete pairwise component rows and remove the legacy final-score table."""
    _ensure_component_table()
    with sqlite_cache.connect() as connection:
        connection.execute("DELETE FROM pairwise_components")
        connection.execute("DROP TABLE IF EXISTS pairwise_scores")
        connection.commit()


def strong_criteria_count(result: CombinedSimilarityResult) -> int:
    """Count score dimensions above the diagnostic pairwise cache threshold."""
    scores = [
        result.name_result.score_final if result.name_result and result.name_result.status == "ok" else None,
        result.phonetic_result.score_final if result.phonetic_result and result.phonetic_result.status == "ok" else None,
        result.geo_result.score if result.geo_result and result.geo_result.status == "ok" else None,
        result.temporal_result.score if result.temporal_result and result.temporal_result.status == "ok" else None,
    ]
    return sum(1 for score in scores if score is not None and float(score) > CACHE_THRESHOLD)


def should_store_component_result(result: CombinedSimilarityResult) -> bool:
    """Return True when a pair result satisfies the component cache policy."""
    return strong_criteria_count(result) >= 2


def should_store_component_row(row: dict[str, Any]) -> bool:
    """Return True when a prepared component row satisfies the cache policy."""
    count = _optional_int(row.get("strong_criteria_count"))
    return count is not None and count >= 2


def _result_to_component_row(
    id_a: str,
    id_b: str,
    analysis_config: AnalysisConfig,
    result: CombinedSimilarityResult,
) -> dict[str, Any]:
    """Flatten stable reusable pair components into a SQLite row dictionary."""
    name_result = result.name_result
    phonetic_result = result.phonetic_result
    geo_result = result.geo_result
    temporal_result = result.temporal_result
    return {
        "id_A": id_a,
        "id_B": id_b,
        "cache_key": make_pair_key(id_a, id_b),
        "component_hash": _component_hash_from_result_ids(id_a, id_b, analysis_config, result),
        "score_name": name_result.score_final if name_result else None,
        "score_phonetic": phonetic_result.score_final if phonetic_result else None,
        "phonetic_status": phonetic_result.status if phonetic_result else None,
        "name_direct_structured_score": name_result.direct_structured_score if name_result else None,
        "name_direct_full_name_score": name_result.direct_full_name_score if name_result else None,
        "name_reordered_full_name_score": name_result.reordered_full_name_score if name_result else None,
        "name_token_alignment_score": name_result.token_alignment_score if name_result else None,
        "name_order_mismatch_suspected": name_result.order_mismatch_suspected if name_result else None,
        "name_best_method": name_result.best_method if name_result else None,
        "distance_km": geo_result.distance_km if geo_result else None,
        "distance_days": temporal_result.distance_days if temporal_result else None,
        "cache_policy_version": "two_scores_over_0_5_components_v2",
        "strong_criteria_count": strong_criteria_count(result),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def _component_hash_from_result_ids(
    id_a: str,
    id_b: str,
    analysis_config: AnalysisConfig,
    result: CombinedSimilarityResult,
) -> str:
    """Build a component hash fallback when full records are unavailable at store time."""
    payload = {
        "version": COMPONENT_CACHE_VERSION,
        "ids": sorted([str(id_a), str(id_b)]),
        "phonetic_lang": str(analysis_config.filter_parameters.get("phonetic_lang", "spa-Latn")),
        "component_presence": {
            "name": result.name_result is not None,
            "phonetic": result.phonetic_result is not None,
            "geographic_distance": result.geo_result.distance_km is not None if result.geo_result else False,
            "temporal_distance": result.temporal_result.distance_days is not None if result.temporal_result else False,
        },
        "name_engine_version": "weighted_name_score_power2_v1",
        "phonetic_engine_version": "reordered_ipa_panphon_power2_5_v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def component_row_for_records(
    id_a: str,
    id_b: str,
    record_a: MissingPersonRecord,
    record_b: MissingPersonRecord,
    analysis_config: AnalysisConfig,
    result: CombinedSimilarityResult,
) -> dict[str, Any]:
    """Build a component row with a record-content hash for engine-side storage."""
    id_record_pairs = sorted([(str(id_a), record_a), (str(id_b), record_b)], key=lambda item: item[0])
    left, left_record = id_record_pairs[0]
    right, right_record = id_record_pairs[1]
    row = _result_to_component_row(left, right, analysis_config, result)
    row["component_hash"] = make_component_hash(left_record, right_record, analysis_config)
    return row


def _component_columns() -> list[str]:
    """Return pairwise component cache column names in storage order."""
    return [
        "id_A",
        "id_B",
        "cache_key",
        "component_hash",
        "score_name",
        "score_phonetic",
        "phonetic_status",
        "name_direct_structured_score",
        "name_direct_full_name_score",
        "name_reordered_full_name_score",
        "name_token_alignment_score",
        "name_order_mismatch_suspected",
        "name_best_method",
        "distance_km",
        "distance_days",
        "cache_policy_version",
        "strong_criteria_count",
        "computed_at",
    ]


def insert_component_rows(rows: list[dict[str, Any]]) -> None:
    """Insert or replace prepared component rows that satisfy the selective cache policy."""
    _insert_component_rows([row for row in rows if should_store_component_row(row)])


def _insert_component_rows(rows: list[dict[str, Any]]) -> None:
    """Insert or replace pairwise component rows in SQLite."""
    if not rows:
        return
    _ensure_component_table()
    columns = _component_columns()
    placeholders = ", ".join("?" for _ in columns)
    update_columns = [column for column in columns if column != "cache_key"]
    updates = ", ".join(f"{column}=excluded.{column}" for column in update_columns)
    sql = (
        f"INSERT INTO pairwise_components ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT(cache_key) DO UPDATE SET {updates}"
    )
    values = [tuple(_sqlite_value(row.get(column)) for column in columns) for row in rows]
    with sqlite_cache.connect() as connection:
        connection.executemany(sql, values)
        connection.commit()


def _ensure_component_table() -> None:
    """Create or migrate the pairwise component cache table."""
    columns_sql = """
        id_A TEXT,
        id_B TEXT,
        cache_key TEXT PRIMARY KEY,
        component_hash TEXT NOT NULL,
        score_name REAL,
        score_phonetic REAL,
        phonetic_status TEXT,
        name_direct_structured_score REAL,
        name_direct_full_name_score REAL,
        name_reordered_full_name_score REAL,
        name_token_alignment_score REAL,
        name_order_mismatch_suspected TEXT,
        name_best_method TEXT,
        distance_km REAL,
        distance_days INTEGER,
        cache_policy_version TEXT,
        strong_criteria_count INTEGER,
        computed_at TEXT
    """
    with sqlite_cache.connect() as connection:
        connection.execute(f"CREATE TABLE IF NOT EXISTS pairwise_components ({columns_sql})")
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(pairwise_components)").fetchall()
        }
        for column in _component_columns():
            if column not in existing:
                connection.execute(f"ALTER TABLE pairwise_components ADD COLUMN {column} TEXT")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_pairwise_components_hash ON pairwise_components(component_hash)"
        )
        connection.commit()


def _record_component_payload(record: MissingPersonRecord) -> dict[str, Any]:
    """Return stable record fields that affect pairwise component calculations."""
    return {
        "case_id": record.case_id,
        "full_name_normalized": record.full_name_normalized,
        "first_name_normalized": record.first_name_normalized,
        "last_name_normalized": record.last_name_normalized,
        "name_tokens": record.name_tokens,
        "latitude": record.latitude,
        "longitude": record.longitude,
        "disappearance_date_parsed": record.disappearance_date_parsed,
    }


def _component_row_satisfies_config(row: dict[str, Any], analysis_config: AnalysisConfig) -> bool:
    """Return True when cached components cover all currently enabled expensive filters."""
    enabled = analysis_config.enabled_filters
    if enabled.get("name") and _optional_float(row.get("score_name")) is None:
        return False
    if enabled.get("phonetic"):
        if _optional_str(row.get("phonetic_status")) is None:
            return False
        if _optional_str(row.get("phonetic_status")) == "ok" and _optional_float(row.get("score_phonetic")) is None:
            return False
    if enabled.get("geographic") and _optional_float(row.get("distance_km")) is None:
        return False
    if enabled.get("temporal") and _optional_int(row.get("distance_days")) is None:
        return False
    return True


def _name_result_from_component_row(row: dict[str, Any]) -> NameSimilarityResult | None:
    """Rebuild a name result from cached component fields."""
    score_name = _optional_float(row.get("score_name"))
    if score_name is None:
        return None
    return NameSimilarityResult(
        score_final=score_name,
        direct_structured_score=_optional_float(row.get("name_direct_structured_score")),
        direct_full_name_score=_optional_float(row.get("name_direct_full_name_score")),
        reordered_full_name_score=_optional_float(row.get("name_reordered_full_name_score")),
        token_alignment_score=_optional_float(row.get("name_token_alignment_score")),
        order_mismatch_suspected=_optional_bool(row.get("name_order_mismatch_suspected")) or False,
        best_method=_optional_str(row.get("name_best_method")),
        matched_tokens=[],
        explanation="Name similarity reused from pairwise component cache.",
        status="ok",
    )


def _phonetic_result_from_component_row(row: dict[str, Any]) -> PhoneticSimilarityResult | None:
    """Rebuild a phonetic result from cached component fields."""
    status = _optional_str(row.get("phonetic_status"))
    if status is None:
        return None
    return PhoneticSimilarityResult(
        score_final=_optional_float(row.get("score_phonetic")),
        explanation="Phonetic similarity reused from pairwise component cache.",
        status=status,
    )


def _geo_result_from_distance(row: dict[str, Any], parameters: dict[str, float | int]) -> GeoSimilarityResult:
    """Recompute geographic score from cached raw distance and current thresholds."""
    distance_km = _optional_float(row.get("distance_km"))
    if distance_km is None:
        return GeoSimilarityResult(
            score=None,
            distance_km=None,
            explanation="Geographic similarity cannot be computed: cached distance is missing.",
            status="missing_coordinates",
        )
    distance_min_km = float(parameters.get("geographic_distance_min_km", 5.0))
    distance_max_km = float(parameters.get("geographic_distance_max_km", 50.0))
    if distance_min_km < 0 or distance_max_km <= distance_min_km:
        return GeoSimilarityResult(
            score=None,
            distance_km=distance_km,
            explanation="Geographic similarity cannot be computed: invalid distance parameters.",
            status="invalid_parameters",
        )
    if distance_km <= distance_min_km:
        score = 1.0
    elif distance_km >= distance_max_km:
        score = 0.0
    else:
        score = 1 - (distance_km - distance_min_km) / (distance_max_km - distance_min_km)
    return GeoSimilarityResult(
        score=max(0.0, min(1.0, float(score))),
        distance_km=distance_km,
        explanation=f"GPS distance of {distance_km:.1f} km reused from component cache; score recomputed.",
        status="ok",
    )


def _temporal_result_from_distance(row: dict[str, Any], parameters: dict[str, float | int]) -> TemporalSimilarityResult:
    """Recompute temporal score from cached day gap and current threshold."""
    distance_days = _optional_int(row.get("distance_days"))
    if distance_days is None:
        return TemporalSimilarityResult(
            score=None,
            distance_days=None,
            explanation="Temporal similarity cannot be computed: cached date gap is missing.",
            status="missing_date",
        )
    max_days = int(parameters.get("temporal_max_days", 365))
    if max_days <= 0:
        return TemporalSimilarityResult(
            score=None,
            distance_days=distance_days,
            explanation="Temporal similarity cannot be computed: max_days must be strictly positive.",
            status="invalid_parameters",
        )
    score = 0.0 if distance_days >= max_days else 1 - distance_days / max_days
    return TemporalSimilarityResult(
        score=max(0.0, min(1.0, float(score))),
        distance_days=distance_days,
        explanation=f"Gap of {distance_days} day(s) reused from component cache; score recomputed.",
        status="ok",
    )


def _weighted_scores_from_components(
    parameters: dict[str, float | int],
    enabled_filters: dict[str, bool],
    name_result: NameSimilarityResult | None,
    phonetic_result: PhoneticSimilarityResult | None,
    geo_result: GeoSimilarityResult | None,
    temporal_result: TemporalSimilarityResult | None,
) -> list[tuple[float, float, str]]:
    """Collect enabled component scores with current weights."""
    weighted: list[tuple[float, float, str]] = []
    if enabled_filters.get("name") and name_result is not None and name_result.status == "ok":
        weighted.append((name_result.score_final, float(parameters.get("name_weight", 1.0)), "name"))
    if (
        enabled_filters.get("phonetic")
        and phonetic_result is not None
        and phonetic_result.status == "ok"
        and phonetic_result.score_final is not None
    ):
        weighted.append((phonetic_result.score_final, float(parameters.get("phonetic_weight", 1.0)), "phonetic"))
    if enabled_filters.get("geographic") and geo_result is not None and geo_result.status == "ok" and geo_result.score is not None:
        weighted.append((geo_result.score, float(parameters.get("geographic_weight", 1.0)), "location"))
    if enabled_filters.get("temporal") and temporal_result is not None and temporal_result.status == "ok" and temporal_result.score is not None:
        weighted.append((temporal_result.score, float(parameters.get("temporal_weight", 1.0)), "time"))
    return [(score, weight, label) for score, weight, label in weighted if weight > 0]


def _clamp_optional(value: float | None) -> float | None:
    """Clamp a score when present and preserve None when absent."""
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _sqlite_value(value: Any) -> Any:
    """Convert Python and pandas values to SQLite-safe scalar values."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return str(value)
    return value


def _optional_float(value: Any) -> float | None:
    """Parse an optional float from cached data."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    """Parse an optional integer from cached data."""
    if value is None or pd.isna(value):
        return None
    return int(float(value))


def _optional_bool(value: Any) -> bool | None:
    """Parse an optional boolean from cached data."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _optional_str(value: Any) -> str | None:
    """Parse an optional string from cached data."""
    if value is None or pd.isna(value):
        return None
    return str(value)
