from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.models import AnalysisConfig, CombinedSimilarityResult

CACHE_DIR = Path("cache")
CACHE_FILE = CACHE_DIR / "pairwise_scores.csv"


def make_pair_key(id_a: str, id_b: str) -> str:
    left, right = sorted([str(id_a), str(id_b)])
    return f"{left}__{right}"


def make_config_hash(analysis_config: AnalysisConfig) -> str:
    payload = {
        "enabled_filters": analysis_config.enabled_filters,
        "filter_parameters": analysis_config.filter_parameters,
    }
    if analysis_config.enabled_filters.get("name"):
        payload["name_engine_version"] = "weighted_name_score_power2_v1"
    if analysis_config.enabled_filters.get("phonetic"):
        payload["phonetic_engine_version"] = "reordered_ipa_panphon_power2_5_v1"
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_cached_result(id_a: str, id_b: str, analysis_config: AnalysisConfig) -> dict[str, Any] | None:
    cache = load_cache()
    if cache.empty:
        return None
    pair_key = make_pair_key(id_a, id_b)
    config_hash = make_config_hash(analysis_config)
    rows = cache[(cache["cache_key"] == pair_key) & (cache["config_hash"] == config_hash)]
    if rows.empty:
        return None
    return rows.iloc[-1].to_dict()


def build_cache_lookup(analysis_config: AnalysisConfig) -> dict[str, dict[str, Any]]:
    cache = load_cache()
    if cache.empty:
        return {}
    config_hash = make_config_hash(analysis_config)
    matching_cache = cache[cache["config_hash"] == config_hash]
    if matching_cache.empty:
        return {}
    deduped = matching_cache.drop_duplicates(subset=["cache_key", "config_hash"], keep="last")
    return {str(row["cache_key"]): row.to_dict() for _, row in deduped.iterrows()}


def cached_row_to_result(row: dict[str, Any]) -> CombinedSimilarityResult:
    return CombinedSimilarityResult(
        score_final=_optional_float(row.get("score_final")),
        name_result=None,
        phonetic_result=None,
        geo_result=None,
        temporal_result=None,
        enabled_filters={},
        explanation=str(row.get("explanation") or "Resultat recupere depuis le cache."),
        status="ok",
    )


def store_pair_result(
    id_a: str,
    id_b: str,
    analysis_config: AnalysisConfig,
    result: CombinedSimilarityResult,
) -> None:
    cache = load_cache()
    left, right = sorted([str(id_a), str(id_b)])
    row = _result_to_cache_row(left, right, analysis_config, result)
    if cache.empty:
        updated = pd.DataFrame([row], columns=_cache_columns())
    else:
        updated = pd.concat([cache, pd.DataFrame([row])], ignore_index=True)
    updated = updated.drop_duplicates(subset=["cache_key", "config_hash"], keep="last")
    save_cache(updated)


def store_pair_results_batch(
    rows: list[tuple[str, str, CombinedSimilarityResult]],
    analysis_config: AnalysisConfig,
) -> None:
    if not rows:
        return
    cache = load_cache()
    new_rows = [
        _result_to_cache_row(*sorted([str(id_a), str(id_b)]), analysis_config, result)
        for id_a, id_b, result in rows
    ]
    if cache.empty:
        updated = pd.DataFrame(new_rows, columns=_cache_columns())
    else:
        updated = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
    updated = updated.drop_duplicates(subset=["cache_key", "config_hash"], keep="last")
    save_cache(updated)


def load_cache() -> pd.DataFrame:
    if not CACHE_FILE.exists():
        return pd.DataFrame(columns=_cache_columns())
    try:
        return pd.read_csv(CACHE_FILE, dtype=str, low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return pd.DataFrame(columns=_cache_columns())


def save_cache(cache: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.to_csv(CACHE_FILE, index=False)


def clear_cache() -> None:
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()


def _result_to_cache_row(
    id_a: str,
    id_b: str,
    analysis_config: AnalysisConfig,
    result: CombinedSimilarityResult,
) -> dict[str, Any]:
    name_result = result.name_result
    phonetic_result = result.phonetic_result
    geo_result = result.geo_result
    temporal_result = result.temporal_result
    return {
        "id_A": id_a,
        "id_B": id_b,
        "cache_key": make_pair_key(id_a, id_b),
        "config_hash": make_config_hash(analysis_config),
        "score_final": result.score_final,
        "score_name": name_result.score_final if name_result else None,
        "score_phonetic": phonetic_result.score_final if phonetic_result else None,
        "phonetic_status": phonetic_result.status if phonetic_result else None,
        "score_geographic": geo_result.score if geo_result else None,
        "score_temporal": temporal_result.score if temporal_result else None,
        "name_direct_structured_score": name_result.direct_structured_score if name_result else None,
        "name_direct_full_name_score": name_result.direct_full_name_score if name_result else None,
        "name_reordered_full_name_score": name_result.reordered_full_name_score if name_result else None,
        "name_token_alignment_score": name_result.token_alignment_score if name_result else None,
        "name_order_mismatch_suspected": name_result.order_mismatch_suspected if name_result else None,
        "name_best_method": name_result.best_method if name_result else None,
        "distance_km": geo_result.distance_km if geo_result else None,
        "distance_days": temporal_result.distance_days if temporal_result else None,
        "explanation": result.explanation,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def _cache_columns() -> list[str]:
    return [
        "id_A",
        "id_B",
        "cache_key",
        "config_hash",
        "score_final",
        "score_name",
        "score_phonetic",
        "phonetic_status",
        "score_geographic",
        "score_temporal",
        "name_direct_structured_score",
        "name_direct_full_name_score",
        "name_reordered_full_name_score",
        "name_token_alignment_score",
        "name_order_mismatch_suspected",
        "name_best_method",
        "distance_km",
        "distance_days",
        "explanation",
        "computed_at",
    ]


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
