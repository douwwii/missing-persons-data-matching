from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime
from time import perf_counter
from typing import Any, Callable

from src.matching.pairwise_cache import (
    build_cache_lookup,
    cached_row_to_result,
    make_pair_key,
    store_pair_results_batch,
)
from src.models import AnalysisConfig, CombinedSimilarityResult, MissingPersonRecord, PairMatchingResult, ParallelConfig
from src.similarity.combined_similarity import compute_combined_similarity
from src.similarity.phonetic_similarity import precompute_ipa_for_records


def parallel_compute_pair_scores(
    pairs: list[tuple[MissingPersonRecord, MissingPersonRecord]] | list[tuple[str, str]],
    records_by_id: dict[str, MissingPersonRecord],
    analysis_config: AnalysisConfig,
    num_workers: int,
    chunk_size: int,
    use_cache: bool = True,
    cache_manager: object | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> list[PairMatchingResult]:
    del cache_manager
    started_at = perf_counter()
    normalized_pairs = _normalize_pairs(pairs)
    cached_results: list[PairMatchingResult] = []
    pairs_to_compute: list[tuple[str, str]] = []

    if use_cache:
        cache_lookup = build_cache_lookup(analysis_config)
        for id_a, id_b in normalized_pairs:
            cached = cache_lookup.get(make_pair_key(id_a, id_b))
            if cached is None:
                pairs_to_compute.append((id_a, id_b))
            else:
                cached_results.append(_cached_pair_result(id_a, id_b, records_by_id, cached))
            if progress_callback is not None:
                progress_callback("Verification du cache", len(cached_results) + len(pairs_to_compute), len(normalized_pairs))
    else:
        pairs_to_compute = normalized_pairs

    if analysis_config.enabled_filters.get("phonetic") and pairs_to_compute:
        involved_ids = {case_id for pair in pairs_to_compute for case_id in pair}
        involved_records = [
            record for case_id, record in records_by_id.items()
            if case_id in involved_ids
        ]
        precompute_ipa_for_records(
            involved_records,
            lang=str(analysis_config.filter_parameters.get("phonetic_lang", "spa-Latn")),
            progress_callback=progress_callback,
        )

    try:
        computed_results = _compute_pairs(
            pairs_to_compute,
            records_by_id,
            analysis_config,
            num_workers=num_workers,
            chunk_size=chunk_size,
            progress_callback=progress_callback,
        )
    except RuntimeError:
        computed_results = _compute_pairs(
            pairs_to_compute,
            records_by_id,
            analysis_config,
            num_workers=1,
            chunk_size=chunk_size,
            progress_callback=progress_callback,
        )
        for result in computed_results:
            result.explanation = (
                result.explanation
                + " Le calcul parallele a echoue ; cette paire a ete recalculee en mode debug sequentiel."
            )

    if use_cache:
        rows_to_store = [
            (result.id_A, result.id_B, result.similarity_result)
            for result in computed_results
            if result.similarity_result is not None
        ]
        if progress_callback is not None and rows_to_store:
            progress_callback("Mise a jour du cache", 0, len(rows_to_store))
        store_pair_results_batch(rows_to_store, analysis_config)
        if progress_callback is not None and rows_to_store:
            progress_callback("Mise a jour du cache", len(rows_to_store), len(rows_to_store))

    all_results = cached_results + computed_results
    elapsed = perf_counter() - started_at
    cached_count = len(cached_results)
    for result in all_results:
        result.explanation = result.explanation or f"Score calcule en {elapsed:.3f} s."
    return all_results


def compute_pair_chunk(
    chunk: list[tuple[str, str]],
    serialized_records: dict[str, dict[str, Any]],
    serialized_config: dict[str, Any],
) -> list[PairMatchingResult]:
    records_by_id = {
        case_id: deserialize_record(record_data) for case_id, record_data in serialized_records.items()
    }
    config = deserialize_config(serialized_config)
    results: list[PairMatchingResult] = []
    for id_a, id_b in chunk:
        record_a = records_by_id[id_a]
        record_b = records_by_id[id_b]
        similarity = compute_combined_similarity(record_a, record_b, config)
        results.append(pair_result_from_similarity(record_a, record_b, similarity))
    return results


def should_use_parallelism(
    num_pairs: int,
    num_workers: int,
    phonetic_enabled: bool,
    auto_parallel_threshold_pairs: int = 500,
    auto_parallel_threshold_pairs_with_phonetic: int = 100,
) -> bool:
    if num_workers <= 1:
        return False
    threshold = auto_parallel_threshold_pairs_with_phonetic if phonetic_enabled else auto_parallel_threshold_pairs
    return num_pairs >= threshold


def serialize_record(record: MissingPersonRecord) -> dict[str, Any]:
    data = asdict(record)
    data["disappearance_date_parsed"] = _serialize_date(record.disappearance_date_parsed)
    return data


def deserialize_record(data: dict[str, Any]) -> MissingPersonRecord:
    copied = dict(data)
    copied["disappearance_date_parsed"] = _deserialize_date(copied.get("disappearance_date_parsed"))
    return MissingPersonRecord(**copied)


def serialize_config(config: AnalysisConfig) -> dict[str, Any]:
    return asdict(config)


def deserialize_config(data: dict[str, Any]) -> AnalysisConfig:
    return AnalysisConfig(**data)


def pair_result_from_similarity(
    record_a: MissingPersonRecord,
    record_b: MissingPersonRecord,
    similarity: CombinedSimilarityResult,
) -> PairMatchingResult:
    return PairMatchingResult(
        id_A=record_a.case_id,
        id_B=record_b.case_id,
        name_A=record_a.full_name_raw,
        name_B=record_b.full_name_raw,
        score_final=similarity.score_final,
        score_name=similarity.name_result.score_final if similarity.name_result else None,
        score_phonetic=similarity.phonetic_result.score_final if similarity.phonetic_result else None,
        phonetic_status=similarity.phonetic_result.status if similarity.phonetic_result else None,
        score_geographic=similarity.geo_result.score if similarity.geo_result else None,
        score_temporal=similarity.temporal_result.score if similarity.temporal_result else None,
        name_direct_structured_score=(
            similarity.name_result.direct_structured_score if similarity.name_result else None
        ),
        name_direct_full_name_score=(
            similarity.name_result.direct_full_name_score if similarity.name_result else None
        ),
        name_reordered_full_name_score=(
            similarity.name_result.reordered_full_name_score if similarity.name_result else None
        ),
        name_token_alignment_score=(
            similarity.name_result.token_alignment_score if similarity.name_result else None
        ),
        name_order_mismatch_suspected=(
            similarity.name_result.order_mismatch_suspected if similarity.name_result else None
        ),
        name_best_method=similarity.name_result.best_method if similarity.name_result else None,
        distance_km=similarity.geo_result.distance_km if similarity.geo_result else None,
        distance_days=similarity.temporal_result.distance_days if similarity.temporal_result else None,
        explanation=similarity.explanation,
        similarity_result=similarity,
    )


def _compute_pairs(
    pairs_to_compute: list[tuple[str, str]],
    records_by_id: dict[str, MissingPersonRecord],
    analysis_config: AnalysisConfig,
    num_workers: int,
    chunk_size: int,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> list[PairMatchingResult]:
    if not pairs_to_compute:
        return []
    if num_workers <= 1:
        serialized_records = {case_id: serialize_record(record) for case_id, record in records_by_id.items()}
        results: list[PairMatchingResult] = []
        processed = 0
        chunks = _chunk_pairs(pairs_to_compute, chunk_size)
        total_chunks = len(chunks)
        for chunk_number, chunk in enumerate(chunks, start=1):
            results.extend(compute_pair_chunk(chunk, serialized_records, serialize_config(analysis_config)))
            processed += len(chunk)
            if progress_callback is not None:
                progress_callback(
                    f"Calcul des scores - chunk {chunk_number}/{total_chunks}",
                    processed,
                    len(pairs_to_compute),
                )
        return results

    chunks = _chunk_pairs(pairs_to_compute, chunk_size)
    total_chunks = len(chunks)
    serialized_records = {case_id: serialize_record(record) for case_id, record in records_by_id.items()}
    serialized_config = serialize_config(analysis_config)
    results: list[PairMatchingResult] = []
    completed_chunks = 0
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(compute_pair_chunk, chunk, serialized_records, serialized_config)
                for chunk in chunks
            ]
            for future in as_completed(futures):
                chunk_results = future.result()
                results.extend(chunk_results)
                completed_chunks += 1
                if progress_callback is not None:
                    progress_callback(
                        f"Calcul des scores en parallele - chunk {completed_chunks}/{total_chunks}",
                        min(len(results), len(pairs_to_compute)),
                        len(pairs_to_compute),
                    )
    except Exception as exc:
        raise RuntimeError(
            "Une erreur est survenue pendant le calcul parallele. "
            "Relancez en mode debug pour identifier le probleme."
        ) from exc
    result_order = {pair: index for index, pair in enumerate(pairs_to_compute)}
    results.sort(key=lambda result: result_order.get((result.id_A, result.id_B), 0))
    return results


def _normalize_pairs(
    pairs: list[tuple[MissingPersonRecord, MissingPersonRecord]] | list[tuple[str, str]],
) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for left, right in pairs:
        id_a = left.case_id if isinstance(left, MissingPersonRecord) else str(left)
        id_b = right.case_id if isinstance(right, MissingPersonRecord) else str(right)
        normalized.append((id_a, id_b))
    return normalized


def _cached_pair_result(
    id_a: str,
    id_b: str,
    records_by_id: dict[str, MissingPersonRecord],
    cached_row: dict[str, Any],
) -> PairMatchingResult:
    record_a = records_by_id[id_a]
    record_b = records_by_id[id_b]
    similarity = cached_row_to_result(cached_row)
    result = pair_result_from_similarity(record_a, record_b, similarity)
    result.score_name = _optional_float(cached_row.get("score_name"))
    result.score_phonetic = _optional_float(cached_row.get("score_phonetic"))
    result.phonetic_status = _optional_str(cached_row.get("phonetic_status"))
    result.score_geographic = _optional_float(cached_row.get("score_geographic"))
    result.score_temporal = _optional_float(cached_row.get("score_temporal"))
    result.name_direct_structured_score = _optional_float(cached_row.get("name_direct_structured_score"))
    result.name_direct_full_name_score = _optional_float(cached_row.get("name_direct_full_name_score"))
    result.name_reordered_full_name_score = _optional_float(cached_row.get("name_reordered_full_name_score"))
    result.name_token_alignment_score = _optional_float(cached_row.get("name_token_alignment_score"))
    result.name_order_mismatch_suspected = _optional_bool(cached_row.get("name_order_mismatch_suspected"))
    result.name_best_method = _optional_str(cached_row.get("name_best_method"))
    result.distance_km = _optional_float(cached_row.get("distance_km"))
    result.distance_days = _optional_int(cached_row.get("distance_days"))
    result.similarity_result = None
    result.from_cache = True
    return result


def _chunk_pairs(pairs: list[tuple[str, str]], chunk_size: int) -> list[list[tuple[str, str]]]:
    safe_chunk_size = max(int(chunk_size), 1)
    return [pairs[index : index + safe_chunk_size] for index in range(0, len(pairs), safe_chunk_size)]


def _serialize_date(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, (date, datetime)):
        return str(value)
    return value.isoformat()


def _deserialize_date(value: object) -> date | datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime) or isinstance(value, date):
        return value
    text = str(value)
    try:
        if "T" in text:
            return datetime.fromisoformat(text)
        return date.fromisoformat(text)
    except ValueError:
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or str(value) == "nan":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or str(value) == "nan":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None or str(value) == "nan":
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _optional_str(value: Any) -> str | None:
    if value is None or str(value) == "nan":
        return None
    return str(value)
