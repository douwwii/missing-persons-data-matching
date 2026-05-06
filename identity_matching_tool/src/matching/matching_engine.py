from __future__ import annotations

from time import perf_counter
from typing import Callable

from src.matching.blocking import generate_standard_blocking_pairs
from src.matching.pair_generation import generate_all_pairs, generate_single_case_pairs, theoretical_pair_count
from src.matching.parallel import parallel_compute_pair_scores, should_use_parallelism
from src.models import (
    AnalysisConfig,
    BlockingConfig,
    CandidatePair,
    CombinedSimilarityResult,
    MatchingRunResult,
    MissingPersonRecord,
    PairMatchingResult,
    ParallelConfig,
)


def run_matching(
    records: list[MissingPersonRecord],
    analysis_config: AnalysisConfig,
    blocking_config: BlockingConfig | None = None,
    parallel_config: ParallelConfig | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> MatchingRunResult:
    """Run matching with the configured analysis mode."""
    if analysis_config.analysis_mode == "custom_placeholder":
        return MatchingRunResult(
            status="not_implemented",
            analysis_mode=analysis_config.analysis_mode,
            investigation_mode=analysis_config.investigation_mode,
            total_records=len(records),
            theoretical_pairs_count=_theoretical_for_config(records, analysis_config),
            explanation="Custom mode will be added in a future version.",
        )
    if analysis_config.analysis_mode == "standard_blocking":
        return run_standard_blocking_matching(records, analysis_config, blocking_config, parallel_config, progress_callback)
    return run_exhaustive_matching(records, analysis_config, parallel_config, progress_callback)


def run_exhaustive_matching(
    records: list[MissingPersonRecord],
    analysis_config: AnalysisConfig,
    parallel_config: ParallelConfig | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> MatchingRunResult:
    """Score every pair in scope and return sorted retained results."""
    started_at = perf_counter()
    if progress_callback is not None:
        progress_callback("Pair generation", 0, 1)
    pairs = _pairs_for_config(records, analysis_config)
    if progress_callback is not None:
        progress_callback("Pair generation", 1, 1)
    threshold = float(analysis_config.filter_parameters.get("min_score_threshold", 0.0))
    max_results = int(analysis_config.filter_parameters.get("max_results", 50))
    parallel_config = _effective_parallel_config(parallel_config)
    scoring_started_at = perf_counter()
    pair_results = _score_pairs(records, pairs, analysis_config, parallel_config, use_cache=True, progress_callback=progress_callback)
    scoring_elapsed = perf_counter() - scoring_started_at

    retained = [
        pair_result
        for pair_result in pair_results
        if pair_result.score_final is not None and pair_result.score_final >= threshold
    ]

    retained.sort(key=lambda result: result.score_final or 0.0, reverse=True)
    retained = retained[:max_results]
    elapsed = perf_counter() - started_at
    theoretical = _theoretical_for_config(records, analysis_config)
    stats = _parallel_stats(len(pairs), pair_results, parallel_config, scoring_elapsed)
    return MatchingRunResult(
        status="ok",
        analysis_mode="exhaustive",
        investigation_mode=analysis_config.investigation_mode,
        results=retained,
        total_records=len(records),
        theoretical_pairs_count=theoretical,
        compared_pairs_count=len(pairs),
        retained_results_count=len(retained),
        elapsed_time=elapsed,
        **stats,
        explanation=(
            "Exhaustive / validation mode: every pair in the selected scope was compared. "
            "This mode can be slow on a large dataset."
        ),
    )


def run_standard_blocking_matching(
    records: list[MissingPersonRecord],
    analysis_config: AnalysisConfig,
    blocking_config: BlockingConfig | None = None,
    parallel_config: ParallelConfig | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> MatchingRunResult:
    """Generate standard blocking candidates, score them, and return sorted results."""
    started_at = perf_counter()
    if progress_callback is not None:
        progress_callback("Preparing blocking", 0, 1)
    blocking_config = blocking_config or BlockingConfig()
    if analysis_config.investigation_mode == "single_case":
        blocking_config = BlockingConfig(
            date_threshold_name_rules=blocking_config.date_threshold_name_rules,
            geo_threshold_km=blocking_config.geo_threshold_km,
            date_threshold_geo_rule=blocking_config.date_threshold_geo_rule,
            selected_case_id=analysis_config.selected_case_id,
        )
    blocking_result = generate_standard_blocking_pairs(records, blocking_config)
    if progress_callback is not None:
        progress_callback("Preparing blocking", 1, 1)
        progress_callback(
            "Candidate pairs found",
            blocking_result.candidate_pairs_count,
            max(blocking_result.theoretical_pairs_count, 1),
        )
    threshold = float(analysis_config.filter_parameters.get("min_score_threshold", 0.0))
    max_results = int(analysis_config.filter_parameters.get("max_results", 50))
    parallel_config = _effective_parallel_config(parallel_config)

    candidate_pairs = [(candidate.id_A, candidate.id_B) for candidate in blocking_result.candidate_pairs]
    candidate_by_key = {
        (candidate.id_A, candidate.id_B): candidate for candidate in blocking_result.candidate_pairs
    }
    scoring_started_at = perf_counter()
    pair_results = _score_pairs(
        records,
        candidate_pairs,
        analysis_config,
        parallel_config,
        use_cache=True,
        progress_callback=progress_callback,
    )
    scoring_elapsed = perf_counter() - scoring_started_at
    for pair_result in pair_results:
        candidate = candidate_by_key.get((pair_result.id_A, pair_result.id_B))
        if candidate is None:
            continue
        pair_result.date_diff_days = candidate.date_diff_days
        pair_result.geo_distance_km = candidate.geo_distance_km
        pair_result.matched_rules = candidate.matched_rules

    retained = [
        pair_result
        for pair_result in pair_results
        if pair_result.score_final is not None and pair_result.score_final >= threshold
    ]

    retained.sort(key=lambda result: result.score_final or 0.0, reverse=True)
    retained = retained[:max_results]
    elapsed = perf_counter() - started_at
    stats = _parallel_stats(len(candidate_pairs), pair_results, parallel_config, scoring_elapsed)
    return MatchingRunResult(
        status="ok",
        analysis_mode="standard_blocking",
        investigation_mode=analysis_config.investigation_mode,
        results=retained,
        blocking_result=blocking_result,
        total_records=len(records),
        theoretical_pairs_count=blocking_result.theoretical_pairs_count,
        compared_pairs_count=blocking_result.candidate_pairs_count,
        retained_results_count=len(retained),
        reduction_percentage=blocking_result.reduction_percentage,
        elapsed_time=elapsed,
        **stats,
        explanation="Standard blocking mode: only candidate pairs retained by R1, R2, or R3 were scored.",
    )


def _pairs_for_config(
    records: list[MissingPersonRecord],
    analysis_config: AnalysisConfig,
) -> list[tuple[MissingPersonRecord, MissingPersonRecord]]:
    """Generate pairs according to global or single-case investigation mode."""
    if analysis_config.investigation_mode == "single_case":
        return generate_single_case_pairs(records, analysis_config.selected_case_id or "")
    return generate_all_pairs(records)


def _theoretical_for_config(records: list[MissingPersonRecord], analysis_config: AnalysisConfig) -> int:
    """Return the theoretical pair count for the selected investigation mode."""
    selected_case_id = analysis_config.selected_case_id if analysis_config.investigation_mode == "single_case" else None
    return theoretical_pair_count(len(records), selected_case_id)


def _pair_result(
    record_a: MissingPersonRecord,
    record_b: MissingPersonRecord,
    similarity: CombinedSimilarityResult,
    candidate: CandidatePair | None = None,
) -> PairMatchingResult:
    """Convert detailed similarity output into a pair matching result."""
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
        distance_km=similarity.geo_result.distance_km if similarity.geo_result else None,
        distance_days=similarity.temporal_result.distance_days if similarity.temporal_result else None,
        date_diff_days=candidate.date_diff_days if candidate else None,
        geo_distance_km=candidate.geo_distance_km if candidate else None,
        matched_rules=candidate.matched_rules if candidate else [],
        explanation=similarity.explanation,
        similarity_result=similarity,
    )


def _score_pairs(
    records: list[MissingPersonRecord],
    pairs: list[tuple[MissingPersonRecord, MissingPersonRecord]] | list[tuple[str, str]],
    analysis_config: AnalysisConfig,
    parallel_config: ParallelConfig,
    use_cache: bool,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> list[PairMatchingResult]:
    """Score pairs using cache-aware sequential or parallel execution."""
    records_by_id = {record.case_id: record for record in records}
    normalized_pair_count = len(pairs)
    workers = 1 if parallel_config.debug_mode else parallel_config.num_workers
    phonetic_enabled = analysis_config.enabled_filters.get("phonetic", False)
    use_parallel = parallel_config.enable_parallelism and should_use_parallelism(
        num_pairs=normalized_pair_count,
        num_workers=workers,
        phonetic_enabled=phonetic_enabled,
        auto_parallel_threshold_pairs=parallel_config.auto_parallel_threshold_pairs,
        auto_parallel_threshold_pairs_with_phonetic=parallel_config.auto_parallel_threshold_pairs_with_phonetic,
    )
    effective_workers = workers if use_parallel else 1
    results = parallel_compute_pair_scores(
        pairs=pairs,
        records_by_id=records_by_id,
        analysis_config=analysis_config,
        num_workers=effective_workers,
        chunk_size=parallel_config.chunk_size,
        use_cache=use_cache,
        progress_callback=progress_callback,
    )
    for result in results:
        result.pair_parallel_enabled = use_parallel
        result.pair_num_workers = effective_workers
    return results


def _parallel_stats(
    total_pairs: int,
    pair_results: list[PairMatchingResult],
    parallel_config: ParallelConfig,
    scoring_elapsed: float,
) -> dict[str, object]:
    """Build performance and cache statistics for a matching run."""
    cached_pairs = sum(1 for result in pair_results if result.from_cache)
    computed_pairs = len(pair_results) - cached_pairs
    parallel_enabled = any(result.pair_parallel_enabled for result in pair_results)
    num_workers = max((pair_results[0].pair_num_workers if pair_results else 1), 1)
    return {
        "parallel_enabled": parallel_enabled,
        "num_workers": num_workers,
        "chunk_size": parallel_config.chunk_size,
        "total_pairs": total_pairs,
        "cached_pairs": cached_pairs,
        "computed_pairs": computed_pairs,
        "cache_hit_rate": 0.0 if total_pairs == 0 else cached_pairs / total_pairs,
        "parallel_elapsed_time": scoring_elapsed,
        "execution_mode": "parallel" if parallel_enabled else "sequential",
    }


def _effective_parallel_config(parallel_config: ParallelConfig | None) -> ParallelConfig:
    """Normalize optional parallel configuration into safe runtime values."""
    if parallel_config is None:
        return ParallelConfig()
    if parallel_config.debug_mode:
        parallel_config.num_workers = 1
    parallel_config.num_workers = max(int(parallel_config.num_workers), 1)
    parallel_config.chunk_size = max(int(parallel_config.chunk_size), 1)
    return parallel_config
