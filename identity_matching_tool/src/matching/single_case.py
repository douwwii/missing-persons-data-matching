from __future__ import annotations

from src.models import AnalysisConfig, CombinedSimilarityResult, MissingPersonRecord
from src.similarity.combined_similarity import compute_combined_similarity


def find_single_case_matches(
    records: list[MissingPersonRecord],
    selected_case_id: str,
    config: AnalysisConfig,
) -> list[tuple[MissingPersonRecord, CombinedSimilarityResult]]:
    """Compute sorted matches for one selected case against all other records."""
    target = next((record for record in records if record.case_id == selected_case_id), None)
    if target is None:
        raise ValueError(f"Target case not found: {selected_case_id}")

    results: list[tuple[MissingPersonRecord, CombinedSimilarityResult]] = []
    for candidate in records:
        if candidate.case_id == selected_case_id:
            continue
        similarity = compute_combined_similarity(target, candidate, config)
        if similarity.score_final is None:
            continue
        if similarity.score_final < float(config.filter_parameters.get("min_score_threshold", 0.0)):
            continue
        results.append((candidate, similarity))

    results.sort(key=lambda item: item[1].score_final or 0.0, reverse=True)
    max_results = int(config.filter_parameters.get("max_results", 50))
    return results[:max_results]
