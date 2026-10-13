from __future__ import annotations

from itertools import combinations

from identity_matching.models import AnalysisConfig, CombinedSimilarityResult, MissingPersonRecord
from identity_matching.similarity.combined_similarity import compute_combined_similarity


def find_global_matches(
    records: list[MissingPersonRecord],
    config: AnalysisConfig,
) -> list[tuple[MissingPersonRecord, MissingPersonRecord, CombinedSimilarityResult]]:
    """Compute sorted pairwise matches across all records."""
    results: list[tuple[MissingPersonRecord, MissingPersonRecord, CombinedSimilarityResult]] = []
    threshold = float(config.filter_parameters.get("min_score_threshold", 0.0))

    for record_a, record_b in combinations(records, 2):
        similarity = compute_combined_similarity(record_a, record_b, config)
        if similarity.score_final is None or similarity.score_final < threshold:
            continue
        results.append((record_a, record_b, similarity))

    results.sort(key=lambda item: item[2].score_final or 0.0, reverse=True)
    max_results = int(config.filter_parameters.get("max_results", 50))
    return results[:max_results]
