from __future__ import annotations

from identity_matching.models import (
    AnalysisConfig,
    CombinedSimilarityResult,
    GeoSimilarityResult,
    MissingPersonRecord,
    NameSimilarityResult,
    PhoneticSimilarityResult,
    TemporalSimilarityResult,
)
from identity_matching.similarity.geo_similarity import compute_geo_similarity
from identity_matching.similarity.name_similarity import compute_name_similarity
from identity_matching.similarity.phonetic_similarity import compute_phonetic_similarity
from identity_matching.similarity.temporal_similarity import compute_temporal_similarity


def compute_combined_similarity(
    record_a: MissingPersonRecord,
    record_b: MissingPersonRecord,
    config: AnalysisConfig,
) -> CombinedSimilarityResult:
    """Compute enabled similarity filters and combine their calculable scores."""
    enabled_filters = dict(config.enabled_filters)
    parameters = config.filter_parameters
    cache_context_needed = sum(1 for enabled in enabled_filters.values() if enabled) < 2

    name_result = compute_name_similarity(record_a, record_b) if enabled_filters.get("name") else None
    phonetic_result = (
        compute_phonetic_similarity(record_a, record_b, lang=str(parameters.get("phonetic_lang", "spa-Latn")))
        if enabled_filters.get("phonetic")
        else None
    )
    geo_result = (
        compute_geo_similarity(
            record_a,
            record_b,
            distance_min_km=float(parameters.get("geographic_distance_min_km", 5.0)),
            distance_max_km=float(parameters.get("geographic_distance_max_km", 50.0)),
        )
        if enabled_filters.get("geographic") or cache_context_needed
        else None
    )
    temporal_result = (
        compute_temporal_similarity(
            record_a,
            record_b,
            max_days=int(parameters.get("temporal_max_days", 365)),
        )
        if enabled_filters.get("temporal") or cache_context_needed
        else None
    )

    weighted_scores = _weighted_scores(
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

    explanation = _build_combined_explanation(
        weighted_scores=weighted_scores,
        name_result=name_result,
        phonetic_result=phonetic_result,
        geo_result=geo_result,
        temporal_result=temporal_result,
        score_final=score_final,
    )

    return CombinedSimilarityResult(
        score_final=_clamp_optional(score_final),
        name_result=name_result,
        phonetic_result=phonetic_result,
        geo_result=geo_result,
        temporal_result=temporal_result,
        enabled_filters=enabled_filters,
        explanation=explanation,
        status=status,
    )


def _weighted_scores(
    parameters: dict[str, float | int],
    enabled_filters: dict[str, bool],
    name_result: NameSimilarityResult | None,
    phonetic_result: PhoneticSimilarityResult | None,
    geo_result: GeoSimilarityResult | None,
    temporal_result: TemporalSimilarityResult | None,
) -> list[tuple[float, float, str]]:
    """Collect calculable enabled scores with their configured weights."""
    weighted: list[tuple[float, float, str]] = []
    if enabled_filters.get("name") and name_result is not None and name_result.status == "ok":
        weighted.append((name_result.score_final, float(parameters.get("name_weight", 1.0)), "name"))
    if (
        enabled_filters.get("phonetic")
        and phonetic_result is not None
        and phonetic_result.status == "ok"
        and phonetic_result.score_final is not None
    ):
        weighted.append(
            (phonetic_result.score_final, float(parameters.get("phonetic_weight", 1.0)), "phonetic")
        )
    if (
        enabled_filters.get("geographic")
        and geo_result is not None
        and geo_result.status == "ok"
        and geo_result.score is not None
    ):
        weighted.append((geo_result.score, float(parameters.get("geographic_weight", 1.0)), "location"))
    if (
        enabled_filters.get("temporal")
        and temporal_result is not None
        and temporal_result.status == "ok"
        and temporal_result.score is not None
    ):
        weighted.append((temporal_result.score, float(parameters.get("temporal_weight", 1.0)), "time"))
    return [(score, weight, label) for score, weight, label in weighted if weight > 0]


def _build_combined_explanation(
    weighted_scores: list[tuple[float, float, str]],
    name_result: NameSimilarityResult | None,
    phonetic_result: PhoneticSimilarityResult | None,
    geo_result: GeoSimilarityResult | None,
    temporal_result: TemporalSimilarityResult | None,
    score_final: float | None,
) -> str:
    """Build a human-readable explanation for the combined score."""
    parts: list[str] = []
    if score_final is None:
        parts.append("No combined score can be computed with the enabled filters.")
    else:
        labels = ", ".join(label for _, _, label in weighted_scores)
        parts.append(f"Combined score {score_final:.3f}, computed with the following dimensions: {labels}.")

    skipped = []
    for label, result in [
        ("name", name_result),
        ("phonetic", phonetic_result),
        ("location", geo_result),
        ("time", temporal_result),
    ]:
        if result is not None and result.status != "ok":
            skipped.append(f"{label} ({result.status})")
    if skipped:
        parts.append("Dimensions ignored because they cannot be computed: " + ", ".join(skipped) + ".")

    if name_result is not None and name_result.order_mismatch_suspected:
        parts.append(
            "The name has high similarity after token reordering, suggesting an inversion "
            "or a different order of name elements."
        )
        if (
            phonetic_result is not None
            and phonetic_result.status == "ok"
            and phonetic_result.score_final is not None
            and phonetic_result.score_final < 0.70
        ):
            parts.append("The reordered phonetic score remains low despite reordering the name elements.")

    return " ".join(parts)


def _clamp_optional(value: float | None) -> float | None:
    """Clamp a score when present and preserve None when absent."""
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))
