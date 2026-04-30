from __future__ import annotations

from datetime import date, datetime

from src.models import MissingPersonRecord, TemporalSimilarityResult
from src.preprocessing.safe_parsing import parse_optional_date


def compute_temporal_similarity(
    record_a: MissingPersonRecord,
    record_b: MissingPersonRecord,
    max_days: int,
) -> TemporalSimilarityResult:
    if max_days <= 0:
        return TemporalSimilarityResult(
            score=None,
            distance_days=None,
            explanation="Similarite temporelle non calculable : max_days doit etre strictement positif.",
            status="invalid_parameters",
        )

    date_a = _as_date(record_a.disappearance_date_parsed)
    date_b = _as_date(record_b.disappearance_date_parsed)
    if date_a is None or date_b is None:
        return TemporalSimilarityResult(
            score=None,
            distance_days=None,
            explanation="Similarite temporelle non calculable : date de disparition manquante ou invalide.",
            status="missing_date",
        )

    distance_days = abs((date_a - date_b).days)
    score = 0.0 if distance_days >= max_days else 1 - distance_days / max_days

    return TemporalSimilarityResult(
        score=_clamp(score),
        distance_days=distance_days,
        explanation=f"Ecart de {distance_days} jour(s) entre les dates de disparition.",
        status="ok",
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


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
