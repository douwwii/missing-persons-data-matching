from __future__ import annotations

from datetime import date

import pytest

from identity_matching.models import MissingPersonRecord
from identity_matching.similarity.temporal_similarity import compute_temporal_similarity


def record(value: object) -> MissingPersonRecord:
    return MissingPersonRecord(
        case_id="x",
        full_name_raw="a",
        full_name_normalized="a",
        name_tokens=["a"],
        disappearance_city_verified="city",
        disappearance_date_parsed=value,
    )


def test_same_date_scores_one() -> None:
    result = compute_temporal_similarity(record(date(2024, 1, 1)), record(date(2024, 1, 1)), 365)

    assert result.status == "ok"
    assert result.distance_days == 0
    assert result.score == 1.0


def test_near_date_interpolates() -> None:
    result = compute_temporal_similarity(record("2024-01-01"), record("2024-07-01"), 365)

    assert result.distance_days == 182
    assert result.score == pytest.approx(1 - 182 / 365)


def test_far_date_has_low_score() -> None:
    result = compute_temporal_similarity(record("2024-01-01"), record("2024-10-01"), 365)

    assert result.status == "ok"
    assert result.score is not None
    assert 0 < result.score < 0.30


def test_date_above_max_days_scores_zero() -> None:
    result = compute_temporal_similarity(record("2024-01-01"), record("2025-01-01"), 365)

    assert result.score == 0.0


def test_missing_date() -> None:
    result = compute_temporal_similarity(record(None), record("2024-01-01"), 365)

    assert result.status == "missing_date"
    assert result.score is None


def test_invalid_max_days() -> None:
    result = compute_temporal_similarity(record("2024-01-01"), record("2024-01-02"), 0)

    assert result.status == "invalid_parameters"
    assert result.score is None
