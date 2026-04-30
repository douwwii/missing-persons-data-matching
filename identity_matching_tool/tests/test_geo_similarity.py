from __future__ import annotations

import pytest

from src.models import MissingPersonRecord
from src.similarity.geo_similarity import compute_geo_similarity


def record(lat: float | None, lon: float | None) -> MissingPersonRecord:
    return MissingPersonRecord(
        case_id="x",
        full_name_raw="a",
        full_name_normalized="a",
        name_tokens=["a"],
        disappearance_city_verified="city",
        latitude=lat,
        longitude=lon,
    )


def test_zero_distance_scores_one() -> None:
    result = compute_geo_similarity(record(0, 0), record(0, 0), 10, 100)

    assert result.status == "ok"
    assert result.score == 1.0
    assert result.distance_km == pytest.approx(0)


def test_small_distance_under_min_scores_one() -> None:
    result = compute_geo_similarity(record(48.8566, 2.3522), record(48.8570, 2.3530), 10, 100)

    assert result.score == 1.0


def test_distance_above_max_scores_zero() -> None:
    result = compute_geo_similarity(record(48.8566, 2.3522), record(43.6047, 1.4442), 10, 100)

    assert result.score == 0.0


def test_interpolation_is_correct() -> None:
    result = compute_geo_similarity(record(0, 0), record(0, 0.4946), 10, 100)

    assert result.status == "ok"
    assert result.distance_km == pytest.approx(55, abs=1)
    assert result.score == pytest.approx(0.5, abs=0.03)


def test_missing_coordinates() -> None:
    result = compute_geo_similarity(record(None, 0), record(0, 0), 10, 100)

    assert result.status == "missing_coordinates"
    assert result.score is None


def test_invalid_parameters() -> None:
    result = compute_geo_similarity(record(0, 0), record(0, 0), 100, 10)

    assert result.status == "invalid_parameters"
    assert result.score is None

