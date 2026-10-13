from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

from identity_matching.models import GeoSimilarityResult, MissingPersonRecord
from identity_matching.preprocessing.safe_parsing import parse_optional_float


def compute_geo_similarity(
    record_a: MissingPersonRecord,
    record_b: MissingPersonRecord,
    distance_min_km: float,
    distance_max_km: float,
) -> GeoSimilarityResult:
    """Compute geographic similarity from existing GPS coordinates."""
    if distance_min_km < 0 or distance_max_km <= distance_min_km:
        return GeoSimilarityResult(
            score=None,
            distance_km=None,
            explanation="Geographic similarity cannot be computed: invalid distance parameters.",
            status="invalid_parameters",
        )

    lat_a = parse_optional_float(record_a.latitude)
    lon_a = parse_optional_float(record_a.longitude)
    lat_b = parse_optional_float(record_b.latitude)
    lon_b = parse_optional_float(record_b.longitude)
    if not _valid_coordinates(lat_a, lon_a) or not _valid_coordinates(lat_b, lon_b):
        return GeoSimilarityResult(
            score=None,
            distance_km=None,
            explanation="Geographic similarity cannot be computed: missing or invalid GPS coordinates.",
            status="missing_coordinates",
        )

    distance_km = haversine_distance_km(lat_a, lon_a, lat_b, lon_b)
    if distance_km <= distance_min_km:
        score = 1.0
    elif distance_km >= distance_max_km:
        score = 0.0
    else:
        score = 1 - (distance_km - distance_min_km) / (distance_max_km - distance_min_km)

    return GeoSimilarityResult(
        score=_clamp(score),
        distance_km=distance_km,
        explanation=f"GPS distance of {distance_km:.1f} km between the two case records.",
        status="ok",
    )


def haversine_distance_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Return the great-circle distance in kilometers between two GPS points."""
    earth_radius_km = 6371.0088
    delta_lat = radians(lat_b - lat_a)
    delta_lon = radians(lon_b - lon_a)
    rad_lat_a = radians(lat_a)
    rad_lat_b = radians(lat_b)
    haversine = sin(delta_lat / 2) ** 2 + cos(rad_lat_a) * cos(rad_lat_b) * sin(delta_lon / 2) ** 2
    return 2 * earth_radius_km * asin(sqrt(haversine))


def _valid_coordinates(latitude: float | None, longitude: float | None) -> bool:
    """Return True when both coordinate values are present and inside GPS ranges."""
    if latitude is None or longitude is None:
        return False
    return -90 <= latitude <= 90 and -180 <= longitude <= 180


def _clamp(value: float) -> float:
    """Constrain a numeric score to the inclusive 0..1 range."""
    return max(0.0, min(1.0, float(value)))
