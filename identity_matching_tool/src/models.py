from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import os
from typing import Any, Literal


@dataclass(slots=True)
class MissingPersonRecord:
    case_id: str
    full_name_raw: str
    full_name_normalized: str
    name_tokens: list[str]
    disappearance_city_verified: str
    first_name_raw: str | None = None
    first_name_normalized: str | None = None
    last_name_raw: str | None = None
    last_name_normalized: str | None = None
    disappearance_department_verified: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    disappearance_date_parsed: date | datetime | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnalysisConfig:
    investigation_mode: Literal["single_case", "global"] = "single_case"
    analysis_mode: Literal["exhaustive", "standard_blocking", "custom_placeholder"] = "exhaustive"
    selected_case_id: str | None = None
    enabled_filters: dict[str, bool] = field(
        default_factory=lambda: {
            "name": True,
            "phonetic": False,
            "geographic": False,
            "temporal": False,
        }
    )
    filter_parameters: dict[str, float | int] = field(
        default_factory=lambda: {
            "geographic_distance_min_km": 5.0,
            "geographic_distance_max_km": 50.0,
            "temporal_max_days": 365,
            "max_results": 50,
            "min_score_threshold": 0.0,
        }
    )


@dataclass(slots=True)
class DatasetValidationResult:
    is_valid: bool
    missing_required_columns: list[str] = field(default_factory=list)
    missing_recommended_columns: list[str] = field(default_factory=list)
    total_rows: int = 0
    valid_coordinates_count: int = 0
    missing_coordinates_count: int = 0
    valid_dates_count: int = 0
    missing_dates_count: int = 0
    duplicated_case_ids_count: int = 0
    missing_case_ids_count: int = 0
    message: str = ""


@dataclass(slots=True)
class NameSimilarityResult:
    score_final: float
    direct_structured_score: float | None
    direct_full_name_score: float | None
    reordered_full_name_score: float | None
    token_alignment_score: float | None
    order_mismatch_suspected: bool
    best_method: str | None
    matched_tokens: list[dict[str, Any]] = field(default_factory=list)
    explanation: str = ""
    status: str = "ok"


@dataclass(slots=True)
class PhoneticSimilarityResult:
    score_final: float | None
    ipa_a: str | None = None
    ipa_b: str | None = None
    explanation: str = ""
    status: str = "ok"


@dataclass(slots=True)
class GeoSimilarityResult:
    score: float | None
    distance_km: float | None = None
    explanation: str = ""
    status: str = "ok"


@dataclass(slots=True)
class TemporalSimilarityResult:
    score: float | None
    distance_days: int | None = None
    explanation: str = ""
    status: str = "ok"


@dataclass(slots=True)
class CombinedSimilarityResult:
    score_final: float | None
    name_result: NameSimilarityResult | None = None
    phonetic_result: PhoneticSimilarityResult | None = None
    geo_result: GeoSimilarityResult | None = None
    temporal_result: TemporalSimilarityResult | None = None
    enabled_filters: dict[str, bool] = field(default_factory=dict)
    explanation: str = ""
    status: str = "ok"


@dataclass(slots=True)
class BlockingConfig:
    date_threshold_name_rules: int = 60
    geo_threshold_km: float = 20.0
    date_threshold_geo_rule: int = 10
    selected_case_id: str | None = None


@dataclass(slots=True)
class ParallelConfig:
    enable_parallelism: bool = True
    num_workers: int = field(default_factory=lambda: max(os.cpu_count() or 1, 1))
    chunk_size: int = 200
    auto_parallel_threshold_pairs: int = 500
    auto_parallel_threshold_pairs_with_phonetic: int = 100
    debug_mode: bool = False


@dataclass(slots=True)
class BlockingSignature:
    case_id: str
    tokens: list[str]
    initials_bag: dict[str, int]
    phonetic_onset_bag: dict[str, int]
    parsed_date: date | None
    latitude: float | None
    longitude: float | None
    city: str | None
    department: str | None
    full_name_normalized: str


@dataclass(slots=True)
class BlockingIndex:
    cache_key: str
    signatures: dict[str, BlockingSignature] = field(default_factory=dict)
    initials_index: dict[tuple[tuple[str, ...], int], list[str]] = field(default_factory=dict)
    phonetic_onset_index: dict[tuple[tuple[str, ...], int], list[str]] = field(default_factory=dict)
    geo_time_index: dict[tuple[int, int, int], list[str]] = field(default_factory=dict)
    export_dir: str | None = None


@dataclass(slots=True)
class CandidatePair:
    id_A: str
    id_B: str
    name_A: str
    name_B: str
    date_A: date | None = None
    date_B: date | None = None
    date_diff_days: int | None = None
    city_A: str | None = None
    city_B: str | None = None
    department_A: str | None = None
    department_B: str | None = None
    geo_distance_km: float | None = None
    matched_rules: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BlockingResult:
    candidate_pairs: list[CandidatePair] = field(default_factory=list)
    total_records: int = 0
    theoretical_pairs_count: int = 0
    candidate_pairs_count: int = 0
    reduction_percentage: float = 0.0
    r1_count: int = 0
    r2_count: int = 0
    r4_count: int = 0
    elapsed_time: float = 0.0
    blocking_index_cache_dir: str | None = None
    explanation: str = ""


@dataclass(slots=True)
class PairMatchingResult:
    id_A: str
    id_B: str
    name_A: str
    name_B: str
    score_final: float | None
    score_name: float | None = None
    score_phonetic: float | None = None
    phonetic_status: str | None = None
    score_geographic: float | None = None
    score_temporal: float | None = None
    name_direct_structured_score: float | None = None
    name_direct_full_name_score: float | None = None
    name_reordered_full_name_score: float | None = None
    name_token_alignment_score: float | None = None
    name_order_mismatch_suspected: bool | None = None
    name_best_method: str | None = None
    distance_km: float | None = None
    distance_days: int | None = None
    date_diff_days: int | None = None
    geo_distance_km: float | None = None
    matched_rules: list[str] = field(default_factory=list)
    explanation: str = ""
    similarity_result: CombinedSimilarityResult | None = None
    from_cache: bool = False
    pair_parallel_enabled: bool = False
    pair_num_workers: int = 1


@dataclass(slots=True)
class MatchingRunResult:
    status: str
    analysis_mode: str
    investigation_mode: str
    results: list[PairMatchingResult] = field(default_factory=list)
    blocking_result: BlockingResult | None = None
    total_records: int = 0
    theoretical_pairs_count: int = 0
    compared_pairs_count: int = 0
    retained_results_count: int = 0
    reduction_percentage: float | None = None
    elapsed_time: float = 0.0
    parallel_enabled: bool = False
    num_workers: int = 1
    chunk_size: int = 200
    total_pairs: int = 0
    cached_pairs: int = 0
    computed_pairs: int = 0
    cache_hit_rate: float = 0.0
    parallel_elapsed_time: float = 0.0
    execution_mode: str = "sequential"
    explanation: str = ""
