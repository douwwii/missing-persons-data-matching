from __future__ import annotations

from src.models import MissingPersonRecord
from src.similarity.name_similarity import compute_name_similarity


def record(
    full_name: str = "",
    tokens: list[str] | None = None,
    first: str | None = None,
    last: str | None = None,
) -> MissingPersonRecord:
    return MissingPersonRecord(
        case_id="x",
        full_name_raw=full_name,
        full_name_normalized=full_name,
        first_name_normalized=first,
        last_name_normalized=last,
        name_tokens=tokens if tokens is not None else full_name.split(),
        disappearance_city_verified="city",
    )


def test_identical_names_score_near_one() -> None:
    result = compute_name_similarity(record("juan pablo payar"), record("juan pablo payar"))

    assert result.status == "ok"
    assert result.score_final > 0.99
    assert result.explanation


def test_structured_first_last_direct_score() -> None:
    result = compute_name_similarity(
        record("jose perez", first="jose", last="perez"),
        record("jose peres", first="jose", last="peres"),
    )

    assert result.direct_structured_score is not None
    assert result.direct_structured_score > 0.90
    assert result.direct_full_name_score is None
    assert result.best_method == "weighted_name_score"


def test_direct_full_name_is_fallback_when_structured_fields_are_missing() -> None:
    result = compute_name_similarity(record("jose perez"), record("jose peres"))

    assert result.direct_structured_score is None
    assert result.direct_full_name_score is not None
    assert result.direct_full_name_score > 0.90


def test_reordered_full_name_detects_order_difference() -> None:
    result = compute_name_similarity(
        record("juan pablo payar", ["juan", "pablo", "payar"]),
        record("payar juan pablo", ["payar", "juan", "pablo"]),
    )

    assert result.direct_full_name_score is not None
    assert result.reordered_full_name_score is not None
    assert result.token_alignment_score is not None
    assert result.direct_full_name_score < result.reordered_full_name_score
    assert result.reordered_full_name_score > 0.99
    assert result.token_alignment_score > 0.99
    assert result.order_mismatch_suspected is True
    assert "ordre different" in result.explanation


def test_extra_tokens_keep_high_but_not_perfect_alignment() -> None:
    result = compute_name_similarity(
        record("juan pablo exposito tetos payar"),
        record("juan pablo payar"),
    )

    assert result.score_final > 0.55
    assert result.score_final < 1.0
    assert result.token_alignment_score is not None
    assert result.token_alignment_score > 0.55
    assert result.explanation


def test_completely_different_names_are_low() -> None:
    result = compute_name_similarity(record("carlos rodriguez"), record("maria gomez"))

    assert result.score_final < 0.65


def test_single_common_token_is_penalized() -> None:
    result = compute_name_similarity(record("juan carlos ramirez"), record("juan gomez"))

    assert result.token_alignment_score is not None
    assert result.token_alignment_score < 0.70
    assert result.score_final < 0.75


def test_missing_values_do_not_crash() -> None:
    result = compute_name_similarity(record("", []), record("", []))

    assert result.status == "missing_name"
    assert result.score_final == 0.0
    assert result.explanation


def test_score_is_always_between_zero_and_one() -> None:
    result = compute_name_similarity(record("jose perez"), record("jose peres"))

    assert 0 <= result.score_final <= 1
    for score in [
        result.direct_structured_score,
        result.direct_full_name_score,
        result.reordered_full_name_score,
        result.token_alignment_score,
    ]:
        if score is not None:
            assert 0 <= score <= 1
