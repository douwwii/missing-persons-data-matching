from __future__ import annotations

from src.matching.pair_generation import generate_all_pairs, generate_single_case_pairs
from src.models import MissingPersonRecord


def record(case_id: str) -> MissingPersonRecord:
    return MissingPersonRecord(
        case_id=case_id,
        full_name_raw=case_id,
        full_name_normalized=case_id,
        name_tokens=[case_id],
        disappearance_city_verified="city",
    )


def test_generate_all_pairs_count() -> None:
    records = [record("A"), record("B"), record("C"), record("D")]

    pairs = generate_all_pairs(records)

    assert len(pairs) == 6


def test_generate_all_pairs_has_no_self_pair() -> None:
    records = [record("A"), record("B"), record("C")]

    pairs = generate_all_pairs(records)

    assert all(left.case_id != right.case_id for left, right in pairs)


def test_generate_all_pairs_has_no_reverse_duplicate() -> None:
    records = [record("A"), record("B"), record("C")]

    pairs = generate_all_pairs(records)
    normalized = {tuple(sorted([left.case_id, right.case_id])) for left, right in pairs}

    assert len(normalized) == len(pairs)


def test_single_case_generates_n_minus_one_pairs() -> None:
    records = [record("A"), record("B"), record("C"), record("D")]

    pairs = generate_single_case_pairs(records, "A")

    assert len(pairs) == 3
    assert all(left.case_id == "A" for left, _ in pairs)

