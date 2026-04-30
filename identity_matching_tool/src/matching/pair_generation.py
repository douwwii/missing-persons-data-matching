from __future__ import annotations

from itertools import combinations

from src.models import MissingPersonRecord


def generate_all_pairs(records: list[MissingPersonRecord]) -> list[tuple[MissingPersonRecord, MissingPersonRecord]]:
    return list(combinations(records, 2))


def generate_single_case_pairs(
    records: list[MissingPersonRecord],
    selected_case_id: str,
) -> list[tuple[MissingPersonRecord, MissingPersonRecord]]:
    target = next((record for record in records if record.case_id == selected_case_id), None)
    if target is None:
        raise ValueError(f"Dossier cible introuvable : {selected_case_id}")
    return [(target, record) for record in records if record.case_id != selected_case_id]


def theoretical_pair_count(records_count: int, selected_case_id: str | None = None) -> int:
    if selected_case_id is not None:
        return max(records_count - 1, 0)
    return records_count * (records_count - 1) // 2

