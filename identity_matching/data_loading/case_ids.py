"""Validate row identifiers and prepare an explicit, lossless repair for matching."""

from __future__ import annotations

import pandas as pd


def normalized_case_ids(values: pd.Series) -> pd.Series:
    """Keep IDs as text and use consistent whitespace/missing-value handling."""
    return values.astype("string").fillna("").str.strip()


def case_id_issue_counts(values: pd.Series) -> tuple[int, int]:
    """Return missing IDs and the number of rows sharing a nonempty ID."""
    ids = normalized_case_ids(values)
    missing = ids.eq("")
    duplicates = ids.duplicated(keep=False) & ~missing
    return int(missing.sum()), int(duplicates.sum())


def prepare_unique_case_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Assign IDs to ambiguous rows, keeping every row and the original source IDs.

    Existing unique IDs are retained. Generated IDs are deterministic for the
    uploaded row order and cannot collide with any ID already in the file.
    """
    if "case_id" not in df.columns:
        raise ValueError("The case_id column is required before assigning row IDs.")
    ids = normalized_case_ids(df["case_id"])
    needs_id = ids.eq("") | ids.duplicated(keep=False)
    changed = int(needs_id.sum())
    repaired = df.copy(deep=True)
    if not changed:
        return repaired, 0

    if "source_case_id" not in repaired.columns:
        repaired["source_case_id"] = df["case_id"].copy()
    else:
        # Preserve an earlier source reference and the current input IDs as well.
        backup = "input_case_id"
        suffix = 2
        while backup in repaired.columns:
            backup = f"input_case_id_{suffix}"
            suffix += 1
        repaired[backup] = df["case_id"].copy()
    occupied = set(ids[ids.ne("")])
    assigned = ids.tolist()
    for position, needs_change in enumerate(needs_id.tolist()):
        if not needs_change:
            continue
        base = f"record-{position + 1:06d}"
        candidate = base
        suffix = 2
        while candidate in occupied:
            candidate = f"{base}-{suffix}"
            suffix += 1
        assigned[position] = candidate
        occupied.add(candidate)
    repaired["case_id"] = pd.array(assigned, dtype="string")
    return repaired, changed
