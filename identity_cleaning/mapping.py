from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


IGNORE = "ignore"
OTHER = "other"
REPEATABLE_FIELDS = {"first_name", "last_name"}
REPEATABLE_SEPARATOR = "__"

PERSON_FIELDS = [
    "case_id",
    "full_name",
    "first_name",
    "last_name",
    "disappearance_city",
    "disappearance_department",
    "disappearance_date",
    "country",
    OTHER,
    IGNORE,
]

CITY_FIELDS = [
    "city",
    "department",
    "latitude",
    "longitude",
    "country",
    OTHER,
    IGNORE,
]


FIELD_LABELS = {
    "case_id": "id du dossier",
    "full_name": "nom complet",
    "first_name": "prenom",
    "last_name": "nom",
    "disappearance_city": "ville de disparition",
    "disappearance_department": "departement / province / region",
    "disappearance_date": "date de disparition",
    "country": "pays",
    "city": "ville",
    "department": "departement / province / region",
    "latitude": "latitude",
    "longitude": "longitude",
    OTHER: "autre colonne utile",
    IGNORE: "ignorer",
}


def repeatable_field(base_field: str, index: int) -> str:
    return f"{base_field}{REPEATABLE_SEPARATOR}{index}"


def field_base(field: str) -> str:
    if REPEATABLE_SEPARATOR not in field:
        return field
    base, suffix = field.rsplit(REPEATABLE_SEPARATOR, 1)
    if base in REPEATABLE_FIELDS and suffix.isdigit():
        return base
    return field


def field_index(field: str) -> int | None:
    if REPEATABLE_SEPARATOR not in field:
        return None
    base, suffix = field.rsplit(REPEATABLE_SEPARATOR, 1)
    if base in REPEATABLE_FIELDS and suffix.isdigit():
        return int(suffix)
    return None


def field_label(field: str) -> str:
    base = field_base(field)
    index = field_index(field)
    label = FIELD_LABELS.get(base, field)
    if index is None:
        return label
    return f"{label} {index}"


@dataclass(frozen=True)
class ColumnMapping:
    columns_to_fields: Mapping[str, str]

    def source_for(self, field: str) -> str | None:
        sources = self.sources_for(field)
        return sources[0] if sources else None

    def sources_for(self, field: str) -> list[str]:
        requested_base = field_base(field)
        sources: list[str] = []
        for column, mapped_field in self.columns_to_fields.items():
            if mapped_field == field:
                sources.append(column)
            elif requested_base in REPEATABLE_FIELDS and field_base(mapped_field) == requested_base:
                sources.append(column)
        return sources

    def require_unique_fields(self) -> None:
        seen: dict[str, str] = {}
        for column, field in self.columns_to_fields.items():
            if field in {IGNORE, OTHER}:
                continue
            if field_base(field) in REPEATABLE_FIELDS:
                continue
            if field in seen:
                raise ValueError(
                    f"Le champ '{field_label(field)}' est selectionne plusieurs fois "
                    f"({seen[field]} et {column})."
                )
            seen[field] = column


def validate_required_fields(mapping: ColumnMapping, required_fields: list[str]) -> None:
    mapping.require_unique_fields()
    missing = [field for field in required_fields if mapping.source_for(field) is None]
    if missing:
        labels = ", ".join(field_label(field) for field in missing)
        raise ValueError(f"Colonnes obligatoires manquantes : {labels}.")


def read_mapped_value(row: pd.Series, mapping: ColumnMapping, field: str) -> object:
    source = mapping.source_for(field)
    if source is None:
        return None
    return row.get(source)


def read_mapped_values(row: pd.Series, mapping: ColumnMapping, field: str) -> list[object]:
    return [row.get(source) for source in mapping.sources_for(field)]
