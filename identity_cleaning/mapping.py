from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


IGNORE = "ignore"
OTHER = "other"

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


@dataclass(frozen=True)
class ColumnMapping:
    columns_to_fields: Mapping[str, str]

    def source_for(self, field: str) -> str | None:
        for column, mapped_field in self.columns_to_fields.items():
            if mapped_field == field:
                return column
        return None

    def require_unique_fields(self) -> None:
        seen: dict[str, str] = {}
        for column, field in self.columns_to_fields.items():
            if field in {IGNORE, OTHER}:
                continue
            if field in seen:
                raise ValueError(
                    f"Le champ '{FIELD_LABELS.get(field, field)}' est selectionne plusieurs fois "
                    f"({seen[field]} et {column})."
                )
            seen[field] = column


def validate_required_fields(mapping: ColumnMapping, required_fields: list[str]) -> None:
    mapping.require_unique_fields()
    missing = [field for field in required_fields if mapping.source_for(field) is None]
    if missing:
        labels = ", ".join(FIELD_LABELS.get(field, field) for field in missing)
        raise ValueError(f"Colonnes obligatoires manquantes : {labels}.")


def read_mapped_value(row: pd.Series, mapping: ColumnMapping, field: str) -> object:
    source = mapping.source_for(field)
    if source is None:
        return None
    return row.get(source)

