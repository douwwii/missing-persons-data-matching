from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import pandas as pd

from data_cleaning.cities import CityReference, CityResolution, resolve_city
from data_cleaning.mapping import (
    ColumnMapping,
    read_mapped_value,
    read_mapped_values,
    validate_required_fields,
)
from data_cleaning.normalization import (
    name_tokens,
    normalize_text,
    parse_disappearance_date,
    tokens_as_string,
)

FINAL_COLUMNS = [
    "case_id",
    "full_name_raw",
    "full_name_normalized",
    "first_name_raw",
    "first_name_normalized",
    "last_name_raw",
    "last_name_normalized",
    "name_tokens",
    "disappearance_city_raw",
    "disappearance_city_normalized",
    "disappearance_city_verified",
    "disappearance_department_raw",
    "disappearance_department_normalized",
    "disappearance_department_verified",
    "latitude",
    "longitude",
    "disappearance_date_raw",
    "disappearance_date_parsed",
    "city_resolution_status",
    "city_resolution_score",
    "city_resolution_explanation",
    "human_validation_decision",
]


@dataclass(frozen=True)
class HumanDecision:
    action: str
    reference_id: str | None = None
    corrected_city: str | None = None
    note: str = ""


def build_full_name(first_name: object, last_name: object) -> str:
    parts = [str(part).strip() for part in [first_name, last_name] if part is not None and str(part).strip()]
    return " ".join(parts)


def _format_raw(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _format_joined(values: list[object]) -> str:
    parts = [_format_raw(value) for value in values]
    return " ".join(part for part in parts if part)


def _apply_human_decision(
    decision: HumanDecision,
    resolution: CityResolution,
    references_by_id: Mapping[str, CityReference],
) -> tuple[CityReference | None, str, str, float | None, str]:
    if decision.action == "accept_candidate" and decision.reference_id:
        ref = references_by_id.get(decision.reference_id)
        if ref:
            return (
                ref,
                "human_validated_city",
                f"Validation humaine: candidat {ref.city_raw} accepte.",
                100.0,
                "accept_candidate",
            )
    if decision.action == "manual_reference" and decision.reference_id:
        ref = references_by_id.get(decision.reference_id)
        if ref:
            return (
                ref,
                "human_selected_city",
                f"Validation humaine: ville {ref.city_raw} selectionnee manuellement.",
                100.0,
                "manual_reference",
            )
    if decision.action == "set_empty":
        return (
            None,
            "human_marked_city_empty",
            "Validation humaine: ville consideree comme vide.",
            None,
            "set_empty",
        )
    return resolution.selected, resolution.status, resolution.explanation, resolution.score, ""


def build_clean_database(
    persons_df: pd.DataFrame,
    person_mapping: ColumnMapping,
    city_references: list[CityReference],
    human_decisions: Mapping[str, HumanDecision] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    validate_required_fields(person_mapping, ["case_id"])
    references_by_id = {ref.ref_id: ref for ref in city_references}
    human_decisions = human_decisions or {}
    rows: list[dict[str, object]] = []
    total = len(persons_df)

    for position, (_, source_row) in enumerate(persons_df.iterrows(), start=1):
        case_id = _format_raw(read_mapped_value(source_row, person_mapping, "case_id"))
        if progress_callback and (position == 1 or position == total or position % 25 == 0):
            progress_callback(position, total, case_id)
        first_name_raw = _format_joined(read_mapped_values(source_row, person_mapping, "first_name"))
        last_name_raw = _format_joined(read_mapped_values(source_row, person_mapping, "last_name"))
        full_name_raw = _format_raw(read_mapped_value(source_row, person_mapping, "full_name"))
        if not full_name_raw:
            full_name_raw = build_full_name(first_name_raw, last_name_raw)

        city_raw = _format_raw(read_mapped_value(source_row, person_mapping, "disappearance_city"))
        department_raw = _format_raw(
            read_mapped_value(source_row, person_mapping, "disappearance_department")
        )
        country_raw = _format_raw(read_mapped_value(source_row, person_mapping, "country"))
        date_raw = _format_raw(read_mapped_value(source_row, person_mapping, "disappearance_date"))
        resolution = resolve_city(city_raw, department_raw, city_references, country_raw)

        selected = resolution.selected
        status = resolution.status
        explanation = resolution.explanation
        score = resolution.score
        human_validation = ""
        decision = human_decisions.get(case_id)
        if decision:
            selected, status, explanation, score, human_validation = _apply_human_decision(
                decision, resolution, references_by_id
            )

        rows.append(
            {
                "case_id": case_id,
                "full_name_raw": full_name_raw,
                "full_name_normalized": normalize_text(full_name_raw),
                "first_name_raw": first_name_raw,
                "first_name_normalized": normalize_text(first_name_raw),
                "last_name_raw": last_name_raw,
                "last_name_normalized": normalize_text(last_name_raw),
                "name_tokens": tokens_as_string(name_tokens(full_name_raw)),
                "disappearance_city_raw": city_raw,
                "disappearance_city_normalized": normalize_text(city_raw),
                "disappearance_city_verified": selected.city_raw if selected else "",
                "disappearance_department_raw": department_raw,
                "disappearance_department_normalized": normalize_text(department_raw),
                "disappearance_department_verified": selected.department_raw if selected else "",
                "latitude": selected.latitude if selected else "",
                "longitude": selected.longitude if selected else "",
                "disappearance_date_raw": date_raw,
                "disappearance_date_parsed": parse_disappearance_date(date_raw),
                "city_resolution_status": status,
                "city_resolution_score": "" if score is None else score,
                "city_resolution_explanation": explanation,
                "human_validation_decision": human_validation,
            }
        )
    return pd.DataFrame(rows, columns=FINAL_COLUMNS)


def quality_summary(clean_df: pd.DataFrame) -> dict[str, object]:
    total = len(clean_df)

    def present(column: str) -> pd.Series:
        values = clean_df[column]
        return values.notna() & values.astype(str).str.strip().ne("")

    latitude = pd.to_numeric(clean_df["latitude"], errors="coerce")
    longitude = pd.to_numeric(clean_df["longitude"], errors="coerce")
    with_coordinates = latitude.between(-90, 90) & longitude.between(-180, 180)
    city_verified = present("disappearance_city_verified")
    missing_city = clean_df["city_resolution_status"].eq("missing_city")
    unresolved = clean_df["city_resolution_status"].isin(
        [
            "unresolved_city",
            "fuzzy_city_candidates",
            "ambiguous_city_match",
            "ambiguous_city_department_match",
            "city_match_department_mismatch",
        ]
    )
    human = present("human_validation_decision")
    parsed_dates = present("disappearance_date_parsed")
    date_raw_present = present("disappearance_date_raw")
    invalid_or_absent_dates = ~parsed_dates

    def pct(mask: pd.Series) -> float:
        return round(float(mask.sum()) / total * 100, 2) if total else 0.0

    return {
        "nombre total de dossiers": total,
        "nombre de dossiers avec ville verifiee": int(city_verified.sum()),
        "nombre de dossiers avec latitude/longitude": int(with_coordinates.sum()),
        "nombre de dossiers sans ville": int(missing_city.sum()),
        "nombre de dossiers non resolus": int(unresolved.sum()),
        "nombre de dossiers corriges par un humain": int(human.sum()),
        "nombre de dates correctement parsees": int(parsed_dates.sum()),
        "nombre de dates invalides ou absentes": int(invalid_or_absent_dates.sum()),
        "pourcentage exploitable pour la similarite geographique": pct(with_coordinates),
        "pourcentage exploitable pour la similarite temporelle": pct(parsed_dates & date_raw_present),
    }
