from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from data_cleaning.cities import build_city_references, resolve_city
from data_cleaning.io import read_table
from data_cleaning.mapping import (
    CITY_FIELDS,
    IGNORE,
    PERSON_FIELDS,
    REPEATABLE_FIELDS,
    ColumnMapping,
    field_base,
    field_index,
    repeatable_field,
)
from data_cleaning.pipeline import HumanDecision, build_clean_database, quality_summary

st.set_page_config(page_title="Identity Matching Preparation", layout="wide")


def install_scroll_fallback() -> None:
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .stMain,
        section.main, [data-testid="stVerticalBlock"], [data-baseweb="popover"], [role="listbox"] {
            overscroll-behavior-y: contain !important;
        }
        [data-testid="stElementContainer"], [data-testid="stVerticalBlock"] {
            overflow: visible !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    components.html(
        """
        <script>
        (() => {
            try {
                const doc = window.parent.document;
                if (doc.__identityScrollFallbackInstalled) return;
                doc.__identityScrollFallbackInstalled = true;

                const canScrollInternally = (element, deltaY) => {
                    if (!element || element === doc.body || element === doc.documentElement) return false;
                    const style = window.parent.getComputedStyle(element);
                    const scrollable = /(auto|scroll)/.test(style.overflowY || "");
                    if (!scrollable || element.scrollHeight <= element.clientHeight + 1) return false;
                    if (deltaY > 0) return element.scrollTop + element.clientHeight < element.scrollHeight - 1;
                    if (deltaY < 0) return element.scrollTop > 1;
                    return false;
                };

                const findAppScroller = () => {
                    const candidates = [
                        ...doc.querySelectorAll('[data-testid="stAppViewContainer"], [data-testid="stMain"], .stMain, section.main, main, body, html')
                    ];
                    let best = doc.scrollingElement || doc.documentElement;
                    for (const element of candidates) {
                        if (!element) continue;
                        const scrollableDistance = element.scrollHeight - element.clientHeight;
                        const bestDistance = best.scrollHeight - best.clientHeight;
                        if (scrollableDistance > bestDistance) best = element;
                    }
                    return best;
                };

                doc.addEventListener("wheel", (event) => {
                    if (Math.abs(event.deltaY) < Math.abs(event.deltaX)) return;

                    const listbox = event.target.closest('[role="listbox"], [data-baseweb="popover"]');
                    if (listbox && canScrollInternally(listbox, event.deltaY)) return;

                    let element = event.target;
                    while (element && element !== doc.body) {
                        if (canScrollInternally(element, event.deltaY)) return;
                        element = element.parentElement;
                    }

                    const appScroller = findAppScroller();
                    const beforeElement = appScroller.scrollTop;
                    const beforeWindow = window.parent.scrollY;

                    appScroller.scrollTop += event.deltaY;
                    window.parent.scrollBy(0, event.deltaY);

                    if (appScroller.scrollTop !== beforeElement || window.parent.scrollY !== beforeWindow) {
                        event.preventDefault();
                        event.stopPropagation();
                    }
                }, { capture: true, passive: false });
            } catch (error) {
                // If the browser blocks parent access, Streamlit's default scroll behavior remains.
            }
        })();
        </script>
        """,
        height=0,
    )


install_scroll_fallback()


SCROLL_ANCHOR_KEY = "scroll_to_anchor_id"


def request_scroll_to_anchor(anchor_id: str) -> None:
    st.session_state[SCROLL_ANCHOR_KEY] = anchor_id


def emit_scroll_to_anchor(anchor_id: str) -> None:
    components.html(
        f"""
        <script>
        (() => {{
            const scrollToAnchor = () => {{
                const doc = window.parent.document;
                const target = doc.getElementById("{anchor_id}");
                if (!target) return;
                target.scrollIntoView({{ behavior: "auto", block: "center" }});
            }};
            setTimeout(scrollToAnchor, 50);
            setTimeout(scrollToAnchor, 250);
            setTimeout(scrollToAnchor, 600);
        }})();
        </script>
        """,
        height=0,
    )


def scroll_to_anchor_if_requested(anchor_id: str) -> None:
    if st.session_state.get(SCROLL_ANCHOR_KEY) != anchor_id:
        return
    st.session_state.pop(SCROLL_ANCHOR_KEY, None)
    emit_scroll_to_anchor(anchor_id)


def scroll_anchor(anchor_id: str) -> None:
    st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)
    scroll_to_anchor_if_requested(anchor_id)


def anchored_button(label: str, anchor_id: str, **kwargs) -> bool:
    scroll_anchor(anchor_id)
    clicked = st.button(label, **kwargs)
    if clicked:
        request_scroll_to_anchor(anchor_id)
        emit_scroll_to_anchor(anchor_id)
    return clicked


TEXT = {
    "title": "Prepare the Database for Identity Matching",
    "people_block_title": "Missing Persons Database",
    "people_upload_instruction": "Upload the missing persons database, then check that each column is mapped to the right case field.",
    "people_mapping_instruction": "Review the proposed mapping and adjust any incorrect column before validating this block.",
    "mapping_caption": (
        "Map each column to the corresponding field. Each field can be used only once, "
        "except for ignore and other useful column. First name and last name can be selected multiple times."
    ),
    "duplicate_fields": "Field selected more than once: {fields}",
    "missing_required": "Missing required columns: {fields}",
    "validate_people_block": "Validate missing persons import and mapping",
    "people_block_validated": "Missing persons database and mapping validated.",
    "people_header": "1. Import the Missing Persons Database",
    "people_file": "Missing persons database",
    "people_mapping_header": "2. Map case columns",
    "people_read_error": "Unable to read the case database: {error}",
    "city_block_title": "City Reference Database",
    "city_upload_instruction": "Upload the city reference database or load the local file, then map the city, department and coordinate columns.",
    "city_mapping_instruction": "Review the city reference mapping. City, department, latitude and longitude are required.",
    "validate_city_block": "Validate city reference import and mapping",
    "city_block_validated": "City reference database and mapping validated.",
    "cities_header": "3. Import or Load the City Reference Database",
    "cities_file": "City reference database",
    "city_mapping_header": "4. Map city database columns",
    "cities_read_error": "Unable to read the city database: {error}",
    "build_block_title": "Enriched Database Build",
    "build_instruction": "Start the build once both database blocks are validated. This creates the cleaned database used for conflict resolution.",
    "build_header": "5. Build the Enriched Database",
    "build_info": "The build adds coordinates only from the city reference database.",
    "build_button": "Start building the enriched database",
    "build_prepare": "Preparing the build...",
    "build_refs": "Loading and normalizing the city reference database...",
    "people_to_process": "cases to process",
    "usable_reference_cities": "usable reference cities",
    "build_cleaning": "City database ready. Cleaning cases...",
    "build_progress": "Build in progress: case {done}/{total} processed",
    "last_case": "Last processed case: {case_id}",
    "missing_id": "missing id",
    "build_done": "Build complete.",
    "build_success": "Enriched database built. Conflicts can now be resolved.",
    "processed_cases": "processed cases",
    "verified_cities": "verified cities",
    "unresolved_cities": "conflicts or unresolved cities",
    "build_error": "Build failed: {error}",
    "build_block_validated": "Enriched database built.",
    "conflicts_block_title": "City Conflict Resolution",
    "conflicts_instruction": "Review the current case and choose the best city resolution before moving to the next conflict.",
    "conflicts_header": "6. Resolve City Conflicts",
    "cases_to_resolve": "Cases to resolve",
    "no_city_conflicts": "No city conflicts to resolve.",
    "conflict_case": "Conflict case",
    "case_id": "case id",
    "original_city": "original city",
    "original_department": "original department",
    "status": "status",
    "count": "count",
    "score": "score",
    "explanation": "explanation",
    "bulk_caption": "This decision will be applied to {count} case(s) with the same original city and department.",
    "candidate_to_accept": "Candidate to accept",
    "candidate_details": "Candidate details",
    "hide_candidate_details": "Hide candidate details",
    "accept_candidate": "Accept this candidate",
    "candidate_accepted": "Candidate accepted for {count} case(s) with the same original city and department.",
    "correct_city": "Correct the city",
    "search_correction": "Search",
    "corrected_candidate": "Corrected city candidate",
    "apply_correction": "Apply this correction",
    "correction_selected": "Correction applied for {count} case(s) with the same original city and department.",
    "set_city_empty": "Treat this city as empty",
    "city_empty": "City treated as empty for {count} case(s) with the same original city and department.",
    "quality_header": "7. Quality Summary",
    "indicator": "indicator",
    "value": "value",
    "export_block_title": "Final Review and Export",
    "export_instruction": "Review the quality summary, then export the final database.",
    "export_header": "8. Export the Final CSV",
    "export_final": "Export final database",
    "reviewed_case": "Case under review",
}

FIELD_LABELS_EN = {
    "case_id": "case id",
    "full_name": "full name",
    "first_name": "first name",
    "last_name": "last name",
    "disappearance_city": "disappearance city",
    "disappearance_department": "department / province / region",
    "disappearance_date": "disappearance date",
    "country": "country",
    "city": "city",
    "department": "department / province / region",
    "latitude": "latitude",
    "longitude": "longitude",
    "other": "other useful column",
    "ignore": "ignore",
}

STATUS_LABELS = {
    "missing_city": "missing city",
    "unresolved_city": "unresolved city",
    "fuzzy_city_candidates": "fuzzy candidates",
    "ambiguous_city_match": "ambiguous city",
    "ambiguous_city_department_match": "ambiguous city/department",
    "city_match_department_mismatch": "department mismatch",
    "human_validated_city": "validated city",
    "human_selected_city": "selected city",
    "human_marked_city_empty": "city marked empty",
}

SUMMARY_LABELS = {
    "nombre total de dossiers": "total cases",
    "nombre de dossiers avec ville verifiee": "cases with verified city",
    "nombre de dossiers avec latitude/longitude": "cases with latitude/longitude",
    "nombre de dossiers sans ville": "cases without a city",
    "nombre de dossiers non resolus": "unresolved cases",
    "nombre de dossiers corriges par un humain": "cases corrected by a human",
    "nombre de dates correctement parsees": "correctly parsed dates",
    "nombre de dates invalides ou absentes": "invalid or missing dates",
    "pourcentage exploitable pour la similarite geographique": "usable percentage for geographic similarity",
    "pourcentage exploitable pour la similarite temporelle": "usable percentage for temporal similarity",
}


def t(key: str, **kwargs) -> str:
    text = TEXT.get(key, key)
    return text.format(**kwargs) if kwargs else text


def field_label(field: str) -> str:
    base = field_base(field)
    index = field_index(field)
    label = FIELD_LABELS_EN.get(base, base)
    return label if index is None else f"{label} {index}"


def display_status(status: str) -> str:
    return STATUS_LABELS.get(str(status), str(status))


def translate_summary_key(key: str) -> str:
    return SUMMARY_LABELS.get(key, key)


@st.cache_data(show_spinner=False)
def read_uploaded_table(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    buffer = BytesIO(file_bytes)
    buffer.name = file_name
    return read_table(buffer)


@st.cache_data(show_spinner=False)
def read_local_table(path: str, mtime: float) -> pd.DataFrame:
    del mtime
    return read_table(path)


@st.cache_data(show_spinner=False)
def build_city_references_cached(
    df: pd.DataFrame,
    city_col: str,
    department_col: str,
    latitude_col: str,
    longitude_col: str,
    country_col: str | None,
    scoring_version: str,
):
    del scoring_version
    return build_city_references(df, city_col, department_col, latitude_col, longitude_col, country_col)


def show_error(message: str) -> None:
    st.error(message)


def show_table(df: pd.DataFrame, max_rows: int | None = None) -> None:
    display_df = df.head(max_rows) if max_rows is not None else df
    st.table(display_df)


def mapping_editor(
    df: pd.DataFrame, fields: list[str], key_prefix: str, columns_per_row: int = 2
) -> ColumnMapping:
    guessed = guess_mapping(df.columns, fields)
    st.caption(t("mapping_caption"))
    mapping_values = current_mapping_values(df.columns, fields, key_prefix, guessed)
    mapping: dict[str, str] = {}
    columns_per_row = max(1, columns_per_row)
    for start in range(0, len(df.columns), columns_per_row):
        row_columns = list(df.columns[start : start + columns_per_row])
        layout_columns = st.columns(len(row_columns))
        for layout_column, source_column in zip(layout_columns, row_columns):
            column = str(source_column)
            options = options_for_column(fields, mapping_values, column)
            default = mapping_values.get(column, IGNORE)
            index = options.index(default) if default in options else options.index(IGNORE)
            with layout_column:
                selected = st.selectbox(
                    column,
                    options,
                    index=index,
                    format_func=field_label,
                    key=f"{key_prefix}_{column}",
                )
            mapping[column] = selected
    result = ColumnMapping(mapping)
    duplicates = duplicated_fields(result)
    if duplicates:
        st.warning(t("duplicate_fields", fields=", ".join(field_label(field) for field in duplicates)))
    return result


def current_mapping_values(
    columns, fields: list[str], key_prefix: str, guessed: dict[str, str]
) -> dict[str, str]:
    values: dict[str, str] = {}
    repeatable_counts: dict[str, int] = {}
    for column in columns:
        column_key = str(column)
        state_key = f"{key_prefix}_{column}"
        value = st.session_state.get(state_key, guessed.get(column, IGNORE))
        base = field_base(value)
        if base in REPEATABLE_FIELDS and base in fields:
            repeatable_counts[base] = repeatable_counts.get(base, 0) + 1
            value = repeatable_field(base, repeatable_counts[base])
        value = value if value in fields or field_base(value) in fields else IGNORE
        if state_key in st.session_state and st.session_state[state_key] != value:
            st.session_state[state_key] = value
        values[column_key] = value
    return values


def options_for_column(fields: list[str], mapping_values: dict[str, str], column: str) -> list[str]:
    current_value = mapping_values.get(column, IGNORE)
    options: list[str] = []
    for field in fields:
        if field in REPEATABLE_FIELDS:
            current_base = field_base(current_value)
            if current_base == field and current_value not in options:
                options.append(current_value)
            used_indices = [
                field_index(value) or 1
                for other_column, value in mapping_values.items()
                if other_column != column and field_base(value) == field
            ]
            next_value = repeatable_field(field, max(used_indices, default=0) + 1)
            if next_value not in options:
                options.append(next_value)
            continue
        options.append(field)
    return options


def duplicated_fields(mapping: ColumnMapping) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for field in mapping.columns_to_fields.values():
        if field in {IGNORE, "other"}:
            continue
        base = field_base(field)
        if base in REPEATABLE_FIELDS:
            continue
        if base in seen and base not in duplicates:
            duplicates.append(base)
        seen.add(base)
    return duplicates


def guess_mapping(columns, fields: list[str]) -> dict[str, str]:
    guesses: dict[str, str] = {}
    for column in columns:
        normalized = str(column).lower().replace("\ufeff", "")
        if "id_dedup" in normalized or normalized == "id" or "case" in normalized:
            guesses[column] = "case_id"
        elif "nombre_completo" in normalized or "full" in normalized:
            guesses[column] = "full_name"
        elif "primer_nombre" in normalized or "first" in normalized:
            guesses[column] = "first_name"
        elif "apellido" in normalized or "last" in normalized:
            guesses[column] = "last_name"
        elif "municipio" in normalized or "city" in normalized or "ville" in normalized:
            guesses[column] = "city" if "city" in fields else "disappearance_city"
        elif (
            "depto" in normalized
            or "departamento" in normalized
            or "province" in normalized
            or "region" in normalized
        ):
            guesses[column] = "department" if "department" in fields else "disappearance_department"
        elif "fecha" in normalized or "date" in normalized:
            guesses[column] = "disappearance_date"
        elif "lat" in normalized:
            guesses[column] = "latitude"
        elif "lon" in normalized:
            guesses[column] = "longitude"
        elif "pais" in normalized or "country" in normalized:
            guesses[column] = "country"
    return {column: field for column, field in guesses.items() if field in fields}


def mapping_is_valid(mapping: ColumnMapping, required: list[str]) -> bool:
    duplicates = duplicated_fields(mapping)
    if duplicates:
        show_error(t("duplicate_fields", fields=", ".join(field_label(field) for field in duplicates)))
        return False
    try:
        mapping.require_unique_fields()
    except ValueError as exc:
        show_error(str(exc))
        return False
    missing = [field for field in required if mapping.source_for(field) is None]
    if missing:
        show_error(t("missing_required", fields=", ".join(field_label(field) for field in missing)))
        return False
    return True


def mapping_is_complete(mapping: ColumnMapping | None, required: list[str]) -> bool:
    if mapping is None or duplicated_fields(mapping):
        return False
    try:
        mapping.require_unique_fields()
    except ValueError:
        return False
    return all(mapping.source_for(field) is not None for field in required)


def mapping_signature(mapping: ColumnMapping | None) -> tuple[tuple[str, str], ...] | None:
    if mapping is None:
        return None
    return tuple(sorted((str(column), str(field)) for column, field in mapping.columns_to_fields.items()))


def block_signature(
    df: pd.DataFrame | None, source_name: str, mapping: ColumnMapping | None
) -> tuple[tuple[str, str], ...] | None:
    mapping_part = mapping_signature(mapping)
    if df is None or mapping_part is None:
        return None
    source_part = (
        ("__source__", source_name),
        ("__rows__", str(len(df))),
        ("__columns__", "|".join(str(column) for column in df.columns)),
    )
    return source_part + mapping_part


def clear_build_state() -> None:
    for key in ["clean_df", "city_refs", "human_decisions", "last_bulk_validation_message"]:
        st.session_state.pop(key, None)


def clear_city_block_state() -> None:
    st.session_state["city_block_validated"] = False
    st.session_state.pop("city_block_signature", None)
    clear_build_state()


def candidate_table(candidates):
    return pd.DataFrame(
        [
            {
                "reference_id": candidate.reference.ref_id,
                field_label("city"): candidate.reference.city_raw,
                field_label("department"): candidate.reference.department_raw,
                "city_and_department": f"{candidate.reference.city_raw} - {candidate.reference.department_raw}",
                field_label("latitude"): candidate.reference.latitude,
                field_label("longitude"): candidate.reference.longitude,
                t("score"): candidate.score,
                t("explanation"): candidate.explanation,
            }
            for candidate in candidates
        ]
    )


def apply_city_decision_to_clean_df(
    clean_df: pd.DataFrame,
    case_ids: list[str],
    reference_id: str | None,
    refs,
    status: str,
    decision_label: str,
) -> pd.DataFrame:
    updated = clean_df.copy()
    normalized_case_ids = {str(case_id) for case_id in case_ids}
    mask = updated["case_id"].astype(str).isin(normalized_case_ids)
    ref = next((city_ref for city_ref in refs if str(city_ref.ref_id) == str(reference_id)), None)
    if ref is None and reference_id is not None:
        return updated

    updated.loc[mask, "city_resolution_status"] = status
    updated.loc[mask, "human_validation_decision"] = decision_label
    if ref is None:
        updated.loc[mask, "disappearance_city_verified"] = ""
        updated.loc[mask, "disappearance_department_verified"] = ""
        updated.loc[mask, "latitude"] = ""
        updated.loc[mask, "longitude"] = ""
        updated.loc[mask, "city_resolution_score"] = ""
        updated.loc[mask, "city_resolution_explanation"] = "Human validation: city treated as empty."
        return updated

    updated.loc[mask, "disappearance_city_verified"] = ref.city_raw
    updated.loc[mask, "disappearance_department_verified"] = ref.department_raw
    updated.loc[mask, "latitude"] = ref.latitude
    updated.loc[mask, "longitude"] = ref.longitude
    updated.loc[mask, "city_resolution_score"] = 100.0
    updated.loc[mask, "city_resolution_explanation"] = f"Human validation: city {ref.city_raw} selected."
    return updated


def matching_input_case_ids(clean_df: pd.DataFrame, source_case_id: str) -> list[str]:
    source_rows = clean_df[clean_df["case_id"].astype(str) == str(source_case_id)]
    if source_rows.empty:
        return [str(source_case_id)]
    source = source_rows.iloc[0]
    same_input = clean_df["disappearance_city_normalized"].astype(str).eq(
        str(source["disappearance_city_normalized"])
    ) & clean_df["disappearance_department_normalized"].astype(str).eq(
        str(source["disappearance_department_normalized"])
    )
    return clean_df.loc[same_input, "case_id"].astype(str).tolist()


st.title(t("title"))

persons_df = None
person_mapping = None
person_mapping_ready = False
person_block_validated = st.session_state.get("person_block_validated", False)
with st.expander(t("people_block_title"), expanded=not person_block_validated):
    st.caption(t("people_upload_instruction"))
    person_file = st.file_uploader(t("people_file"), type=["csv", "xls", "xlsx"], key="person_file")
    if person_file:
        try:
            persons_df = read_uploaded_table(person_file.name, person_file.getvalue())
            show_table(persons_df, max_rows=10)
            st.caption(t("people_mapping_instruction"))
            person_mapping = mapping_editor(persons_df, PERSON_FIELDS, "person_map", columns_per_row=2)
            person_mapping_ready = mapping_is_complete(person_mapping, ["case_id"])
            person_signature = block_signature(persons_df, person_file.name, person_mapping)
            if st.session_state.get("person_block_signature") != person_signature:
                if st.session_state.get("person_block_validated"):
                    clear_city_block_state()
                st.session_state["person_block_validated"] = False
                person_block_validated = False
            if person_mapping_ready and anchored_button(
                t("validate_people_block"), "validate-people-block-button", type="primary"
            ):
                st.session_state["person_block_validated"] = True
                st.session_state["person_block_signature"] = person_signature
                st.rerun()
        except Exception as exc:
            show_error(t("people_read_error", error=exc))

person_block_validated = st.session_state.get("person_block_validated", False)
if person_block_validated:
    st.success(t("people_block_validated"))

cities_df = None
city_mapping = None
city_mapping_ready = False
city_block_validated = st.session_state.get("city_block_validated", False)
if person_block_validated:
    with st.expander(t("city_block_title"), expanded=not city_block_validated):
        st.caption(t("city_upload_instruction"))
        city_file = st.file_uploader(t("cities_file"), type=["csv", "xls", "xlsx"], key="city_file")

        if city_file:
            try:
                cities_df = read_uploaded_table(city_file.name, city_file.getvalue())
                city_source_name = city_file.name
                show_table(cities_df, max_rows=10)
                st.caption(t("city_mapping_instruction"))
                city_mapping = mapping_editor(cities_df, CITY_FIELDS, "city_map", columns_per_row=3)
                city_mapping_ready = mapping_is_complete(
                    city_mapping, ["city", "department", "latitude", "longitude"]
                )
                city_signature = block_signature(cities_df, city_source_name, city_mapping)
                if st.session_state.get("city_block_signature") != city_signature:
                    if st.session_state.get("city_block_validated"):
                        clear_build_state()
                    st.session_state["city_block_validated"] = False
                    city_block_validated = False
                if city_mapping_ready and anchored_button(
                    t("validate_city_block"), "validate-city-block-button", type="primary"
                ):
                    st.session_state["city_block_validated"] = True
                    st.session_state["city_block_signature"] = city_signature
                    st.rerun()
            except Exception as exc:
                show_error(t("cities_read_error", error=exc))

city_block_validated = st.session_state.get("city_block_validated", False)
if city_block_validated:
    st.success(t("city_block_validated"))

can_build = (
    persons_df is not None
    and person_mapping is not None
    and person_block_validated
    and cities_df is not None
    and city_mapping is not None
    and city_block_validated
)
if can_build:
    with st.expander(t("build_block_title"), expanded="clean_df" not in st.session_state):
        st.caption(t("build_instruction"))
        st.info(t("build_info"))
        if anchored_button(t("build_button"), "build-database-button", type="primary"):
            if mapping_is_valid(person_mapping, ["case_id"]) and mapping_is_valid(
                city_mapping, ["city", "department", "latitude", "longitude"]
            ):
                progress_bar = st.progress(0, text=t("build_prepare"))
                status_box = st.empty()
                metrics_box = st.empty()
                try:
                    total_persons = len(persons_df)
                    status_box.info(t("build_refs"))
                    refs = build_city_references_cached(
                        cities_df,
                        city_mapping.source_for("city"),
                        city_mapping.source_for("department"),
                        city_mapping.source_for("latitude"),
                        city_mapping.source_for("longitude"),
                        city_mapping.source_for("country"),
                        "generic_abbreviation_scoring_v1",
                    )
                    metrics_box.write(
                        {
                            t("people_to_process"): total_persons,
                            t("usable_reference_cities"): len(refs),
                        }
                    )
                    progress_bar.progress(5, text=t("build_cleaning"))

                    def update_build_progress(done: int, total: int, case_id: str) -> None:
                        percent = 5 + int((done / total) * 95) if total else 100
                        progress_bar.progress(
                            min(percent, 100),
                            text=t("build_progress", done=done, total=total),
                        )
                        status_box.info(t("last_case", case_id=case_id or t("missing_id")))

                    clean_df = build_clean_database(
                        persons_df,
                        person_mapping,
                        refs,
                        st.session_state.get("human_decisions", {}),
                        progress_callback=update_build_progress,
                    )
                    st.session_state["city_refs"] = refs
                    st.session_state["clean_df"] = clean_df
                    summary = quality_summary(clean_df)
                    progress_bar.progress(100, text=t("build_done"))
                    status_box.success(t("build_success"))
                    metrics_box.write(
                        {
                            t("processed_cases"): summary["nombre total de dossiers"],
                            t("verified_cities"): summary["nombre de dossiers avec ville verifiee"],
                            t("unresolved_cities"): summary["nombre de dossiers non resolus"],
                        }
                    )
                    st.rerun()
                except Exception as exc:
                    progress_bar.empty()
                    show_error(t("build_error", error=exc))

if "clean_df" in st.session_state:
    st.success(t("build_block_validated"))

if "clean_df" in st.session_state:
    clean_df = st.session_state["clean_df"]
    refs = st.session_state["city_refs"]
    conflict_statuses = {
        "unresolved_city",
        "fuzzy_city_candidates",
        "ambiguous_city_match",
        "ambiguous_city_department_match",
        "city_match_department_mismatch",
    }
    conflicts = clean_df[clean_df["city_resolution_status"].isin(conflict_statuses)]
    if st.session_state.get("last_bulk_validation_message"):
        st.success(st.session_state.pop("last_bulk_validation_message"))

    with st.expander(t("conflicts_block_title"), expanded=not conflicts.empty):
        if conflicts.empty:
            st.success(t("no_city_conflicts"))
        else:
            left, center, right = st.columns([1, 2.4, 1])
            with center:
                st.caption(t("conflicts_instruction"))
                status_counts = (
                    conflicts["city_resolution_status"]
                    .map(display_status)
                    .value_counts()
                    .rename_axis(t("status"))
                    .reset_index(name=t("count"))
                )
                show_table(status_counts)

                selected_case = st.selectbox(t("conflict_case"), conflicts["case_id"].tolist())
                row = conflicts[conflicts["case_id"] == selected_case].iloc[0]
                st.markdown(f"### {t('reviewed_case')}: `{row['case_id']}`")
                info_cols = st.columns(2)
                info_cols[0].metric(t("original_city"), row["disappearance_city_raw"] or "-")
                info_cols[1].metric(t("original_department"), row["disappearance_department_raw"] or "-")
                st.caption(row["city_resolution_explanation"])

                resolution = resolve_city(
                    row["disappearance_city_raw"], row["disappearance_department_raw"], refs
                )
                candidates_df = candidate_table(resolution.candidates)
                matching_case_ids = matching_input_case_ids(st.session_state["clean_df"], str(selected_case))

                if not candidates_df.empty:
                    details_key = f"show_candidate_details_{selected_case}"
                    details_visible = st.session_state.get(details_key, False)
                    details_label = t("hide_candidate_details") if details_visible else t("candidate_details")
                    if anchored_button(
                        details_label, f"candidate-details-button-{selected_case}", use_container_width=True
                    ):
                        details_visible = not details_visible
                        st.session_state[details_key] = details_visible
                        request_scroll_to_anchor(f"candidate-details-button-{selected_case}")
                        st.rerun()
                    if details_visible:
                        show_table(candidates_df)
                    st.caption(t("bulk_caption", count=len(matching_case_ids)))
                    scroll_anchor("conflict-action-button")
                    chosen = st.selectbox(
                        t("candidate_to_accept"),
                        candidates_df["reference_id"].tolist(),
                        format_func=lambda ref_id: candidates_df[
                            candidates_df["reference_id"] == ref_id
                        ].iloc[0]["city_and_department"],
                    )
                    if anchored_button(
                        t("accept_candidate"),
                        "accept-candidate-button",
                        type="primary",
                        use_container_width=True,
                    ):
                        decisions = st.session_state.setdefault("human_decisions", {})
                        for case_id in matching_case_ids:
                            decisions[str(case_id)] = HumanDecision(
                                "accept_candidate", reference_id=str(chosen)
                            )
                        st.session_state["clean_df"] = apply_city_decision_to_clean_df(
                            st.session_state["clean_df"],
                            matching_case_ids,
                            str(chosen),
                            refs,
                            "human_validated_city",
                            "accept_candidate",
                        )
                        st.session_state["last_bulk_validation_message"] = t(
                            "candidate_accepted", count=len(matching_case_ids)
                        )
                        request_scroll_to_anchor("conflict-action-button")
                        st.rerun()

                if anchored_button(t("set_city_empty"), "set-city-empty-button", use_container_width=True):
                    matching_case_ids = matching_input_case_ids(
                        st.session_state["clean_df"], str(selected_case)
                    )
                    decisions = st.session_state.setdefault("human_decisions", {})
                    for case_id in matching_case_ids:
                        decisions[str(case_id)] = HumanDecision("set_empty")
                    st.session_state["clean_df"] = apply_city_decision_to_clean_df(
                        st.session_state["clean_df"],
                        matching_case_ids,
                        None,
                        refs,
                        "human_marked_city_empty",
                        "set_empty",
                    )
                    st.session_state["last_bulk_validation_message"] = t(
                        "city_empty", count=len(matching_case_ids)
                    )
                    request_scroll_to_anchor("conflict-action-button")
                    st.rerun()

                st.markdown(f"#### {t('correct_city')}")
                corrected_city = st.text_input(t("correct_city"), key=f"corrected_city_{selected_case}")
                if anchored_button(
                    t("search_correction"),
                    f"search-correction-button-{selected_case}",
                    use_container_width=True,
                ):
                    st.session_state[f"searched_corrected_city_{selected_case}"] = corrected_city.strip()
                searched_corrected_city = st.session_state.get(f"searched_corrected_city_{selected_case}", "")
                if searched_corrected_city:
                    new_resolution = resolve_city(
                        searched_corrected_city, row["disappearance_department_raw"], refs
                    )
                    correction_df = candidate_table(new_resolution.candidates)
                    show_table(correction_df)
                    if not correction_df.empty:
                        selected_ref = st.selectbox(
                            t("corrected_candidate"),
                            correction_df["reference_id"].tolist(),
                            format_func=lambda ref_id: correction_df[
                                correction_df["reference_id"] == ref_id
                            ].iloc[0]["city_and_department"],
                        )
                        if anchored_button(
                            t("apply_correction"), "apply-correction-button", use_container_width=True
                        ):
                            matching_case_ids = matching_input_case_ids(
                                st.session_state["clean_df"], str(selected_case)
                            )
                            decisions = st.session_state.setdefault("human_decisions", {})
                            for case_id in matching_case_ids:
                                decisions[str(case_id)] = HumanDecision(
                                    "manual_reference", reference_id=str(selected_ref)
                                )
                            st.session_state["clean_df"] = apply_city_decision_to_clean_df(
                                st.session_state["clean_df"],
                                matching_case_ids,
                                str(selected_ref),
                                refs,
                                "human_selected_city",
                                "manual_reference",
                            )
                            st.session_state["last_bulk_validation_message"] = t(
                                "correction_selected", count=len(matching_case_ids)
                            )
                            request_scroll_to_anchor("conflict-action-button")
                            st.rerun()

    if conflicts.empty:
        with st.expander(t("export_block_title"), expanded=True):
            st.caption(t("export_instruction"))
            summary = quality_summary(st.session_state["clean_df"])
            show_table(
                pd.DataFrame(
                    [(translate_summary_key(key), value) for key, value in summary.items()],
                    columns=[t("indicator"), t("value")],
                )
            )

            final_df = st.session_state["clean_df"]
            final_csv = final_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                t("export_final"), data=final_csv, file_name="missing_persons_clean.csv", mime="text/csv"
            )
            show_table(final_df, max_rows=30)
