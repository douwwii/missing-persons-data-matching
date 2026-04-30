from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from identity_cleaning.cities import build_city_references, resolve_city
from identity_cleaning.io import read_table
from identity_cleaning.mapping import CITY_FIELDS, FIELD_LABELS, IGNORE, PERSON_FIELDS, ColumnMapping
from identity_cleaning.pipeline import HumanDecision, build_clean_database, quality_summary


st.set_page_config(page_title="Preparation identity matching", layout="wide")


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


def mapping_editor(df: pd.DataFrame, fields: list[str], key_prefix: str) -> ColumnMapping:
    options = fields
    guessed = guess_mapping(df.columns, fields)
    st.caption("Associez chaque colonne au champ metier correspondant. Un champ ne peut etre utilise qu'une fois, sauf ignorer et autre colonne utile.")
    mapping: dict[str, str] = {}
    for column in df.columns:
        default = guessed.get(column, IGNORE)
        index = options.index(default) if default in options else options.index(IGNORE)
        selected = st.selectbox(
            str(column),
            options,
            index=index,
            format_func=lambda value: FIELD_LABELS.get(value, value),
            key=f"{key_prefix}_{column}",
        )
        mapping[str(column)] = selected
    result = ColumnMapping(mapping)
    duplicates = duplicated_fields(result)
    if duplicates:
        st.warning("Champ selectionne plusieurs fois : " + ", ".join(FIELD_LABELS.get(field, field) for field in duplicates))
    return result


def duplicated_fields(mapping: ColumnMapping) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for field in mapping.columns_to_fields.values():
        if field in {IGNORE, "other"}:
            continue
        if field in seen and field not in duplicates:
            duplicates.append(field)
        seen.add(field)
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
        elif "depto" in normalized or "departamento" in normalized or "province" in normalized or "region" in normalized:
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
    try:
        mapping.require_unique_fields()
    except ValueError as exc:
        show_error(str(exc))
        return False
    missing = [field for field in required if mapping.source_for(field) is None]
    if missing:
        show_error("Colonnes obligatoires manquantes : " + ", ".join(FIELD_LABELS.get(field, field) for field in missing))
        return False
    return True


def candidate_table(candidates):
    return pd.DataFrame(
        [
            {
                "reference_id": candidate.reference.ref_id,
                "ville": candidate.reference.city_raw,
                "departement": candidate.reference.department_raw,
                "ville_et_departement": f"{candidate.reference.city_raw} - {candidate.reference.department_raw}",
                "latitude": candidate.reference.latitude,
                "longitude": candidate.reference.longitude,
                "score": candidate.score,
                "explication": candidate.explanation,
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
        updated.loc[mask, "city_resolution_explanation"] = "Validation humaine: ville consideree comme vide."
        return updated

    updated.loc[mask, "disappearance_city_verified"] = ref.city_raw
    updated.loc[mask, "disappearance_department_verified"] = ref.department_raw
    updated.loc[mask, "latitude"] = ref.latitude
    updated.loc[mask, "longitude"] = ref.longitude
    updated.loc[mask, "city_resolution_score"] = 100.0
    updated.loc[mask, "city_resolution_explanation"] = f"Validation humaine: ville {ref.city_raw} selectionnee."
    return updated


def matching_input_case_ids(clean_df: pd.DataFrame, source_case_id: str) -> list[str]:
    source_rows = clean_df[clean_df["case_id"].astype(str) == str(source_case_id)]
    if source_rows.empty:
        return [str(source_case_id)]
    source = source_rows.iloc[0]
    same_input = (
        clean_df["disappearance_city_normalized"].astype(str).eq(str(source["disappearance_city_normalized"]))
        & clean_df["disappearance_department_normalized"].astype(str).eq(str(source["disappearance_department_normalized"]))
    )
    return clean_df.loc[same_input, "case_id"].astype(str).tolist()


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="base_clean")
    return output.getvalue()


def simplified_export_df(clean_df: pd.DataFrame) -> pd.DataFrame:
    columns_to_remove = [
        "city_resolution_status",
        "city_resolution_score",
        "city_resolution_explanation",
        "human_validation_decision",
    ]
    return clean_df.drop(columns=[column for column in columns_to_remove if column in clean_df.columns])


st.title("Preparation de la base pour identity matching")

st.header("1. Importer la base de personnes disparues")
person_file = st.file_uploader("Base de personnes disparues", type=["csv", "xls", "xlsx"], key="person_file")
persons_df = None
person_mapping = None
if person_file:
    try:
        persons_df = read_uploaded_table(person_file.name, person_file.getvalue())
        st.dataframe(persons_df.head(20), use_container_width=True)
        st.subheader("2. Associer les colonnes des dossiers")
        person_mapping = mapping_editor(persons_df, PERSON_FIELDS, "person_map")
    except Exception as exc:
        show_error(f"Impossible de lire la base de dossiers : {exc}")

st.header("3. Importer ou charger la base de reference des villes")
city_file = st.file_uploader("Base de reference des villes", type=["csv", "xls", "xlsx"], key="city_file")
if city_file is None and st.button("Charger Consulta_Coordenada_2.0.csv local"):
    st.session_state["use_local_city_file"] = True

cities_df = None
city_mapping = None
if city_file or st.session_state.get("use_local_city_file"):
    try:
        if city_file:
            cities_df = read_uploaded_table(city_file.name, city_file.getvalue())
        else:
            local_path = Path("Consulta_Coordenada_2.0.csv")
            cities_df = read_local_table(str(local_path), local_path.stat().st_mtime)
        st.dataframe(cities_df.head(20), use_container_width=True)
        st.subheader("4. Associer les colonnes de la base de villes")
        city_mapping = mapping_editor(cities_df, CITY_FIELDS, "city_map")
    except Exception as exc:
        show_error(f"Impossible de lire la base de villes : {exc}")

can_build = persons_df is not None and person_mapping is not None and cities_df is not None and city_mapping is not None
st.header("5. Construire la base enrichie")
if can_build:
    st.info("La construction ajoute des coordonnees uniquement depuis la base de reference des villes.")
    if st.button("Lancer la construction de la base enrichie", type="primary"):
        if mapping_is_valid(person_mapping, ["case_id"]) and mapping_is_valid(city_mapping, ["city", "department", "latitude", "longitude"]):
            progress_bar = st.progress(0, text="Preparation de la construction...")
            status_box = st.empty()
            metrics_box = st.empty()
            try:
                total_persons = len(persons_df)
                status_box.info("Chargement et normalisation de la base de reference des villes...")
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
                        "dossiers a traiter": total_persons,
                        "villes de reference utilisables": len(refs),
                    }
                )
                progress_bar.progress(5, text="Base de villes prete. Nettoyage des dossiers en cours...")

                def update_build_progress(done: int, total: int, case_id: str) -> None:
                    percent = 5 + int((done / total) * 95) if total else 100
                    progress_bar.progress(
                        min(percent, 100),
                        text=f"Construction en cours : dossier {done}/{total} traite",
                    )
                    status_box.info(f"Dernier dossier traite : {case_id or 'id absent'}")

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
                progress_bar.progress(100, text="Construction terminee.")
                status_box.success("Base enrichie construite. Les conflits peuvent maintenant etre traites.")
                metrics_box.write(
                    {
                        "dossiers traites": summary["nombre total de dossiers"],
                        "villes verifiees": summary["nombre de dossiers avec ville verifiee"],
                        "conflits ou villes non resolues": summary["nombre de dossiers non resolus"],
                    }
                )
            except Exception as exc:
                progress_bar.empty()
                show_error(f"La construction a echoue : {exc}")

if "clean_df" in st.session_state:
    clean_df = st.session_state["clean_df"]
    refs = st.session_state["city_refs"]
    st.header("6. Traiter les conflits de villes")
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
    st.metric("Dossiers a resoudre", len(conflicts))
    if not conflicts.empty:
        status_counts = conflicts["city_resolution_status"].value_counts().rename_axis("statut").reset_index(name="nombre")
        st.dataframe(status_counts, use_container_width=True, hide_index=True)
    if conflicts.empty:
        st.success("Aucun conflit de ville a traiter.")
    else:
        selected_case = st.selectbox("Dossier en conflit", conflicts["case_id"].tolist())
        row = conflicts[conflicts["case_id"] == selected_case].iloc[0]
        st.write(
            {
                "id du dossier": row["case_id"],
                "ville originale": row["disappearance_city_raw"],
                "departement original": row["disappearance_department_raw"],
                "statut": row["city_resolution_status"],
                "explication": row["city_resolution_explanation"],
            }
        )
        resolution = resolve_city(row["disappearance_city_raw"], row["disappearance_department_raw"], refs)
        candidates_df = candidate_table(resolution.candidates)
        if not candidates_df.empty:
            st.dataframe(candidates_df, use_container_width=True)
            matching_case_ids = matching_input_case_ids(st.session_state["clean_df"], str(selected_case))
            st.caption(f"Cette decision sera appliquee a {len(matching_case_ids)} dossier(s) avec la meme ville et le meme departement d'origine.")
            chosen = st.selectbox(
                "Candidat a accepter",
                candidates_df["reference_id"].tolist(),
                format_func=lambda ref_id: candidates_df[candidates_df["reference_id"] == ref_id].iloc[0]["ville_et_departement"],
            )
            if st.button("Accepter ce candidat"):
                decisions = st.session_state.setdefault("human_decisions", {})
                for case_id in matching_case_ids:
                    decisions[str(case_id)] = HumanDecision("accept_candidate", reference_id=str(chosen))
                st.session_state["clean_df"] = apply_city_decision_to_clean_df(
                    st.session_state["clean_df"],
                    matching_case_ids,
                    str(chosen),
                    refs,
                    "human_validated_city",
                    "accept_candidate",
                )
                st.session_state["last_bulk_validation_message"] = f"Candidat accepte pour {len(matching_case_ids)} dossier(s) avec la meme ville et le meme departement d'origine."
                st.rerun()
        corrected_city = st.text_input("Corriger la ville puis relancer la recherche")
        if st.button("Relancer avec cette correction") and corrected_city.strip():
            new_resolution = resolve_city(corrected_city, row["disappearance_department_raw"], refs)
            st.dataframe(candidate_table(new_resolution.candidates), use_container_width=True)
        manual_query = st.text_input("Choisir manuellement une ville de reference")
        if manual_query:
            matches = [ref for ref in refs if manual_query.lower() in ref.city_raw.lower()][:50]
            if matches:
                selected_ref = st.selectbox(
                    "Ville de reference",
                    [ref.ref_id for ref in matches],
                    format_func=lambda ref_id: (
                        f"{next(ref for ref in matches if ref.ref_id == ref_id).city_raw} - "
                        f"{next(ref for ref in matches if ref.ref_id == ref_id).department_raw}"
                    ),
                )
                if st.button("Selectionner cette ville manuellement"):
                    matching_case_ids = matching_input_case_ids(st.session_state["clean_df"], str(selected_case))
                    decisions = st.session_state.setdefault("human_decisions", {})
                    for case_id in matching_case_ids:
                        decisions[str(case_id)] = HumanDecision("manual_reference", reference_id=str(selected_ref))
                    st.session_state["clean_df"] = apply_city_decision_to_clean_df(
                        st.session_state["clean_df"],
                        matching_case_ids,
                        str(selected_ref),
                        refs,
                        "human_selected_city",
                        "manual_reference",
                    )
                    st.session_state["last_bulk_validation_message"] = f"Ville selectionnee manuellement pour {len(matching_case_ids)} dossier(s) avec la meme ville et le meme departement d'origine."
                    st.rerun()
        if st.button("Considerer la ville comme vide pour ce dossier"):
            matching_case_ids = matching_input_case_ids(st.session_state["clean_df"], str(selected_case))
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
            st.session_state["last_bulk_validation_message"] = f"Ville consideree comme vide pour {len(matching_case_ids)} dossier(s) avec la meme ville et le meme departement d'origine."
            st.rerun()

    st.header("7. Resume qualite")
    summary = quality_summary(st.session_state["clean_df"])
    st.dataframe(pd.DataFrame(summary.items(), columns=["indicateur", "valeur"]), use_container_width=True)

    st.header("8. Exporter le CSV final")
    final_df = st.session_state["clean_df"]
    final_csv = final_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Exporter la base finale", data=final_csv, file_name="missing_persons_clean.csv", mime="text/csv")
    detailed_excel = dataframe_to_excel_bytes(final_df)
    simple_excel = dataframe_to_excel_bytes(simplified_export_df(final_df))
    st.download_button(
        "Exporter Excel detaille",
        data=detailed_excel,
        file_name="missing_persons_clean_detail.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "Exporter Excel simplifie",
        data=simple_excel,
        file_name="missing_persons_clean_simple.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.dataframe(final_df.head(100), use_container_width=True)
