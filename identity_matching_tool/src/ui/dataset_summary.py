from __future__ import annotations

import streamlit as st

from src.config import RECOMMENDED_COLUMNS, REQUIRED_COLUMNS
from src.models import DatasetValidationResult


def render_dataset_summary(validation: DatasetValidationResult) -> None:
    st.subheader("Etape 3 - Validation du fichier")
    st.write(validation.message)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Dossiers", validation.total_rows)
    col_b.metric("Coordonnees GPS valides", validation.valid_coordinates_count)
    col_c.metric("Dates valides", validation.valid_dates_count)

    col_d, col_e, col_f = st.columns(3)
    col_d.metric("Coordonnees GPS manquantes ou invalides", validation.missing_coordinates_count)
    col_e.metric("Dates manquantes ou invalides", validation.missing_dates_count)
    col_f.metric("Identifiants dupliques", validation.duplicated_case_ids_count)

    _render_column_status("Colonnes obligatoires", REQUIRED_COLUMNS, validation.missing_required_columns)
    _render_column_status("Colonnes recommandees", RECOMMENDED_COLUMNS, validation.missing_recommended_columns)

    if validation.missing_case_ids_count:
        st.warning(f"{validation.missing_case_ids_count} dossier(s) ont un case_id manquant.")

    if validation.missing_recommended_columns:
        st.warning(
            "Certaines colonnes recommandees sont absentes. Le fichier reste utilisable, "
            "mais certaines analyses futures pourront etre moins riches."
        )

    if not validation.is_valid:
        st.error(
            "La suite est bloquee. Corrigez la sortie de l'etape 1 pour inclure toutes "
            "les colonnes obligatoires, puis rechargez le fichier."
        )


def _render_column_status(title: str, expected_columns: list[str], missing_columns: list[str]) -> None:
    with st.expander(title, expanded=bool(missing_columns)):
        present_columns = [column for column in expected_columns if column not in missing_columns]
        if present_columns:
            st.success("Presentes : " + ", ".join(present_columns))
        if missing_columns:
            st.error("Manquantes : " + ", ".join(missing_columns))
        if not missing_columns:
            st.info("Aucune colonne manquante.")

