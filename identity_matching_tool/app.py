from __future__ import annotations

import streamlit as st

from src.data_loading.clean_file_loader import dataframe_to_records
from src.data_loading.schema_validator import validate_clean_dataset
from src.ui.configuration_panel import render_configuration_panel
from src.ui.dataset_summary import render_dataset_summary
from src.ui.upload_section import render_upload_section


def main() -> None:
    st.set_page_config(page_title="Identity Matching Tool", layout="wide")

    st.title("Recherche de correspondances potentielles")
    st.write(
        "Cet outil local aide un enqueteur a preparer une analyse de correspondances "
        "potentielles entre dossiers de personnes disparues. Il ne prend pas la decision "
        "a la place de l'enqueteur : les futurs scores seront des aides a l'analyse, pas des preuves."
    )

    df = render_upload_section()
    if df is None:
        return

    file_signature = st.session_state.get("uploaded_file_signature")
    if (
        st.session_state.get("validated_file_signature") == file_signature
        and "dataset_validation" in st.session_state
        and "missing_person_records" in st.session_state
    ):
        validation = st.session_state["dataset_validation"]
        records = st.session_state["missing_person_records"]
    else:
        validation_progress = st.progress(0, text="Validation du schema")
        validation = validate_clean_dataset(df)
        validation_progress.progress(50, text="Creation des dossiers internes")
        records = dataframe_to_records(df) if validation.is_valid else []
        validation_progress.progress(100, text="Preparation terminee")
        validation_progress.empty()
        st.session_state["validated_file_signature"] = file_signature
        st.session_state["dataset_validation"] = validation
        st.session_state["missing_person_records"] = records

    render_dataset_summary(validation)
    if not validation.is_valid:
        return

    st.session_state["clean_dataset"] = df
    has_coordinates = validation.valid_coordinates_count > 0
    has_dates = validation.valid_dates_count > 0

    render_configuration_panel(df, records=records, has_coordinates=has_coordinates, has_dates=has_dates)


if __name__ == "__main__":
    main()
