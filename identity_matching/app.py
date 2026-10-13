from __future__ import annotations

import streamlit as st

from identity_matching.data_loading.clean_file_loader import dataframe_to_records
from identity_matching.data_loading.schema_validator import validate_clean_dataset
from identity_matching.ui.configuration_panel import render_configuration_panel
from identity_matching.ui.dataset_summary import render_dataset_summary
from identity_matching.ui.upload_section import render_upload_section


def main() -> None:
    """Run the Streamlit application workflow from upload to analysis configuration."""
    st.set_page_config(page_title="Identity Matching Tool", layout="wide")

    st.title("Potential Match Search")
    st.write(
        "This local tool helps an investigator prepare an analysis of potential matches "
        "between missing person case files. It does not make decisions on behalf of the "
        "investigator: scores are analytical aids, not evidence."
    )

    has_analysis_results = "analysis_results" in st.session_state
    if has_analysis_results:
        with st.expander("Step 1 - Loaded file", expanded=False):
            df = render_upload_section()
    else:
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
        with st.spinner("Checking the dataset and preparing case records..."):
            validation = validate_clean_dataset(df)
            records = dataframe_to_records(df) if validation.is_valid else []
        st.session_state["validated_file_signature"] = file_signature
        st.session_state["dataset_validation"] = validation
        st.session_state["missing_person_records"] = records

    if has_analysis_results:
        with st.expander("Dataset summary", expanded=False):
            render_dataset_summary(validation)
    else:
        render_dataset_summary(validation)
    if not validation.is_valid:
        return

    st.session_state["clean_dataset"] = df
    has_coordinates = validation.valid_coordinates_count > 0
    has_dates = validation.valid_dates_count > 0

    render_configuration_panel(df, records=records, has_coordinates=has_coordinates, has_dates=has_dates)


if __name__ == "__main__":
    main()
