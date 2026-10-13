from __future__ import annotations

import streamlit as st

from identity_matching.models import DatasetValidationResult


def render_dataset_summary(validation: DatasetValidationResult) -> None:
    """Render file quality metrics and blocking validation messages."""
    st.subheader("Dataset summary")
    st.write(validation.message)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Case records", validation.total_rows)
    col_b.metric("Valid GPS coordinates", validation.valid_coordinates_count)
    col_c.metric("Valid dates", validation.valid_dates_count)

    col_d, col_e = st.columns(2)
    col_d.metric("Missing or invalid GPS coordinates", validation.missing_coordinates_count)
    col_e.metric("Missing or invalid dates", validation.missing_dates_count)

    if validation.missing_case_ids_count:
        st.warning(f"{validation.missing_case_ids_count} case record(s) have a missing case_id.")

    if validation.duplicated_case_ids_count:
        st.warning(
            f"{validation.duplicated_case_ids_count} case record(s) share duplicate case_id values. "
            "These rows need distinct identifiers for matching."
        )

    if validation.missing_recommended_columns:
        st.warning(
            "Some recommended columns are missing. The file can still be used, "
            "but some future analyses may be less informative."
        )

    if not validation.is_valid:
        st.error("The workflow is blocked. " + validation.message)
