"""Offer an explicit row-ID repair without discarding uploaded records."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from identity_matching.data_loading.case_ids import normalized_case_ids, prepare_unique_case_ids
from identity_matching.models import DatasetValidationResult
from identity_matching.ui.upload_section import clear_dataset_analysis


def render_case_id_resolution(df: pd.DataFrame, validation: DatasetValidationResult) -> bool:
    """Return whether analysis can continue; repair only after a button click."""
    if validation.is_valid:
        changed = st.session_state.get("case_id_repair_count", 0)
        if changed:
            st.success(
                f"Assigned unique row IDs to {changed} record(s). All {len(df)} rows are retained; "
                "original identifiers are preserved in source_case_id."
            )
            st.download_button(
                "Download CSV with unique row IDs",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="missing_persons_matching.csv",
                mime="text/csv",
                key="download_id_safe_csv",
            )
        return True

    if validation.missing_required_columns:
        return False

    ids = normalized_case_ids(df["case_id"])
    ambiguous = ids.eq("") | ids.duplicated(keep=False)
    with st.expander("Review case ID issues", expanded=True):
        st.write(
            "Check that case_id comes from the intended identifier column. "
            "If several rows intentionally share an identifier, you can give those rows "
            "unique IDs for matching. Every row and its original identifier will be kept."
        )
        preview = df.loc[ambiguous, ["case_id"]].head(20).copy()
        preview.insert(0, "CSV row", [position + 2 for position, bad in enumerate(ambiguous) if bad][:20])
        st.dataframe(preview, hide_index=True, use_container_width=True)
        if st.button("Use unique row IDs and continue", key="repair_case_ids", type="primary"):
            repaired, changed = prepare_unique_case_ids(df)
            st.session_state["loaded_clean_dataset"] = repaired
            st.session_state["case_id_repair_count"] = changed
            clear_dataset_analysis()
            st.rerun()
    return False
