from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from identity_matching.data_loading.clean_file_loader import CleanFileLoadingError, load_clean_csv

_CSV_LOADER_VERSION = 2


def clear_dataset_analysis() -> None:
    """Invalidate records and results whenever the effective input table changes."""
    for key in (
        "analysis_config",
        "analysis_results",
        "analysis_history",
        "analysis_history_index",
        "validated_file_signature",
        "dataset_validation",
        "missing_person_records",
        "clean_dataset",
        "open_match_detail_index",
    ):
        st.session_state.pop(key, None)


def render_upload_section():
    """Render the CSV upload area and return the loaded clean dataset."""
    st.subheader("Step 1 - Load the clean file")
    uploaded_file = st.file_uploader(
        "Upload the clean CSV file produced by step 1",
        type=["csv"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Drop a clean CSV file to start the analysis.")
        return None

    file_bytes = uploaded_file.getvalue()
    file_signature = _file_signature(uploaded_file.name, file_bytes)
    if (
        st.session_state.get("uploaded_file_signature") == file_signature
        and "loaded_clean_dataset" in st.session_state
        and st.session_state.get("dataset_loader_version") == _CSV_LOADER_VERSION
    ):
        df = st.session_state["loaded_clean_dataset"]
        st.success("File already loaded. Data is kept during this session.")
        st.caption("Preview of the first case records")
        st.dataframe(df.head(10), use_container_width=True)
        return df

    try:
        with st.spinner("Loading the clean CSV file..."):
            df = _load_csv_from_bytes(file_bytes)
    except CleanFileLoadingError as exc:
        st.error(str(exc))
        return None

    st.session_state["uploaded_file_signature"] = file_signature
    st.session_state["loaded_clean_dataset"] = df
    st.session_state["dataset_loader_version"] = _CSV_LOADER_VERSION
    st.session_state.pop("case_id_repair_count", None)
    clear_dataset_analysis()

    st.success("File loaded successfully.")
    st.caption("Preview of the first case records")
    st.dataframe(df.head(10), use_container_width=True)
    return df


def _file_signature(file_name: str, file_bytes: bytes) -> str:
    """Create a stable session signature for an uploaded file."""
    digest = hashlib.sha256(file_bytes).hexdigest()
    return f"{file_name}:{digest}"


@st.cache_data(show_spinner=False)
def _load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Load CSV bytes while preserving literal and zero-padded identifiers."""
    from io import BytesIO

    return load_clean_csv(BytesIO(file_bytes))
