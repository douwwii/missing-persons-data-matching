from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from src.data_loading.clean_file_loader import CleanFileLoadingError, load_clean_csv


def render_upload_section():
    st.subheader("Etape 2 - Chargement du fichier clean")
    uploaded_file = st.file_uploader(
        "Charger le fichier CSV clean produit par l'etape 1",
        type=["csv"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Deposez un fichier CSV clean pour commencer l'analyse.")
        return None

    file_bytes = uploaded_file.getvalue()
    file_signature = _file_signature(uploaded_file.name, file_bytes)
    if (
        st.session_state.get("uploaded_file_signature") == file_signature
        and "loaded_clean_dataset" in st.session_state
    ):
        df = st.session_state["loaded_clean_dataset"]
        st.success("Fichier deja charge. Les donnees sont conservees pendant la session.")
        st.caption("Apercu des premiers dossiers")
        st.dataframe(df.head(10), use_container_width=True)
        return df

    try:
        progress = st.progress(0, text="Lecture du fichier CSV")
        progress.progress(25, text="Lecture du fichier CSV")
        df = _load_csv_from_bytes(file_bytes)
        progress.progress(80, text="Preparation de l'apercu")
    except CleanFileLoadingError as exc:
        st.error(str(exc))
        return None
    finally:
        if "progress" in locals():
            progress.empty()

    st.session_state["uploaded_file_signature"] = file_signature
    st.session_state["loaded_clean_dataset"] = df
    st.session_state.pop("analysis_config", None)
    st.session_state.pop("analysis_results", None)

    st.success("Fichier charge avec succes.")
    st.caption("Apercu des premiers dossiers")
    st.dataframe(df.head(10), use_container_width=True)
    return df


def _file_signature(file_name: str, file_bytes: bytes) -> str:
    digest = hashlib.sha256(file_bytes).hexdigest()
    return f"{file_name}:{digest}"


@st.cache_data(show_spinner=False)
def _load_csv_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO

    return load_clean_csv(BytesIO(file_bytes))
