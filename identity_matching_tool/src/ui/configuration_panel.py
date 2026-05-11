from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.config import INVESTIGATION_MODE_GLOBAL, INVESTIGATION_MODE_SINGLE
from src.matching.matching_engine import run_matching, run_matching_on_pair_results
from src.models import AnalysisConfig, BlockingConfig, MatchingRunResult, MissingPersonRecord, ParallelConfig
from src.similarity.combined_similarity import compute_combined_similarity
from src.ui.analysis_history import append_refinement_step, initialize_analysis_history, navigate_history

INVESTIGATION_MODE_OPTIONS = {
    "Single case matching": (
        INVESTIGATION_MODE_SINGLE,
        "Compare one selected case against the other records in the dataset.",
    ),
    "Global matching": (
        INVESTIGATION_MODE_GLOBAL,
        "Search potential matches across the whole dataset. This may evaluate many case pairs.",
    ),
}

ANALYSIS_MODE_OPTIONS = {
    "Exhaustive / validation": (
        "exhaustive",
        "Compares every possible pair. Useful for small datasets or filter validation, but it can be slow.",
    ),
    "Standard blocking": (
        "standard_blocking",
        "Only compares candidate pairs selected by simple name, date, and location rules. Recommended for large datasets.",
    ),
    "Custom": (
        "custom_placeholder",
        "Planned for a future version. It will let the investigator choose custom filtering rules.",
    ),
}

@st.fragment
def render_configuration_panel(
    df: pd.DataFrame,
    records: list[MissingPersonRecord],
    has_coordinates: bool,
    has_dates: bool,
) -> None:
    """Render the investigator workflow for analysis and filter configuration."""
    st.subheader("Step 2 - Analysis configuration")

    st.markdown("**Investigation mode**")
    mode_label = st.radio(
        "Choose mode",
        list(INVESTIGATION_MODE_OPTIONS.keys()),
        horizontal=True,
        key="investigation_mode_label",
    )
    investigation_mode = INVESTIGATION_MODE_OPTIONS[mode_label][0]
    _render_investigation_mode_description(investigation_mode)
    _sync_default_analysis_mode(investigation_mode)

    selected_case_id = None
    if investigation_mode == INVESTIGATION_MODE_SINGLE:
        case_ids = _available_case_ids(df)
        selected_case_id = (
            st.selectbox(
                "Target case",
                case_ids,
                key="selected_case_id",
                help="Case record used as the starting point for single case matching.",
            )
            if case_ids
            else None
        )
        if not case_ids:
            st.warning("No usable case_id is available to select a target case.")
    else:
        st.info("This mode may compare many case pairs during the next steps.")

    st.markdown("**Analysis mode**")
    analysis_mode_label = st.radio(
        "Choose analysis mode",
        list(ANALYSIS_MODE_OPTIONS.keys()),
        horizontal=True,
        key="analysis_mode_label",
    )
    analysis_mode = ANALYSIS_MODE_OPTIONS[analysis_mode_label][0]
    _render_analysis_mode_description(analysis_mode)
    if investigation_mode == INVESTIGATION_MODE_GLOBAL and analysis_mode == "exhaustive":
        pair_count = len(records) * (len(records) - 1) // 2
        st.warning(
            f"Global exhaustive mode will compare {pair_count} possible case pair(s). "
            "This can take a long time on large datasets."
        )

    st.markdown("**Similarity filters**")
    col_a, col_b = st.columns(2)
    use_name = col_a.checkbox(
        "Name",
        value=True,
        key="filter_name",
        help="Compare already-normalized written names with structured, reordered, and token-alignment scores.",
    )
    use_phonetic = col_a.checkbox(
        "Name phonetics",
        value=False,
        key="filter_phonetic",
        help="Compare names after IPA transliteration with PanPhon. More costly, so it is disabled by default.",
    )
    use_geographic = col_b.checkbox(
        "Location",
        value=has_coordinates,
        key="filter_geographic",
        help="Compare existing GPS coordinates from the clean dataset. No city resolution is performed here.",
    )
    use_temporal = col_b.checkbox(
        "Time",
        value=has_dates,
        key="filter_temporal",
        help="Compare parsed disappearance dates using a maximum accepted day gap.",
    )

    st.markdown("**Filter parameters**")
    filter_parameters: dict[str, float | int] = {}
    if use_geographic:
        with st.container(border=True):
            st.caption("Location")
            geo_col_a, geo_col_b = st.columns(2)
            filter_parameters["geographic_distance_min_km"] = geo_col_a.number_input(
                "Distance below which similarity is maximal (km)",
                min_value=0.0,
                value=5.0,
                step=1.0,
                key="geographic_distance_min_km",
                help="Pairs closer than this distance receive the maximum location similarity score.",
            )
            filter_parameters["geographic_distance_max_km"] = geo_col_b.number_input(
                "Distance above which similarity is minimal (km)",
                min_value=0.0,
                value=50.0,
                step=5.0,
                key="geographic_distance_max_km",
                help="Pairs farther than this distance receive the minimum location similarity score.",
            )
    else:
        filter_parameters["geographic_distance_min_km"] = 5.0
        filter_parameters["geographic_distance_max_km"] = 50.0

    if use_temporal:
        with st.container(border=True):
            st.caption("Time")
            filter_parameters["temporal_max_days"] = st.number_input(
                "Number of days after which similarity is zero",
                min_value=1,
                value=365,
                step=30,
                key="temporal_max_days",
                help="Date gaps greater than or equal to this value receive a time similarity score of zero.",
            )
    else:
        filter_parameters["temporal_max_days"] = 365

    with st.container(border=True):
        st.caption("General parameters")
        filter_parameters["max_results"] = st.number_input(
            "Maximum number of potential matches",
            min_value=1,
            value=50,
            step=10,
            key="max_results",
            help="Limits the number of highest-scoring potential matches displayed after the analysis.",
        )
        filter_parameters["min_score_threshold"] = 0.0

    blocking_config = BlockingConfig()
    parallel_config = ParallelConfig()

    enabled_filters = {
        "name": use_name,
        "phonetic": use_phonetic,
        "geographic": use_geographic,
        "temporal": use_temporal,
    }

    config = AnalysisConfig(
        investigation_mode=investigation_mode,
        analysis_mode=analysis_mode,
        selected_case_id=str(selected_case_id) if selected_case_id is not None else None,
        enabled_filters=enabled_filters,
        filter_parameters=filter_parameters,
    )

    st.subheader("Step 3 - Confirm configuration")
    if st.button(
        "Confirm configuration",
        type="primary",
        help="Store the selected mode, filters, and parameters before running the analysis.",
    ):
        st.session_state["analysis_config"] = config
        st.session_state["blocking_config"] = blocking_config
        st.session_state["parallel_config"] = parallel_config
        st.session_state.pop("analysis_results", None)
        st.session_state.pop("analysis_history", None)
        st.session_state.pop("analysis_history_index", None)
        st.session_state.pop("open_match_detail_index", None)
        st.session_state.pop("scroll_to_results_table", None)
        st.session_state.pop("scroll_to_detail_top", None)

    if "analysis_config" in st.session_state:
        render_config_summary(st.session_state["analysis_config"])
        render_analysis_launcher(
            records,
            st.session_state["analysis_config"],
            st.session_state.get("blocking_config"),
            st.session_state.get("parallel_config"),
        )


def render_config_summary(config: AnalysisConfig) -> None:
    """Display the confirmed analysis configuration."""
    mode_label = "Single case matching" if config.investigation_mode == INVESTIGATION_MODE_SINGLE else "Global matching"
    enabled = [name for name, enabled in config.enabled_filters.items() if enabled]
    st.success("Configuration confirmed.")
    st.write(
        {
            "mode": mode_label,
            "analysis_mode": config.analysis_mode,
            "target_case": config.selected_case_id,
            "selected_filters": enabled,
            "parameters": config.filter_parameters,
        }
    )


def render_analysis_launcher(
    records: list[MissingPersonRecord],
    config: AnalysisConfig,
    blocking_config: BlockingConfig | None,
    parallel_config: ParallelConfig | None,
) -> None:
    """Render the explicit analysis launch button, progress bar, and results."""
    st.subheader("Step 4 - Analysis")
    st.caption("The analysis computes investigative support scores. It does not produce evidence.")

    if not any(config.enabled_filters.values()):
        st.warning("Enable at least one similarity filter before running the analysis.")
        return

    if config.investigation_mode == INVESTIGATION_MODE_SINGLE and not config.selected_case_id:
        st.warning("Select a target case before running the analysis.")
        return

    if config.investigation_mode == INVESTIGATION_MODE_GLOBAL:
        pair_count = len(records) * (len(records) - 1) // 2
        st.info(f"Global mode: up to {pair_count} case pair(s) may be evaluated locally.")

    if config.analysis_mode == "custom_placeholder":
        st.warning("Custom mode will be added in a future version.")

    if st.button(
        "Run analysis",
        type="primary",
        key="run_similarity_analysis",
        help="Start scoring potential matches with the confirmed configuration.",
    ):
        try:
            progress = st.progress(0, text="Preparing analysis")
            status_placeholder = st.empty()

            def update_progress(step_label: str, current: int, total: int) -> None:
                """Update Streamlit progress widgets from matching engine callbacks."""
                total = max(total, 1)
                ratio = min(max(current / total, 0.0), 1.0)
                start, end = _progress_range_for_step(step_label)
                percent = int(start + ratio * (end - start))
                progress.progress(percent, text=f"{step_label} : {current}/{total}")
                status_placeholder.caption(f"{step_label} : {current}/{total}")

            with st.spinner("Computing potential matches..."):
                result = run_matching(
                    records,
                    config,
                    blocking_config,
                    parallel_config,
                    progress_callback=update_progress,
                )
                initialize_analysis_history(st.session_state, result, config)
            progress.progress(100, text="Analysis complete")
            status_placeholder.empty()
            progress.empty()
        except ValueError as exc:
            st.error(str(exc))
            return
        except RuntimeError as exc:
            st.error(str(exc))
            return

    if "analysis_results" in st.session_state:
        result = st.session_state["analysis_results"]
        _render_matching_result(result, records)


def _render_matching_result(result: MatchingRunResult, records: list[MissingPersonRecord]) -> None:
    """Render matching metrics, technical details, performance details, and result rows."""
    if result.status == "not_implemented":
        st.info(result.explanation)
        return
    open_detail_index = st.session_state.get("open_match_detail_index")
    if open_detail_index is not None:
        try:
            selected_index = int(open_detail_index)
            pair = result.results[selected_index]
        except (TypeError, ValueError, IndexError):
            st.session_state.pop("open_match_detail_index", None)
        else:
            _render_pair_details_page(pair, {record.case_id: record for record in records})
            return

    _render_analysis_history_navigation()

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Case records", result.total_records)
    metric_b.metric("Theoretical pairs", result.theoretical_pairs_count)
    metric_c.metric("Compared pairs", result.compared_pairs_count)
    metric_d.metric("Retained results", result.retained_results_count)

    metric_e, metric_f = st.columns(2)
    if result.reduction_percentage is not None:
        metric_e.metric("Reduction", f"{result.reduction_percentage:.1%}")
    metric_f.metric("Runtime", f"{result.elapsed_time:.2f} s")

    with st.expander("Technical details"):
        st.caption(
            "This section explains how candidate pairs were prepared before scoring. "
            "Blocking only reduces the number of pairs to inspect; it does not decide whether two records match."
        )
        st.write(result.explanation)
        if result.blocking_result is not None:
            st.write(result.blocking_result.explanation)
            if result.blocking_result.blocking_index_cache_dir:
                st.markdown(
                    "**Blocking bucket cache**  \n"
                    "Standard blocking groups records into reusable buckets. A bucket is a group of case IDs "
                    "sharing the same blocking characteristic, such as compatible name initials within a date window "
                    "or a nearby geo-time cell. During analysis, the app crosses these buckets to find candidate pairs "
                    "without scanning every possible pair."
                )
                st.write(
                    {
                        "bucket_cache_folder": result.blocking_result.blocking_index_cache_dir,
                        "metadata.json": "General information about the generated blocking index.",
                        "bucket_summary.csv": "Bucket counts and bucket size statistics for each rule.",
                        "r1_initials_buckets.csv": "Buckets based on compatible name initials and close dates.",
                        "r2_phonetic_onset_buckets.csv": "Buckets based on simple phonetic name starts and close dates.",
                        "r3_geo_time_buckets.csv": "Buckets based on nearby GPS coordinates and close dates.",
                        "signatures.csv": "One lightweight blocking signature per case record.",
                    }
                )

    with st.expander("Performance details"):
        st.caption(
            "This section describes how much work was actually done during the run. "
            "Cached pairs were reused, computed pairs were scored during this run."
        )
        st.write(
            {
                "execution_mode": result.execution_mode,
                "execution_mode_explanation": (
                    "sequential uses one worker; parallel distributes score computation across local CPU workers."
                ),
                "parallelization_enabled": result.parallel_enabled,
                "workers": result.num_workers,
                "chunk_size": result.chunk_size,
                "total_pairs": result.total_pairs,
                "total_pairs_explanation": "Number of pairs selected for cache lookup and possible scoring.",
                "computed_pairs": result.computed_pairs,
                "cached_pairs": result.cached_pairs,
                "cache_hit_rate": f"{result.cache_hit_rate:.1%}",
                "cache_hit_rate_explanation": "Share of selected pairs that were already available in the pairwise cache.",
                "scoring_time": f"{result.parallel_elapsed_time:.3f} s",
                "total_time": f"{result.elapsed_time:.3f} s",
            }
        )
        if result.blocking_result is not None:
            st.markdown(
                "**Standard blocking rules**  \n"
                "R1 keeps pairs with compatible name initials and close disappearance dates.  \n"
                "R2 keeps pairs with compatible simple phonetic name starts and close disappearance dates.  \n"
                "R3 keeps pairs with strong geographic and temporal proximity."
            )
            st.write(
                {
                    "pairs_kept_by_R1": result.blocking_result.r1_count,
                    "pairs_kept_by_R2": result.blocking_result.r2_count,
                    "pairs_kept_by_R3": result.blocking_result.r3_count,
                    "reduction": f"{result.blocking_result.reduction_percentage:.1%}",
                    "reduction_explanation": (
                        "Percentage of theoretical pairs avoided by blocking before score computation."
                    ),
                }
            )

    _render_refinement_panel(result, records)

    rows = _result_rows(result)
    if not rows:
        st.info("No potential matches to display.")
        return
    st.markdown('<div id="results-table-anchor"></div>', unsafe_allow_html=True)
    if st.session_state.pop("scroll_to_results_table", False):
        _scroll_to_results_table()
    st.success(f"{len(rows)} potential match(es) displayed.")
    st.caption("Use the Detail match button at the end of a row to open the dedicated match review page.")
    _render_result_rows_with_buttons(result, rows, records)


def _render_analysis_history_navigation() -> None:
    """Render previous/next navigation for temporary analysis history."""
    history = st.session_state.get("analysis_history", [])
    if not history:
        return
    index = int(st.session_state.get("analysis_history_index", 0))
    left, middle, right = st.columns([1, 2, 1])
    if left.button("Previous analysis", disabled=index <= 0, use_container_width=True):
        navigate_history(st.session_state, -1)
        st.rerun()
    middle.caption(f"Analysis step {index + 1} of {len(history)}")
    if right.button("Next analysis", disabled=index >= len(history) - 1, use_container_width=True):
        navigate_history(st.session_state, 1)
        st.rerun()


def _render_refinement_panel(result: MatchingRunResult, records: list[MissingPersonRecord]) -> None:
    """Render controls for rescoring the currently displayed result pairs."""
    if not result.results:
        return
    with st.expander("Refine current results"):
        st.caption("Only the potential matches currently displayed below will be rescored.")
        config = _refinement_config_from_controls(result)
        if not any(config.enabled_filters.values()):
            st.warning("Enable at least one similarity filter before running a refinement.")
            return
        if st.button("Run refinement on displayed results", type="primary", key="run_visible_results_refinement"):
            progress = st.progress(0, text="Preparing refinement")
            status_placeholder = st.empty()

            def update_progress(step_label: str, current: int, total: int) -> None:
                total = max(total, 1)
                ratio = min(max(current / total, 0.0), 1.0)
                start, end = _progress_range_for_step(step_label)
                percent = int(start + ratio * (end - start))
                progress.progress(percent, text=f"{step_label} : {current}/{total}")
                status_placeholder.caption(f"{step_label} : {current}/{total}")

            with st.spinner("Refining displayed potential matches..."):
                refined = run_matching_on_pair_results(
                    records,
                    result.results,
                    config,
                    st.session_state.get("parallel_config"),
                    progress_callback=update_progress,
                )
                append_refinement_step(st.session_state, refined, config)
            progress.progress(100, text="Refinement complete")
            status_placeholder.empty()
            progress.empty()
            st.rerun()


def _refinement_config_from_controls(result: MatchingRunResult) -> AnalysisConfig:
    """Build a refinement config from compact controls scoped to visible pairs."""
    base_config = st.session_state.get("analysis_config")
    base_filters = base_config.enabled_filters if base_config is not None else {}
    base_parameters = base_config.filter_parameters if base_config is not None else {}

    col_a, col_b = st.columns(2)
    use_name = col_a.checkbox("Name", value=bool(base_filters.get("name", True)), key="refine_filter_name")
    use_phonetic = col_a.checkbox(
        "Name phonetics",
        value=bool(base_filters.get("phonetic", False)),
        key="refine_filter_phonetic",
    )
    use_geographic = col_b.checkbox(
        "Location",
        value=bool(base_filters.get("geographic", False)),
        key="refine_filter_geographic",
    )
    use_temporal = col_b.checkbox(
        "Time",
        value=bool(base_filters.get("temporal", False)),
        key="refine_filter_temporal",
    )

    parameters: dict[str, float | int] = {}
    if use_geographic:
        geo_col_a, geo_col_b = st.columns(2)
        parameters["geographic_distance_min_km"] = geo_col_a.number_input(
            "Refinement distance below which similarity is maximal (km)",
            min_value=0.0,
            value=float(base_parameters.get("geographic_distance_min_km", 5.0)),
            step=1.0,
            key="refine_geographic_distance_min_km",
        )
        parameters["geographic_distance_max_km"] = geo_col_b.number_input(
            "Refinement distance above which similarity is minimal (km)",
            min_value=0.0,
            value=float(base_parameters.get("geographic_distance_max_km", 50.0)),
            step=5.0,
            key="refine_geographic_distance_max_km",
        )
    else:
        parameters["geographic_distance_min_km"] = float(base_parameters.get("geographic_distance_min_km", 5.0))
        parameters["geographic_distance_max_km"] = float(base_parameters.get("geographic_distance_max_km", 50.0))

    if use_temporal:
        parameters["temporal_max_days"] = st.number_input(
            "Refinement number of days after which similarity is zero",
            min_value=1,
            value=int(base_parameters.get("temporal_max_days", 365)),
            step=30,
            key="refine_temporal_max_days",
        )
    else:
        parameters["temporal_max_days"] = int(base_parameters.get("temporal_max_days", 365))

    parameters["max_results"] = st.number_input(
        "Refinement maximum number of potential matches",
        min_value=1,
        value=int(base_parameters.get("max_results", max(len(result.results), 1))),
        step=10,
        key="refine_max_results",
    )
    parameters["min_score_threshold"] = float(base_parameters.get("min_score_threshold", 0.0))

    return AnalysisConfig(
        investigation_mode=result.investigation_mode,
        analysis_mode="exhaustive",
        selected_case_id=None,
        enabled_filters={
            "name": use_name,
            "phonetic": use_phonetic,
            "geographic": use_geographic,
            "temporal": use_temporal,
        },
        filter_parameters=parameters,
    )


def _result_rows(result: MatchingRunResult) -> list[dict[str, object]]:
    """Convert pair matching results into table rows for Streamlit display."""
    return [
        {
            "_result_index": index,
            "score_final": round(pair.score_final or 0.0, 3),
            "case_A": f"{pair.id_A} - {pair.name_A}",
            "case_B": f"{pair.id_B} - {pair.name_B}",
            "name_score": _round_optional(pair.score_name),
            "phonetic_score": _round_optional(pair.score_phonetic),
            "geographic_score": _round_optional(pair.score_geographic),
            "temporal_score": _round_optional(pair.score_temporal),
            "date_diff_days": pair.date_diff_days if pair.date_diff_days is not None else pair.distance_days,
            "geo_distance_km": (
                round(pair.geo_distance_km, 1)
                if pair.geo_distance_km is not None
                else round(pair.distance_km, 1)
                if pair.distance_km is not None
                else None
            ),
            "matched_rules": ", ".join(pair.matched_rules),
        }
        for index, pair in enumerate(result.results)
    ]


def _scroll_to_results_table() -> None:
    """Scroll the browser viewport to the results table anchor after returning from details."""
    components.html(
        """
        <script>
        const anchor = window.parent.document.getElementById("results-table-anchor");
        if (anchor) {
            anchor.scrollIntoView({behavior: "smooth", block: "start"});
        }
        </script>
        """,
        height=0,
    )


def _scroll_to_match_detail_top() -> None:
    """Scroll the browser viewport to the top of the match detail page."""
    components.html(
        """
        <script>
        const anchor = window.parent.document.getElementById("match-detail-top");
        if (anchor) {
            anchor.scrollIntoView({behavior: "smooth", block: "start"});
        } else {
            window.parent.scrollTo({top: 0, behavior: "smooth"});
        }
        </script>
        """,
        height=0,
    )


def _render_result_rows_with_buttons(
    result: MatchingRunResult,
    rows: list[dict[str, object]],
    records: list[MissingPersonRecord],
) -> None:
    """Render result rows with one explicit detail button per potential match."""
    header = st.columns([0.7, 2.3, 2.3, 0.8, 0.9, 0.9, 0.8, 0.7, 0.7, 0.8, 1.0])
    for column, label in zip(
        header,
        ["Score", "Case A", "Case B", "Name", "Phonetic", "Location", "Time", "Days", "Km", "Rules", ""],
    ):
        column.markdown(f"**{label}**")

    for row in rows:
        index = int(row["_result_index"])
        columns = st.columns([0.7, 2.3, 2.3, 0.8, 0.9, 0.9, 0.8, 0.7, 0.7, 0.8, 1.0])
        columns[0].write(_display_value(row["score_final"]))
        columns[1].write(_display_value(row["case_A"]))
        columns[2].write(_display_value(row["case_B"]))
        columns[3].write(_display_value(row["name_score"]))
        columns[4].write(_display_value(row["phonetic_score"]))
        columns[5].write(_display_value(row["geographic_score"]))
        columns[6].write(_display_value(row["temporal_score"]))
        columns[7].write(_display_value(row["date_diff_days"]))
        columns[8].write(_display_value(row["geo_distance_km"]))
        columns[9].write(_display_value(row["matched_rules"]))
        if columns[10].button("Detail match", key=f"detail_match_{index}", use_container_width=True):
            st.session_state["open_match_detail_index"] = index
            st.session_state["scroll_to_detail_top"] = True
            st.rerun()

def _render_pair_details_page(pair: Any, records_by_id: dict[str, MissingPersonRecord]) -> None:
    """Render a dedicated full-width page for one selected potential match."""
    st.markdown('<div id="match-detail-top"></div>', unsafe_allow_html=True)
    if st.session_state.pop("scroll_to_detail_top", False):
        _scroll_to_match_detail_top()

    st.markdown(
        """
        <style>
        div[data-testid="stAppViewContainer"] .block-container {
            max-width: 98vw;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    record_a = records_by_id.get(pair.id_A)
    record_b = records_by_id.get(pair.id_B)
    if record_a is None or record_b is None:
        st.warning("Case details cannot be displayed because one of the records is no longer available.")
        return

    top_left, top_right = st.columns([4, 1])
    top_left.caption("Results > Match detail")
    top_left.header("Potential match review")
    if top_right.button("Back to results", type="primary", use_container_width=True):
        st.session_state.pop("open_match_detail_index", None)
        st.session_state["scroll_to_results_table"] = True
        st.rerun()

    st.markdown("**Case records side by side**")
    st.dataframe(
        pd.DataFrame(_record_comparison_rows(record_a, record_b)),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Algorithmic evidence**")
    similarity = _full_similarity_for_display(pair, record_a, record_b)
    _render_full_algorithmic_details(pair, similarity)


def _record_comparison_rows(record_a: MissingPersonRecord, record_b: MissingPersonRecord) -> list[dict[str, object]]:
    """Build side-by-side rows for the main fields of two records."""
    fields = [
        ("case_id", record_a.case_id, record_b.case_id),
        ("full_name_raw", record_a.full_name_raw, record_b.full_name_raw),
        ("full_name_normalized", record_a.full_name_normalized, record_b.full_name_normalized),
        ("first_name_normalized", record_a.first_name_normalized, record_b.first_name_normalized),
        ("last_name_normalized", record_a.last_name_normalized, record_b.last_name_normalized),
        ("name_tokens", record_a.name_tokens, record_b.name_tokens),
        ("disappearance_city_verified", record_a.disappearance_city_verified, record_b.disappearance_city_verified),
        (
            "disappearance_department_verified",
            record_a.disappearance_department_verified,
            record_b.disappearance_department_verified,
        ),
        ("latitude", record_a.latitude, record_b.latitude),
        ("longitude", record_a.longitude, record_b.longitude),
        ("disappearance_date_parsed", record_a.disappearance_date_parsed, record_b.disappearance_date_parsed),
    ]
    return [
        {"field": field, "case_A": _display_value(value_a), "case_B": _display_value(value_b)}
        for field, value_a, value_b in fields
    ]


def _full_similarity_for_display(pair: Any, record_a: MissingPersonRecord, record_b: MissingPersonRecord) -> Any:
    """Return full similarity details, recomputing one pair when the row came from cache."""
    if pair.similarity_result is not None:
        return pair.similarity_result
    config = st.session_state.get("analysis_config")
    if config is None:
        return None
    return compute_combined_similarity(record_a, record_b, config)


def _render_full_algorithmic_details(pair: Any, similarity: Any) -> None:
    """Render all available calculation details for each similarity dimension."""
    if similarity is None:
        st.warning("Full algorithmic details are not available for this cached result.")
        return

    if similarity.name_result is not None:
        name = similarity.name_result
        st.markdown("**Name filter**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "operation": "Structured first/last comparison",
                        "result": _score_label(name.direct_structured_score),
                        "how to read it": (
                            "Compares first name with first name and last name with last name. "
                            "Available only when both records contain structured first and last names."
                        ),
                    },
                    {
                        "operation": "Direct full-name comparison",
                        "result": _score_label(name.direct_full_name_score),
                        "how to read it": (
                            "Compares the full normalized names in their original order. "
                            "Used when structured first/last names are missing."
                        ),
                    },
                    {
                        "operation": "Reordered full-name comparison",
                        "result": _score_label(name.reordered_full_name_score),
                        "how to read it": (
                            "Sorts name tokens before comparison. A high value suggests the same name elements "
                            "may appear in a different order."
                        ),
                    },
                    {
                        "operation": "Token alignment",
                        "result": _score_label(name.token_alignment_score),
                        "how to read it": (
                            "Matches each token with its closest token in the other name and penalizes extra tokens. "
                            "Useful when one record has a longer name."
                        ),
                    },
                    {
                        "operation": "Order mismatch suspected",
                        "result": _display_value(name.order_mismatch_suspected),
                        "how to read it": (
                            "True means the direct comparison is weaker but the reordered comparison is strong."
                        ),
                    },
                    {
                        "operation": "Name method retained",
                        "result": _display_value(name.best_method),
                        "how to read it": "Indicates how the final written-name score was assembled.",
                    },
                    {
                        "operation": "Name explanation",
                        "result": name.explanation,
                        "how to read it": "Plain-language summary of the written-name calculation.",
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    if similarity.phonetic_result is not None:
        phonetic = similarity.phonetic_result
        st.markdown("**Phonetic filter**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "operation": "Phonetic status",
                        "result": phonetic.status,
                        "how to read it": "'ok' means IPA conversion and PanPhon comparison were available.",
                    },
                    {
                        "operation": "Phonetic score",
                        "result": _score_label(phonetic.score_final),
                        "how to read it": "Similarity between the two IPA strings. Higher values mean closer pronunciation.",
                    },
                    {
                        "operation": "IPA for case A",
                        "result": _display_value(phonetic.ipa_a),
                        "how to read it": "Name of case A converted to IPA for pronunciation-based comparison.",
                    },
                    {
                        "operation": "IPA for case B",
                        "result": _display_value(phonetic.ipa_b),
                        "how to read it": "Name of case B converted to IPA for pronunciation-based comparison.",
                    },
                    {
                        "operation": "Phonetic explanation",
                        "result": phonetic.explanation,
                        "how to read it": "Plain-language summary of how the phonetic score was produced.",
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    if similarity.geo_result is not None:
        geo = similarity.geo_result
        st.markdown("**Location filter**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "operation": "Location status",
                        "result": geo.status,
                        "how to read it": "'ok' means both records had usable GPS coordinates.",
                    },
                    {
                        "operation": "Distance between locations",
                        "result": f"{_score_label(geo.distance_km)} km" if geo.distance_km is not None else "n/a",
                        "how to read it": "Haversine distance between the two disappearance coordinates.",
                    },
                    {
                        "operation": "Location score",
                        "result": _score_label(geo.score),
                        "how to read it": "Score decreases as GPS distance increases.",
                    },
                    {
                        "operation": "Location explanation",
                        "result": geo.explanation,
                        "how to read it": "Plain-language summary of the location comparison.",
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    if similarity.temporal_result is not None:
        temporal = similarity.temporal_result
        st.markdown("**Time filter**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "operation": "Time status",
                        "result": temporal.status,
                        "how to read it": "'ok' means both records had usable disappearance dates.",
                    },
                    {
                        "operation": "Date gap",
                        "result": (
                            f"{temporal.distance_days} day(s)"
                            if temporal.distance_days is not None
                            else "n/a"
                        ),
                        "how to read it": "Absolute number of days between the two disappearance dates.",
                    },
                    {
                        "operation": "Time score",
                        "result": _score_label(temporal.score),
                        "how to read it": "Score decreases as the date gap increases.",
                    },
                    {
                        "operation": "Time explanation",
                        "result": temporal.explanation,
                        "how to read it": "Plain-language summary of the time comparison.",
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    if pair.matched_rules:
        st.markdown("**Blocking rules that surfaced this pair**")
        st.write(", ".join(pair.matched_rules))


def _round_optional(value: float | None) -> float | None:
    """Round a score-like value for display while preserving missing values."""
    if value is None:
        return None
    return round(value, 3)


def _score_label(value: float | None) -> str:
    """Format a score-like value for metric display."""
    return "n/a" if value is None else f"{value:.3f}"


def _display_value(value: object) -> str:
    """Format a record field value for side-by-side display."""
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _progress_range_for_step(step_label: str) -> tuple[int, int]:
    """Map an engine progress step to its portion of the visible progress bar."""
    if "Pair generation" in step_label or "Preparing blocking" in step_label:
        return 5, 25
    if "Candidate pairs" in step_label:
        return 25, 25
    if "Checking cache" in step_label:
        return 25, 40
    if "IPA precomputation" in step_label:
        return 40, 55
    if "Computing scores" in step_label:
        return 55, 90
    if "Updating cache" in step_label:
        return 90, 97
    if "Visible pair selection" in step_label:
        return 5, 20
    return 0, 100


def _render_investigation_mode_description(investigation_mode: str) -> None:
    """Display the plain-language description of the selected investigation mode."""
    if investigation_mode == INVESTIGATION_MODE_SINGLE:
        st.info(
            "Single case matching compares one selected case against the other records. "
            "Use it when the investigator wants to start from a known case and inspect possible related records."
        )
    else:
        st.info(
            "Global matching searches for potential matches across the whole dataset. "
            "Use it when the investigator wants to detect possible links without choosing a starting case."
        )


def _render_analysis_mode_description(analysis_mode: str) -> None:
    """Display the plain-language description of the selected analysis mode."""
    if analysis_mode == "exhaustive":
        st.info(
            "Compares every possible pair. Very useful for small datasets or filter validation, "
            "but it can be slow."
        )
    elif analysis_mode == "standard_blocking":
        st.info(
            "Only compares candidate pairs selected by simple name, date, and location rules. "
            "Recommended for large datasets."
        )
    else:
        st.info(
            "Planned for a future version. It will let the investigator choose custom filtering rules."
        )


def _sync_default_analysis_mode(investigation_mode: str) -> None:
    """Set the default analysis mode when the investigator switches investigation mode."""
    default_label = (
        "Exhaustive / validation"
        if investigation_mode == INVESTIGATION_MODE_SINGLE
        else "Standard blocking"
    )
    if st.session_state.get("_last_investigation_mode") != investigation_mode:
        st.session_state["analysis_mode_label"] = default_label
        st.session_state["_last_investigation_mode"] = investigation_mode


def _available_case_ids(df: pd.DataFrame) -> list[str]:
    """Return sorted non-empty case IDs available in the uploaded dataset."""
    if "case_id" not in df.columns:
        return []
    case_ids = df["case_id"].dropna().astype(str).str.strip()
    return sorted(case_id for case_id in case_ids.unique().tolist() if case_id)
