from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import INVESTIGATION_MODE_GLOBAL, INVESTIGATION_MODE_SINGLE
from src.matching.matching_engine import run_matching
from src.models import AnalysisConfig, BlockingConfig, MatchingRunResult, MissingPersonRecord, ParallelConfig


@st.fragment
def render_configuration_panel(
    df: pd.DataFrame,
    records: list[MissingPersonRecord],
    has_coordinates: bool,
    has_dates: bool,
) -> None:
    st.subheader("Etape 4 - Configuration de l'analyse")

    st.markdown("**Mode d'investigation**")
    mode_label = st.radio(
        "Choisir le mode",
        ["Single case matching", "Global matching"],
        horizontal=True,
        key="investigation_mode_label",
    )
    investigation_mode = (
        INVESTIGATION_MODE_SINGLE if mode_label == "Single case matching" else INVESTIGATION_MODE_GLOBAL
    )

    selected_case_id = None
    if investigation_mode == INVESTIGATION_MODE_SINGLE:
        case_ids = _available_case_ids(df)
        selected_case_id = st.selectbox("Dossier cible", case_ids, key="selected_case_id") if case_ids else None
        if not case_ids:
            st.warning("Aucun case_id exploitable n'est disponible pour choisir un dossier cible.")
    else:
        st.info("Ce mode comparera potentiellement beaucoup de paires de dossiers lors des prochaines briques.")

    st.markdown("**Mode d'analyse**")
    analysis_mode_label = st.radio(
        "Choisir le mode d'analyse",
        ["Exhaustif / validation", "Blocking standard", "Personnalise"],
        horizontal=True,
        key="analysis_mode_label",
    )
    analysis_mode = _analysis_mode_from_label(analysis_mode_label)
    _render_analysis_mode_description(analysis_mode)

    st.markdown("**Filtres de similarite a preparer**")
    col_a, col_b = st.columns(2)
    use_name = col_a.checkbox("Nom", value=True, key="filter_name")
    use_phonetic = col_a.checkbox("Phonetique du nom", value=False, key="filter_phonetic")
    use_geographic = col_b.checkbox("Localisation", value=has_coordinates, key="filter_geographic")
    use_temporal = col_b.checkbox("Temporalite", value=has_dates, key="filter_temporal")

    st.markdown("**Parametres des filtres**")
    filter_parameters: dict[str, float | int] = {}
    if use_geographic:
        with st.container(border=True):
            st.caption("Localisation")
            geo_col_a, geo_col_b = st.columns(2)
            filter_parameters["geographic_distance_min_km"] = geo_col_a.number_input(
                "Distance sous laquelle la similarite sera maximale (km)",
                min_value=0.0,
                value=5.0,
                step=1.0,
                key="geographic_distance_min_km",
            )
            filter_parameters["geographic_distance_max_km"] = geo_col_b.number_input(
                "Distance au-dessus de laquelle la similarite sera minimale (km)",
                min_value=0.0,
                value=50.0,
                step=5.0,
                key="geographic_distance_max_km",
            )
    else:
        filter_parameters["geographic_distance_min_km"] = 5.0
        filter_parameters["geographic_distance_max_km"] = 50.0

    if use_temporal:
        with st.container(border=True):
            st.caption("Temporalite")
            filter_parameters["temporal_max_days"] = st.number_input(
                "Nombre de jours au-dela duquel la similarite sera nulle",
                min_value=1,
                value=365,
                step=30,
                key="temporal_max_days",
            )
    else:
        filter_parameters["temporal_max_days"] = 365

    with st.container(border=True):
        st.caption("Parametres generaux")
        filter_parameters["max_results"] = st.number_input(
            "Nombre maximum de correspondances potentielles",
            min_value=1,
            value=50,
            step=10,
            key="max_results",
        )
        filter_parameters["min_score_threshold"] = 0.0

    blocking_config = _render_blocking_parameters(analysis_mode)
    parallel_config = _render_parallel_parameters()

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

    st.subheader("Etape 5 - Validation de la configuration")
    if st.button("Valider la configuration", type="primary"):
        st.session_state["analysis_config"] = config
        st.session_state["blocking_config"] = blocking_config
        st.session_state["parallel_config"] = parallel_config
        st.session_state.pop("analysis_results", None)

    if "analysis_config" in st.session_state:
        render_config_summary(st.session_state["analysis_config"])
        render_analysis_launcher(
            records,
            st.session_state["analysis_config"],
            st.session_state.get("blocking_config"),
            st.session_state.get("parallel_config"),
        )


def render_config_summary(config: AnalysisConfig) -> None:
    mode_label = "Single case matching" if config.investigation_mode == INVESTIGATION_MODE_SINGLE else "Global matching"
    enabled = [name for name, enabled in config.enabled_filters.items() if enabled]
    st.success("Configuration validee.")
    st.write(
        {
            "mode": mode_label,
            "mode_analyse": config.analysis_mode,
            "dossier_cible": config.selected_case_id,
            "filtres_selectionnes": enabled,
            "parametres": config.filter_parameters,
        }
    )


def render_analysis_launcher(
    records: list[MissingPersonRecord],
    config: AnalysisConfig,
    blocking_config: BlockingConfig | None,
    parallel_config: ParallelConfig | None,
) -> None:
    st.subheader("Etape 6 - Analyse")
    st.caption("L'analyse calcule des scores d'aide a l'enquete. Elle ne produit pas de preuve.")

    if not any(config.enabled_filters.values()):
        st.warning("Activez au moins un filtre de similarite avant de lancer l'analyse.")
        return

    if config.investigation_mode == INVESTIGATION_MODE_SINGLE and not config.selected_case_id:
        st.warning("Choisissez un dossier cible avant de lancer l'analyse.")
        return

    if config.investigation_mode == INVESTIGATION_MODE_GLOBAL:
        pair_count = len(records) * (len(records) - 1) // 2
        st.info(f"Mode global : jusqu'a {pair_count} paire(s) de dossiers peuvent etre evaluees localement.")

    if config.analysis_mode == "custom_placeholder":
        st.warning("Le mode personnalise sera ajoute dans une prochaine version.")

    if st.button("Lancer l'analyse", type="primary", key="run_similarity_analysis"):
        try:
            progress = st.progress(0, text="Preparation de l'analyse")
            status_placeholder = st.empty()

            def update_progress(step_label: str, current: int, total: int) -> None:
                total = max(total, 1)
                ratio = min(max(current / total, 0.0), 1.0)
                start, end = _progress_range_for_step(step_label)
                percent = int(start + ratio * (end - start))
                progress.progress(percent, text=f"{step_label} : {current}/{total}")
                status_placeholder.caption(f"{step_label} : {current}/{total}")

            with st.spinner("Calcul des correspondances potentielles..."):
                st.session_state["analysis_results"] = run_matching(
                    records,
                    config,
                    blocking_config,
                    parallel_config,
                    progress_callback=update_progress,
                )
            progress.progress(100, text="Analyse terminee")
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
        _render_matching_result(result)


def _render_matching_result(result: MatchingRunResult) -> None:
    if result.status == "not_implemented":
        st.info(result.explanation)
        return

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Dossiers", result.total_records)
    metric_b.metric("Paires theoriques", result.theoretical_pairs_count)
    metric_c.metric("Paires comparees", result.compared_pairs_count)
    metric_d.metric("Resultats retenus", result.retained_results_count)

    metric_e, metric_f = st.columns(2)
    if result.reduction_percentage is not None:
        metric_e.metric("Reduction", f"{result.reduction_percentage:.1%}")
    metric_f.metric("Temps de calcul", f"{result.elapsed_time:.2f} s")

    if result.blocking_result is not None:
        block_a, block_b, block_c = st.columns(3)
        block_a.metric("Paires gardees par R1", result.blocking_result.r1_count)
        block_b.metric("Paires gardees par R2", result.blocking_result.r2_count)
        block_c.metric("Paires gardees par R4", result.blocking_result.r4_count)

    with st.expander("Details techniques"):
        st.write(result.explanation)
        if result.blocking_result is not None:
            st.write(result.blocking_result.explanation)
            if result.blocking_result.blocking_index_cache_dir:
                st.write(
                    {
                        "cache_buckets_blocking": result.blocking_result.blocking_index_cache_dir,
                        "fichiers": [
                            "metadata.json",
                            "bucket_summary.csv",
                            "r1_initials_buckets.csv",
                            "r2_phonetic_onset_buckets.csv",
                            "r4_geo_time_buckets.csv",
                            "signatures.csv",
                        ],
                    }
                )

    with st.expander("Details de performance"):
        st.write(
            {
                "mode_execution": result.execution_mode,
                "parallelisation_active": result.parallel_enabled,
                "workers": result.num_workers,
                "taille_chunks": result.chunk_size,
                "paires_total": result.total_pairs,
                "paires_calculees": result.computed_pairs,
                "paires_cache": result.cached_pairs,
                "cache_hit_rate": f"{result.cache_hit_rate:.1%}",
                "temps_scores": f"{result.parallel_elapsed_time:.3f} s",
                "temps_total": f"{result.elapsed_time:.3f} s",
            }
        )

    rows = _result_rows(result)
    if not rows:
        st.info("Aucune correspondance potentielle a afficher.")
        return
    st.success(f"{len(rows)} correspondance(s) potentielle(s) affichee(s).")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _result_rows(result: MatchingRunResult) -> list[dict[str, object]]:
    return [
        {
            "id_A": pair.id_A,
            "id_B": pair.id_B,
            "score_final": round(pair.score_final or 0.0, 3),
            "score_nom": _round_optional(pair.score_name),
            "nom_jaro_structured": _round_optional(pair.name_direct_structured_score),
            "nom_jaro_direct": _round_optional(pair.name_direct_full_name_score),
            "nom_jaro_reordered": _round_optional(pair.name_reordered_full_name_score),
            "nom_token_alignment": _round_optional(pair.name_token_alignment_score),
            "nom_best_method": pair.name_best_method,
            "ordre_nom_suspect": pair.name_order_mismatch_suspected,
            "score_phonetique": _round_optional(pair.score_phonetic),
            "statut_phonetique": pair.phonetic_status,
            "score_geographique": _round_optional(pair.score_geographic),
            "score_temporel": _round_optional(pair.score_temporal),
            "name_A": pair.name_A,
            "name_B": pair.name_B,
            "date_diff_days": pair.date_diff_days if pair.date_diff_days is not None else pair.distance_days,
            "geo_distance_km": (
                round(pair.geo_distance_km, 1)
                if pair.geo_distance_km is not None
                else round(pair.distance_km, 1)
                if pair.distance_km is not None
                else None
            ),
            "matched_rules": ", ".join(pair.matched_rules),
            "explanation": pair.explanation,
        }
        for pair in result.results
    ]


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def _progress_range_for_step(step_label: str) -> tuple[int, int]:
    if "Generation" in step_label or "Preparation du blocking" in step_label:
        return 5, 25
    if "Paires candidates" in step_label:
        return 25, 25
    if "Verification du cache" in step_label:
        return 25, 40
    if "Pre-calcul IPA" in step_label:
        return 40, 55
    if "Calcul des scores" in step_label:
        return 55, 90
    if "Mise a jour du cache" in step_label:
        return 90, 97
    return 0, 100


def _render_analysis_mode_description(analysis_mode: str) -> None:
    if analysis_mode == "exhaustive":
        st.info(
            "Compare toutes les paires possibles. Tres utile pour les petites bases ou pour valider les filtres, "
            "mais peut etre lent."
        )
    elif analysis_mode == "standard_blocking":
        st.info(
            "Ne compare que les paires candidates selon des regles simples sur les noms, les dates et la localisation. "
            "Recommande pour les grosses bases."
        )
    else:
        st.info(
            "Mode prevu pour une prochaine version. Il permettra a l'enqueteur de choisir ses propres regles de filtrage."
        )


def _render_blocking_parameters(analysis_mode: str) -> BlockingConfig:
    if analysis_mode != "standard_blocking":
        return BlockingConfig()
    with st.container(border=True):
        st.caption("Parametres du blocking standard")
        col_a, col_b, col_c = st.columns(3)
        date_threshold_name_rules = col_a.number_input(
            "Seuil date R1/R2 (jours)",
            min_value=0,
            value=60,
            step=5,
            key="date_threshold_name_rules",
        )
        geo_threshold_km = col_b.number_input(
            "Seuil distance R4 (km)",
            min_value=0.0,
            value=20.0,
            step=5.0,
            key="geo_threshold_km",
        )
        date_threshold_geo_rule = col_c.number_input(
            "Seuil date R4 (jours)",
            min_value=0,
            value=10,
            step=1,
            key="date_threshold_geo_rule",
        )
    return BlockingConfig(
        date_threshold_name_rules=int(date_threshold_name_rules),
        geo_threshold_km=float(geo_threshold_km),
        date_threshold_geo_rule=int(date_threshold_geo_rule),
    )


def _render_parallel_parameters() -> ParallelConfig:
    default_config = ParallelConfig()
    with st.expander("Performances"):
        st.caption(
            "La parallelisation accelere les grosses analyses, mais peut etre inutile sur les petites bases. "
            "Si vous observez un comportement etrange, repassez temporairement en mode debug."
        )
        enable_parallelism = st.checkbox(
            "Activer la parallelisation",
            value=default_config.enable_parallelism,
            key="enable_parallelism",
        )
        debug_mode = st.checkbox(
            "Mode debug (force un seul worker)",
            value=False,
            key="parallel_debug_mode",
        )
        col_a, col_b = st.columns(2)
        num_workers = col_a.number_input(
            "Nombre de workers",
            min_value=1,
            value=default_config.num_workers,
            step=1,
            key="parallel_num_workers",
        )
        chunk_size = col_b.number_input(
            "Taille des chunks",
            min_value=1,
            value=default_config.chunk_size,
            step=50,
            key="parallel_chunk_size",
        )
    if debug_mode:
        num_workers = 1
    return ParallelConfig(
        enable_parallelism=bool(enable_parallelism),
        num_workers=int(num_workers),
        chunk_size=int(chunk_size),
        debug_mode=bool(debug_mode),
    )


def _analysis_mode_from_label(label: str) -> str:
    if label == "Blocking standard":
        return "standard_blocking"
    if label == "Personnalise":
        return "custom_placeholder"
    return "exhaustive"


def _available_case_ids(df: pd.DataFrame) -> list[str]:
    if "case_id" not in df.columns:
        return []
    case_ids = df["case_id"].dropna().astype(str).str.strip()
    return sorted(case_id for case_id in case_ids.unique().tolist() if case_id)
