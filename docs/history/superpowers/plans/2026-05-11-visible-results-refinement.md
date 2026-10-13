> Historical document retained for provenance. Paths and commands below describe the original repository; use the root README for current instructions.

# Visible Results Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let investigators rerun scoring on the currently displayed result pairs with a new filter configuration, while navigating previous refinement steps in memory.

**Architecture:** Add a matching-engine entry point for explicit visible pair rescoring, then add pure session-history helpers that Streamlit can call. The UI will keep the existing normal analysis flow, but after every run it will maintain `analysis_history`, render history navigation, and offer a compact refinement form above the result table.

**Tech Stack:** Python dataclasses, Streamlit session state, existing matching engine and pytest suite.

---

## File Structure

- Modify `src/matching/matching_engine.py`: add `run_matching_on_pair_results()` and a small pair-deduplication helper.
- Modify `tests/test_matching_engine.py`: add TDD coverage for visible-pair refinement behavior.
- Create `src/ui/analysis_history.py`: pure helpers for initial history creation, refinement append, branch truncation, navigation, and current result lookup.
- Create `tests/test_analysis_history.py`: TDD coverage for history helpers without importing Streamlit.
- Modify `src/ui/configuration_panel.py`: wire history helpers, render navigation, render refinement controls, and call the new engine function.
- Modify `src/ui/upload_section.py`: clear refinement history when a new dataset is loaded.

---

### Task 1: Matching Engine Explicit Pair Refinement

**Files:**
- Modify: `src/matching/matching_engine.py`
- Test: `tests/test_matching_engine.py`

- [ ] **Step 1: Write failing tests for explicit visible-pair refinement**

Append these imports and tests to `tests/test_matching_engine.py`.

```python
from src.models import PairMatchingResult, ParallelConfig
from src.matching.matching_engine import run_matching_on_pair_results


def source_pair(id_a: str, id_b: str) -> PairMatchingResult:
    record_by_id = {record.case_id: record for record in records()}
    left = record_by_id[id_a]
    right = record_by_id[id_b]
    return PairMatchingResult(
        id_A=id_a,
        id_B=id_b,
        name_A=left.full_name_raw,
        name_B=right.full_name_raw,
        score_final=0.0,
    )


def test_refinement_scores_only_visible_source_pairs() -> None:
    source_results = [source_pair("A", "B"), source_pair("C", "D")]

    result = run_matching_on_pair_results(
        records(),
        source_results,
        config("exhaustive"),
        ParallelConfig(enable_parallelism=False),
    )

    compared_pairs = {frozenset((pair.id_A, pair.id_B)) for pair in result.results}
    assert result.status == "ok"
    assert result.analysis_mode == "refinement"
    assert result.theoretical_pairs_count == 2
    assert result.compared_pairs_count == 2
    assert compared_pairs <= {frozenset(("A", "B")), frozenset(("C", "D"))}


def test_refinement_applies_sorting_and_max_results_after_recalculation() -> None:
    source_results = [source_pair("C", "D"), source_pair("A", "B")]

    result = run_matching_on_pair_results(
        records(),
        source_results,
        config("exhaustive", max_results=1),
        ParallelConfig(enable_parallelism=False),
    )

    assert len(result.results) == 1
    assert {result.results[0].id_A, result.results[0].id_B} == {"A", "B"}
    assert result.retained_results_count == 1


def test_refinement_skips_pairs_with_missing_records() -> None:
    source_results = [source_pair("A", "B"), PairMatchingResult("A", "Z", "juan pablo payar", "missing", 0.0)]

    result = run_matching_on_pair_results(
        records(),
        source_results,
        config("exhaustive"),
        ParallelConfig(enable_parallelism=False),
    )

    assert result.theoretical_pairs_count == 2
    assert result.compared_pairs_count == 1
    assert all("Z" not in {pair.id_A, pair.id_B} for pair in result.results)
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_matching_engine.py -q
```

Expected: import failure because `run_matching_on_pair_results` does not exist.

- [ ] **Step 3: Implement explicit-pair refinement in the engine**

In `src/matching/matching_engine.py`, add this public function after `run_standard_blocking_matching()`.

```python
def run_matching_on_pair_results(
    records: list[MissingPersonRecord],
    source_results: list[PairMatchingResult],
    analysis_config: AnalysisConfig,
    parallel_config: ParallelConfig | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> MatchingRunResult:
    """Rescore exactly the visible pairs from a previous result step."""
    started_at = perf_counter()
    records_by_id = {record.case_id: record for record in records}
    source_pairs = _unique_valid_pair_ids(source_results, records_by_id)
    if progress_callback is not None:
        progress_callback("Visible pair selection", len(source_pairs), max(len(source_results), 1))

    threshold = float(analysis_config.filter_parameters.get("min_score_threshold", 0.0))
    max_results = int(analysis_config.filter_parameters.get("max_results", 50))
    parallel_config = _effective_parallel_config(parallel_config)

    scoring_started_at = perf_counter()
    pair_results = _score_pairs(
        records,
        source_pairs,
        analysis_config,
        parallel_config,
        use_cache=True,
        progress_callback=progress_callback,
    )
    scoring_elapsed = perf_counter() - scoring_started_at

    retained = [
        pair_result
        for pair_result in pair_results
        if pair_result.score_final is not None and pair_result.score_final >= threshold
    ]
    retained.sort(key=lambda result: result.score_final or 0.0, reverse=True)
    retained = retained[:max_results]
    elapsed = perf_counter() - started_at
    stats = _parallel_stats(len(source_pairs), pair_results, parallel_config, scoring_elapsed)

    return MatchingRunResult(
        status="ok",
        analysis_mode="refinement",
        investigation_mode=analysis_config.investigation_mode,
        results=retained,
        total_records=len(records),
        theoretical_pairs_count=len(source_results),
        compared_pairs_count=len(source_pairs),
        retained_results_count=len(retained),
        elapsed_time=elapsed,
        **stats,
        explanation=(
            "Refinement mode: only potential matches displayed in the previous analysis step "
            "were rescored with the selected filters."
        ),
    )
```

Add this private helper before `_pairs_for_config()`.

```python
def _unique_valid_pair_ids(
    source_results: list[PairMatchingResult],
    records_by_id: dict[str, MissingPersonRecord],
) -> list[tuple[str, str]]:
    """Return unique source pair IDs whose records are still available."""
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for result in source_results:
        if result.id_A not in records_by_id or result.id_B not in records_by_id:
            continue
        left, right = sorted((str(result.id_A), str(result.id_B)))
        key = (left, right)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((result.id_A, result.id_B))
    return pairs
```

- [ ] **Step 4: Run the engine tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_matching_engine.py -q
```

Expected: all tests in `tests/test_matching_engine.py` pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/matching/matching_engine.py tests/test_matching_engine.py
git commit -m "Add visible pair refinement engine"
```

---

### Task 2: Pure Analysis History Helpers

**Files:**
- Create: `src/ui/analysis_history.py`
- Test: `tests/test_analysis_history.py`

- [ ] **Step 1: Write failing tests for history behavior**

Create `tests/test_analysis_history.py`.

```python
from __future__ import annotations

from src.models import AnalysisConfig, MatchingRunResult
from src.ui.analysis_history import (
    append_refinement_step,
    current_history_result,
    initialize_analysis_history,
    navigate_history,
)


def result(label: str) -> MatchingRunResult:
    return MatchingRunResult(status="ok", analysis_mode=label, investigation_mode="global")


def config() -> AnalysisConfig:
    return AnalysisConfig(investigation_mode="global", analysis_mode="exhaustive")


def test_initial_history_replaces_existing_steps() -> None:
    state = {"analysis_history": [{"result": result("old"), "config": config(), "kind": "initial"}]}

    initialize_analysis_history(state, result("new"), config())

    assert len(state["analysis_history"]) == 1
    assert state["analysis_history_index"] == 0
    assert state["analysis_results"].analysis_mode == "new"
    assert state["analysis_history"][0]["kind"] == "initial"


def test_append_refinement_truncates_forward_history() -> None:
    state = {}
    initialize_analysis_history(state, result("initial"), config())
    append_refinement_step(state, result("step-1"), config())
    navigate_history(state, -1)

    append_refinement_step(state, result("replacement"), config())

    assert [step["result"].analysis_mode for step in state["analysis_history"]] == ["initial", "replacement"]
    assert state["analysis_history_index"] == 1
    assert state["analysis_results"].analysis_mode == "replacement"


def test_navigation_updates_current_result_and_clears_detail_state() -> None:
    state = {"open_match_detail_index": 2, "scroll_to_detail_top": True}
    initialize_analysis_history(state, result("initial"), config())
    append_refinement_step(state, result("step-1"), config())

    navigate_history(state, -1)

    assert state["analysis_history_index"] == 0
    assert current_history_result(state).analysis_mode == "initial"
    assert state["analysis_results"].analysis_mode == "initial"
    assert "open_match_detail_index" not in state
    assert "scroll_to_detail_top" not in state
```

- [ ] **Step 2: Run the history tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_analysis_history.py -q
```

Expected: import failure because `src.ui.analysis_history` does not exist.

- [ ] **Step 3: Implement history helpers**

Create `src/ui/analysis_history.py`.

```python
from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from src.models import AnalysisConfig, MatchingRunResult

HistoryState = MutableMapping[str, Any]


def initialize_analysis_history(
    state: HistoryState,
    result: MatchingRunResult,
    config: AnalysisConfig,
) -> None:
    """Replace any previous analysis history with one initial step."""
    state["analysis_history"] = [{"result": result, "config": config, "kind": "initial"}]
    state["analysis_history_index"] = 0
    state["analysis_results"] = result
    _clear_detail_state(state)


def append_refinement_step(
    state: HistoryState,
    result: MatchingRunResult,
    config: AnalysisConfig,
) -> None:
    """Append a refinement step after the current history index."""
    history = list(state.get("analysis_history", []))
    index = int(state.get("analysis_history_index", len(history) - 1))
    if history:
        history = history[: index + 1]
    history.append({"result": result, "config": config, "kind": "refinement"})
    state["analysis_history"] = history
    state["analysis_history_index"] = len(history) - 1
    state["analysis_results"] = result
    _clear_detail_state(state)


def navigate_history(state: HistoryState, offset: int) -> None:
    """Move through analysis history by an offset when possible."""
    history = list(state.get("analysis_history", []))
    if not history:
        return
    current = int(state.get("analysis_history_index", 0))
    next_index = min(max(current + offset, 0), len(history) - 1)
    state["analysis_history_index"] = next_index
    state["analysis_results"] = history[next_index]["result"]
    _clear_detail_state(state)


def current_history_result(state: HistoryState) -> MatchingRunResult | None:
    """Return the current history result when history exists."""
    history = list(state.get("analysis_history", []))
    if not history:
        return state.get("analysis_results")
    index = min(max(int(state.get("analysis_history_index", 0)), 0), len(history) - 1)
    return history[index]["result"]


def _clear_detail_state(state: HistoryState) -> None:
    """Clear UI-only state tied to a specific result row."""
    state.pop("open_match_detail_index", None)
    state.pop("scroll_to_results_table", None)
    state.pop("scroll_to_detail_top", None)
```

- [ ] **Step 4: Run history tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_analysis_history.py -q
```

Expected: all tests in `tests/test_analysis_history.py` pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/ui/analysis_history.py tests/test_analysis_history.py
git commit -m "Add analysis refinement history helpers"
```

---

### Task 3: Streamlit Integration

**Files:**
- Modify: `src/ui/configuration_panel.py`
- Modify: `src/ui/upload_section.py`

- [ ] **Step 1: Update imports**

In `src/ui/configuration_panel.py`, change the matching import and add history imports.

```python
from src.matching.matching_engine import run_matching, run_matching_on_pair_results
from src.ui.analysis_history import append_refinement_step, initialize_analysis_history, navigate_history
```

- [ ] **Step 2: Store normal runs in history**

In `render_analysis_launcher()`, replace the direct assignment to `analysis_results` inside the `Run analysis` spinner.

```python
                result = run_matching(
                    records,
                    config,
                    blocking_config,
                    parallel_config,
                    progress_callback=update_progress,
                )
                initialize_analysis_history(st.session_state, result, config)
```

- [ ] **Step 3: Clear history when confirming a new configuration**

In the `Confirm configuration` button block, after clearing `analysis_results`, add:

```python
        st.session_state.pop("analysis_history", None)
        st.session_state.pop("analysis_history_index", None)
```

- [ ] **Step 4: Render history navigation before metrics**

In `_render_matching_result()`, after the detail-page guard and before metric columns, call a new helper.

```python
    _render_analysis_history_navigation()
```

Add this helper near `_render_matching_result()`.

```python
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
```

- [ ] **Step 5: Render refinement controls above result rows**

In `_render_matching_result()`, after the performance details expander and before `rows = _result_rows(result)`, add:

```python
    _render_refinement_panel(result, records)
```

Add these helpers before `_result_rows()`.

```python
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
        if st.button(
            "Run refinement on displayed results", type="primary", key="run_visible_results_refinement"
        ):
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
        parameters["geographic_distance_min_km"] = float(
            base_parameters.get("geographic_distance_min_km", 5.0)
        )
        parameters["geographic_distance_max_km"] = float(
            base_parameters.get("geographic_distance_max_km", 50.0)
        )

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
```

- [ ] **Step 6: Extend progress mapping for refinement pair selection**

In `_progress_range_for_step()`, add this branch before the final `return 0, 100`.

```python
    if "Visible pair selection" in step_label:
        return 5, 20
```

- [ ] **Step 7: Clear history when uploading a new dataset**

In `src/ui/upload_section.py`, add these pops near the existing `analysis_results` cleanup.

```python
    st.session_state.pop("analysis_history", None)
    st.session_state.pop("analysis_history_index", None)
```

- [ ] **Step 8: Run focused tests**

Run:

```powershell
python -m pytest tests/test_matching_engine.py tests/test_analysis_history.py -q
```

Expected: all focused tests pass.

- [ ] **Step 9: Commit Task 3**

```powershell
git add src/ui/configuration_panel.py src/ui/upload_section.py
git commit -m "Add visible result refinement UI"
```

---

### Task 4: Full Verification

**Files:**
- No code changes expected.

- [ ] **Step 1: Run the full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Inspect git status**

Run:

```powershell
git status --short
```

Expected: only pre-existing unrelated working tree changes remain, or a clean tree if those were resolved separately.

- [ ] **Step 3: Manual Streamlit smoke test**

Run:

```powershell
python -m streamlit run app.py
```

Expected manual flow:

- Load `data/sample_clean_missing_persons.csv`.
- Run a normal analysis.
- Confirm the result page shows `Analysis step 1 of 1`.
- Open `Refine current results`.
- Enable `Name` and `Time`, then run refinement.
- Confirm the result page shows `Analysis step 2 of 2`.
- Use `Previous analysis` and `Next analysis` to navigate.
- Open a match detail from step 2 and return to results.

- [ ] **Step 4: Final status summary**

Report implemented files, verification commands, and any unrelated pre-existing changes left untouched.
