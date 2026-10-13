> Historical document retained for provenance. Paths and commands below describe the original repository; use the root README for current instructions.

# Visible Results Refinement Design

## Context

The Streamlit app currently runs one matching analysis and stores the displayed result in
`st.session_state["analysis_results"]`. The matching engine generates candidate pairs,
scores them, applies `min_score_threshold` and `max_results`, then returns only the retained
results to the UI.

The requested feature lets an investigator start from the currently displayed result rows,
change the scoring filters or parameters, and run a new analysis only on those visible pairs.
The investigator must also be able to navigate back to previous refinement steps until they
start a completely new analysis.

## Chosen Approach

Use a temporary in-memory analysis history in Streamlit session state.

Each normal analysis creates a new history with one step. Each refinement step takes the
currently visible `PairMatchingResult` rows, extracts their `(id_A, id_B)` pairs, recalculates
scores for exactly those pairs with the selected refinement configuration, applies the usual
threshold and maximum result limit, and appends the new result as the next history step.

This avoids persistent storage, keeps behavior aligned with what the user sees, and reuses the
existing cache-aware pair scoring path where possible.

## Functional Behavior

- A normal analysis replaces any previous refinement history.
- A refinement is available only when the current result has at least one displayed pair.
- The refinement scope is exactly the current `MatchingRunResult.results` list.
- Refinement recalculates the score for those visible pairs with the new filter configuration.
- Refinement does not regenerate candidate pairs through exhaustive mode or blocking.
- Refinement results remain sorted by final score descending.
- `min_score_threshold` and `max_results` are applied after recalculation.
- Users can navigate back to previous analysis steps from the same history.
- If the user goes back and launches a new refinement, later forward steps are discarded.
- Uploading a new file or confirming a completely new normal analysis clears the refinement
  history.

## UI Design

The existing analysis result page gains a compact refinement section above the result table.

The section contains:

- A small history indicator, for example `Analysis step 2 of 3`.
- Navigation buttons for previous and next steps when available.
- A `Refine current results` expander.
- The same practical filter controls as the main configuration: name, phonetic, location,
  time, and their relevant parameters.
- A button such as `Run refinement on displayed results`.

The refinement UI should not expose analysis mode choices, because the pair scope is already
fixed by the visible results. The original investigation mode can be preserved for metadata,
but candidate generation is bypassed.

Pair detail navigation continues to work on the current step only. Returning from detail goes
back to the current step result table.

## Engine Design

Add an engine entry point that scores explicit pairs:

`run_matching_on_pair_results(records, source_results, analysis_config, parallel_config, progress_callback)`

Responsibilities:

- Build a record lookup by `case_id`.
- Extract unique visible pair IDs from `source_results`.
- Ignore pairs whose records are no longer available.
- Score those explicit pairs through the existing `_score_pairs` helper.
- Apply `min_score_threshold` and `max_results`.
- Return a `MatchingRunResult` with:
  - `analysis_mode="refinement"`
  - `investigation_mode` copied from the refinement config
  - `theoretical_pairs_count` set to the number of source visible pairs
  - `compared_pairs_count` set to the number of valid explicit pairs scored
  - `retained_results_count` set to displayed retained results
  - normal cache and parallel statistics
  - an explanation stating that only displayed pairs from the previous step were rescored

The existing private `_score_pairs` path is already suitable because it accepts pair ID tuples
and can use the component cache.

## Session State Design

Use these keys:

- `analysis_history`: list of history steps.
- `analysis_history_index`: current zero-based step index.
- `analysis_results`: kept as the current step result for compatibility with existing UI flow.

Each history step stores:

- `result`: `MatchingRunResult`
- `config`: `AnalysisConfig`
- `kind`: `"initial"` or `"refinement"`

On normal run:

- Set `analysis_history` to one initial step.
- Set `analysis_history_index` to `0`.
- Set `analysis_results` to the initial result.

On refinement run:

- Take the current history step result.
- Recalculate visible pairs.
- Drop any steps after the current index.
- Append the refinement step.
- Move `analysis_history_index` to the new step.
- Set `analysis_results` to the new current result.

On history navigation:

- Update `analysis_history_index`.
- Set `analysis_results` to the selected step result.
- Clear detail-page UI state.

## Testing

Add focused tests for the matching engine:

- Refinement scores only the visible source pairs.
- Refinement applies `max_results` and sorting after recalculation.
- Refinement handles missing records by skipping invalid pairs.

Add lightweight UI-state helper tests if history manipulation is extracted into pure helper
functions. Streamlit rendering itself does not need full browser automation for this feature.

## Non-Goals

- Persisting refinement history across app restarts.
- Running refinement on hidden, truncated, or non-displayed pairs.
- Creating new pair candidates from the set of records involved in visible results.
- Changing the existing SQLite component cache policy.
- Replacing the current normal analysis workflow.
