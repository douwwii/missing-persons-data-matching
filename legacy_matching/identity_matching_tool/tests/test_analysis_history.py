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
