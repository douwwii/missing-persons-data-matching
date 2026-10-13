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
