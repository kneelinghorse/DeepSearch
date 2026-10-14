"""LangGraph workflow construction (post-S64 entry-flip + post-S75 deletion).

The deployed workflow is a single node — ``codeagent_finalize_node`` — wired
to ``END``. The CodeAgent (smolagents 1.24.0) owns retrieval directly via its
live tools (``web_search``, ``source_fetch``, ``url_liveness``,
``scratchpad_*``). Pre-S64 stages 1–8 (``research_node`` retrieval +
admission, ``cluster_findings``, ``reflection`` + the
research/cluster/reflection loop) were bypassed at the S64 entry-point flip
and their source files removed at S75 m01.
"""

from __future__ import annotations

from typing import Any

from .codeagent_finalize_node import codeagent_finalize_node
from .state import AgentState


def build_workflow(*, checkpointer: Any | None = None) -> Any:
    """Compile and return the LangGraph application."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "LangGraph is not installed. Install the project dependencies "
            "to build the workflow."
        ) from exc

    workflow = StateGraph(AgentState)
    workflow.add_node("finalize", codeagent_finalize_node)
    workflow.set_entry_point("finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile(checkpointer=checkpointer)


__all__ = ["build_workflow"]
