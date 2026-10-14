"""Static-AST + runtime regression guards for S81 m05 — _derive_summary reroute.

17th-consecutive-sprint RD2 layered-guard contribution. Pins the load-bearing
invariants from PD126:

* ``deepsearch/report/protocol.py`` ``_derive_summary`` function body
  references ``synthesis_telemetry`` (the live-counts substrate populated
  by the CodeAgent finalize node).
* The function body does NOT reference the pre-S77 dead-state paths
  ``reflection_scores``, ``sources_found``, or ``summarize_state_findings``
  inside the synthesized branch. (The import for
  ``summarize_state_findings`` may remain at module scope for other
  consumers but must not feed the synthesized summary path.)
* Runtime: every produced summary string is non-empty and does NOT read
  as the pre-S81 zero-cue falsehood
  (``"0 unique concepts ... from 0 sources. Coverage: 0%"``).
* Hard Rule 10 trivial: no dataclass field DELETION. The fix is purely
  field-read substitution — ``_derive_summary`` switches what state keys
  it reads, but no dataclass schema changes.

These guards fail loudly if a future sprint:

* Reverts the synthesized branch to read ``state['sources_found']`` etc.
* Reintroduces the pre-S81 zero-cue literal output.
* Removes the defensive non-empty fallback.
* Drops a dataclass field that this fix touches (Hard Rule 10 belt-and-
  suspenders).

Tests do not call any LLM, do not hit the network. Pure AST inspection
plus runtime calls on synthetic AgentState instances.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "deepsearch" / "report" / "protocol.py"


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _derive_summary_function_node() -> ast.FunctionDef:
    tree = _module_ast(PROTOCOL_PATH)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_derive_summary":
            return node
    raise AssertionError("_derive_summary function not found in protocol.py.")


# ---------------------------------------------------------------------------
# m05 — function body references synthesis_telemetry.
# ---------------------------------------------------------------------------


def test_derive_summary_reads_synthesis_telemetry_key() -> None:
    """The function body must read ``synthesis_telemetry`` from state."""
    fn = _derive_summary_function_node()
    body_strings = {
        sub.value
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    }
    assert "synthesis_telemetry" in body_strings, (
        "S81 m05 (PD126) — _derive_summary must read state['synthesis_telemetry'] "
        "to emit a truthful summary. Reverting to the pre-S81 state paths "
        "(sources_found / reflection_scores) re-introduces the cosmetic "
        "falsehood that every CodeAgent mission's protocol.summary read "
        "'0 unique concepts ... 0 sources. Coverage: 0%' regardless of "
        "actual outcome."
    )


def test_derive_summary_synthesized_branch_does_not_read_dead_state() -> None:
    """The synthesized branch must NOT read the pre-S77 dead state paths.

    AST guard: the dead-state literals (``sources_found``,
    ``reflection_scores``) must not appear as string constants in the
    function body — they would only be there if the synthesized branch
    were silently reverted.

    Note: ``summarize_state_findings`` may still appear as an import at
    module scope for other consumers; we only check the function body
    here, not the module-level imports.
    """
    fn = _derive_summary_function_node()
    body_strings = {
        sub.value
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    }
    # `sources_found` is the dead state-key literal whose disappearance
    # is the contract.
    assert "sources_found" not in body_strings, (
        "S81 m05 (PD126) — _derive_summary must NOT read state['sources_found'] "
        "(dead post-S77 state path). Reverting regresses the cosmetic falsehood."
    )
    assert "reflection_scores" not in body_strings, (
        "S81 m05 (PD126) — _derive_summary must NOT read "
        "state['reflection_scores'] (dead post-S77 state path)."
    )

    # The function body must not CALL summarize_state_findings either.
    function_calls = {
        getattr(sub.func, "id", None)
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
    }
    assert "summarize_state_findings" not in function_calls, (
        "S81 m05 (PD126) — _derive_summary must NOT call "
        "summarize_state_findings(state) (always returns {} post-S77). "
        "The synthesis_telemetry reroute supersedes this path."
    )


# ---------------------------------------------------------------------------
# m05 — runtime behavior.
# ---------------------------------------------------------------------------


def test_derive_summary_runtime_reads_top_level_synthesis_telemetry() -> None:
    """Runtime: synthesis_telemetry under state["synthesis_telemetry"] feeds the summary."""
    from deepsearch.report.protocol import _derive_summary

    state = {
        "mission_context": {},
        "synthesis_telemetry": {
            "sources_collected": 25,
            "citation_telemetry": {"citations_extracted": 60},
            "deliverable_coverage": {
                "satisfied": ["A", "B", "C"],
                "missing": [],
                "shortfall_score": 0.0,
            },
        },
    }
    summary = _derive_summary(state)
    assert "60 citations" in summary
    assert "25 sources" in summary
    assert "3/3 satisfied" in summary
    assert "100%" in summary


def test_derive_summary_runtime_reads_nested_telemetry_key() -> None:
    """Runtime: synthesis_telemetry nested under state["telemetry"] also works."""
    from deepsearch.report.protocol import _derive_summary

    state = {
        "mission_context": {},
        "telemetry": {
            "synthesis_telemetry": {
                "sources_collected": 7,
                "citation_telemetry": {"citations_extracted": 14},
            }
        },
    }
    summary = _derive_summary(state)
    assert "14 citations" in summary
    assert "7 sources" in summary


def test_derive_summary_runtime_empty_state_returns_non_empty_string() -> None:
    """Runtime: empty state returns the defensive non-empty string, NOT zero-cue."""
    from deepsearch.report.protocol import _derive_summary

    summary = _derive_summary({})
    assert summary, "Defensive contract — summary must always be non-empty."
    # Pre-S81 emitted "0 unique concepts ... 0 sources. Coverage: 0%" — that
    # exact zero-cue falsehood must not appear.
    assert "0 unique concepts" not in summary, (
        "S81 m05 (PD126) — the pre-S81 zero-cue falsehood "
        "(\"0 unique concepts\") must not appear in the empty-state summary."
    )
    assert "Coverage: 0%" not in summary, (
        "S81 m05 (PD126) — the pre-S81 zero-cue falsehood "
        "(\"Coverage: 0%\") must not appear in the empty-state summary."
    )


def test_derive_summary_runtime_explicit_context_summary_still_preferred() -> None:
    """Runtime: explicit mission_context.summary (≥20 chars) still wins."""
    from deepsearch.report.protocol import _derive_summary

    explicit = "This research investigates retrieval-augmented generation pipelines in depth."
    state = {
        "mission_context": {"summary": explicit},
        # Telemetry also populated — must be ignored when explicit summary present.
        "synthesis_telemetry": {
            "sources_collected": 100,
            "citation_telemetry": {"citations_extracted": 200},
        },
    }
    summary = _derive_summary(state)
    assert summary == explicit


def test_derive_summary_runtime_malformed_telemetry_does_not_crash() -> None:
    """Runtime: non-Mapping telemetry / non-int counters / non-list coverage
    collapse to defensive defaults without raising."""
    from deepsearch.report.protocol import _derive_summary

    state = {
        "synthesis_telemetry": {
            "sources_collected": [None, None],  # not an int
            "citation_telemetry": [],  # not a Mapping
            "deliverable_coverage": "garbage",  # not a Mapping
        },
    }
    # Must not raise.
    summary = _derive_summary(state)
    assert summary


# ---------------------------------------------------------------------------
# Hard Rule 10 trivial — no dataclass field DELETION.
# ---------------------------------------------------------------------------


def test_protocol_pydantic_schema_carries_summary_field() -> None:
    """``MissionProtocolComplete.summary`` field still exists post-m05.

    The m05 fix is field-READ substitution only — it changes which state
    keys _derive_summary consults, but does NOT touch the
    MissionProtocolComplete pydantic schema. Hard Rule 10 trivial guard
    pins this contract.
    """
    from deepsearch.report.protocol import MissionProtocolComplete

    assert "summary" in MissionProtocolComplete.model_fields, (
        "S81 m05 (PD126) Hard Rule 10 — MissionProtocolComplete.summary "
        "field must still exist post-m05. The fix is field-READ "
        "substitution; field DELETION is forbidden."
    )
