"""Static-AST regression guards for the S80 m02 single-title-source-of-truth.

16th-consecutive-sprint RD2 layered-guard contribution. Pins the load-bearing
invariants from PD107:

* ``deepsearch/agent/state.py`` — ``AgentState`` TypedDict carries a
  ``mission_title: str`` field, and ``empty_state()`` pins
  ``state["mission_title"]`` from the passed ``mission_context["title"]``
  with a ``mission_id`` fallback.
* ``deepsearch/agent/codeagent_finalize_node.py`` — the ``codeagent_finalize_node``
  function reads ``state.get("mission_title")`` for the report-filename / wrapper
  passthrough and does NOT re-derive from ``mission_context["title"]`` /
  ``mission_context["objective"]`` / ``mission_id``.
* ``deepsearch/report/generator.py`` — ``generate_markdown_report`` reads
  ``state.get("mission_title")`` for the wrapper H1 instead of re-resolving
  from ``mission_context``.
* ``deepsearch/report/protocol.py`` — ``_derive_title`` is collapsed to a
  thin reader of ``state["mission_title"]`` with a ``mission_id`` fallback.
  The pre-S80 objective / "Research Mission: <id>" branches are gone.

These guards fail loudly if a future sprint:

* Renames or removes ``mission_title`` from ``AgentState``.
* Stops pinning ``mission_title`` inside ``empty_state``.
* Re-introduces an objective fallback in ``_derive_title`` (the divergence
  vector closed at S80).
* Re-introduces independent ``context["title"] or context["objective"] or mission_id``
  resolution in ``codeagent_finalize_node`` or ``generator``.

Tests do not call any LLM, do not hit the network, and exercise only AST
inspection plus a runtime ``empty_state`` invocation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPSEARCH_ROOT = REPO_ROOT / "deepsearch"
STATE_PATH = DEEPSEARCH_ROOT / "agent" / "state.py"
FINALIZE_NODE_PATH = DEEPSEARCH_ROOT / "agent" / "codeagent_finalize_node.py"
GENERATOR_PATH = DEEPSEARCH_ROOT / "report" / "generator.py"
PROTOCOL_PATH = DEEPSEARCH_ROOT / "report" / "protocol.py"


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# AgentState surface — `mission_title` field declaration.
# ---------------------------------------------------------------------------


def test_agent_state_typeddict_carries_mission_title_field() -> None:
    """The canonical state key MUST be declared on the TypedDict."""
    tree = _module_ast(STATE_PATH)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "AgentState":
            continue
        annotated_fields = {
            stmt.target.id
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        }
        assert "mission_title" in annotated_fields, (
            "AgentState must declare a `mission_title: str` field (S80 m02 "
            "PD107). Removing this field forces every reader back into the "
            "pre-S80 distributed-resolution pattern."
        )
        return
    raise AssertionError(
        "Top-level `class AgentState(TypedDict, total=False):` declaration "
        "not found in deepsearch/agent/state.py — S80 m02 RD2 invariant failure."
    )


# ---------------------------------------------------------------------------
# empty_state — runtime + AST guards on the pin.
# ---------------------------------------------------------------------------


def test_empty_state_pins_mission_title_from_context_title() -> None:
    """When context['title'] is non-empty, mission_title must equal it."""
    from deepsearch.agent.state import empty_state

    state = empty_state(
        mission_id="S80-M02-GUARD",
        mission_context={"title": "Canonical Title", "objective": "ignored"},
    )
    assert state.get("mission_title") == "Canonical Title", (
        "empty_state must pin state['mission_title'] to context['title'] "
        "when present (S80 m02 PD107)."
    )


def test_empty_state_falls_back_to_mission_id_when_context_title_empty() -> None:
    """The post-S80 fallback contract is mission_id (NOT objective)."""
    from deepsearch.agent.state import empty_state

    state = empty_state(
        mission_id="S80-M02-FALLBACK",
        mission_context={"title": "", "objective": "rich objective text"},
    )
    assert state.get("mission_title") == "S80-M02-FALLBACK", (
        "empty_state must fall back to mission_id (not objective) when "
        "context['title'] is empty — S80 m02 PD107 collapses the prior "
        "distributed resolution chain."
    )


def test_empty_state_falls_back_when_no_context_provided() -> None:
    """Defensive — calling empty_state without mission_context still pins."""
    from deepsearch.agent.state import empty_state

    state = empty_state(mission_id="S80-M02-NOCONTEXT")
    assert state.get("mission_title") == "S80-M02-NOCONTEXT"


def test_empty_state_strips_whitespace_only_titles() -> None:
    """Whitespace-only titles must be treated as empty."""
    from deepsearch.agent.state import empty_state

    state = empty_state(
        mission_id="S80-M02-WHITESPACE",
        mission_context={"title": "    "},
    )
    assert state.get("mission_title") == "S80-M02-WHITESPACE"


# ---------------------------------------------------------------------------
# _derive_title — must be a thin reader.
# ---------------------------------------------------------------------------


def _find_function_body_source(path: Path, function_name: str) -> str:
    """Return the source-text of the named function body, *without* the
    docstring node. AST-walks the body and concatenates ``ast.unparse`` of
    each non-docstring statement, so word-presence assertions cannot trip
    on documentation that legitimately mentions the deleted concept.

    Handles both ``def`` and ``async def`` definitions.
    """
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ):
            statements = list(node.body)
            if (
                statements
                and isinstance(statements[0], ast.Expr)
                and isinstance(statements[0].value, ast.Constant)
                and isinstance(statements[0].value.value, str)
            ):
                statements = statements[1:]
            return "\n".join(ast.unparse(stmt) for stmt in statements)
    raise AssertionError(
        f"Function {function_name!r} not found in {path}"
    )


def test_derive_title_no_longer_references_objective() -> None:
    """The pre-S80 objective fallback branch must be gone.

    The objective fallback was load-bearing for the divergence between the
    wrapper H1 and the protocol .title — re-introducing it regresses PD107.
    """
    body = _find_function_body_source(PROTOCOL_PATH, "_derive_title")
    assert "objective" not in body, (
        "`_derive_title` must not reference 'objective' (S80 m02 PD107 — "
        "the independent fallback resolution that diverged from the wrapper "
        "H1 path was deleted; canonical title comes from "
        "state['mission_title'] only)."
    )


def test_derive_title_reads_canonical_state_key() -> None:
    """The collapsed body must reference state['mission_title']."""
    body = _find_function_body_source(PROTOCOL_PATH, "_derive_title")
    assert "mission_title" in body, (
        "`_derive_title` must read state['mission_title'] (S80 m02 PD107)."
    )


def test_derive_title_no_longer_emits_research_mission_prefix() -> None:
    """Pre-S80 `f"Research Mission: {mission_id}"` branch is gone."""
    body = _find_function_body_source(PROTOCOL_PATH, "_derive_title")
    assert "Research Mission" not in body, (
        "`_derive_title` must not emit the pre-S80 `Research Mission: <id>` "
        "prefix string — the post-S80 fallback returns the bare mission_id."
    )


# ---------------------------------------------------------------------------
# codeagent_finalize_node — must read the canonical key.
# ---------------------------------------------------------------------------


def _normalize_quotes(text: str) -> str:
    """Collapse single + double quotes to one canonical form for substring checks.

    ``ast.unparse`` always emits single-quoted string literals, but the source
    file uses double quotes. Word-presence assertions must be quote-agnostic.
    """
    return text.replace("'", "\"")


def test_codeagent_finalize_node_reads_canonical_state_key() -> None:
    """The finalize node title-resolution site reads state['mission_title']."""
    body = _normalize_quotes(
        _find_function_body_source(FINALIZE_NODE_PATH, "codeagent_finalize_node")
    )
    assert 'state.get("mission_title")' in body, (
        "codeagent_finalize_node must read state['mission_title'] for the "
        "filename / wrapper passthrough (S80 m02 PD107). Reverting to the "
        "pre-S80 `context.get(\"title\") or context.get(\"objective\")` form "
        "regresses the single-source-of-truth contract."
    )


def test_codeagent_finalize_node_no_longer_consults_objective_for_title() -> None:
    """The pre-S80 objective-as-title fallback at line 100 is gone."""
    body = _normalize_quotes(
        _find_function_body_source(FINALIZE_NODE_PATH, "codeagent_finalize_node")
    )
    # The exact pre-S80 line shape must not reappear.
    assert 'context.get("title") or context.get("objective")' not in body, (
        "codeagent_finalize_node must not re-introduce the pre-S80 "
        "`context.get(\"title\") or context.get(\"objective\")` chain "
        "(S80 m02 PD107)."
    )


# ---------------------------------------------------------------------------
# generate_markdown_report — must read the canonical key.
# ---------------------------------------------------------------------------


def test_generate_markdown_report_reads_canonical_state_key() -> None:
    """The wrapper-H1 generator reads state['mission_title']."""
    body = _normalize_quotes(
        _find_function_body_source(GENERATOR_PATH, "generate_markdown_report")
    )
    assert 'state.get("mission_title")' in body, (
        "generate_markdown_report must read state['mission_title'] "
        "(S80 m02 PD107). The wrapper H1 and the protocol .title field must "
        "agree on the same canonical string."
    )


def test_generate_markdown_report_no_longer_consults_objective_for_title() -> None:
    """The pre-S80 objective-as-title fallback at line 48 is gone."""
    body = _normalize_quotes(
        _find_function_body_source(GENERATOR_PATH, "generate_markdown_report")
    )
    assert 'context.get("title") or context.get("objective")' not in body, (
        "generate_markdown_report must not re-introduce the pre-S80 "
        "`context.get(\"title\") or context.get(\"objective\")` chain "
        "(S80 m02 PD107)."
    )
