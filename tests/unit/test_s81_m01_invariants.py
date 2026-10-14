"""Static-AST regression guards for S81 m01 — Coverage Gate Truthfulness.

17th-consecutive-sprint RD2 layered-guard contribution. Pins the load-bearing
invariants from PD122 + PD116a:

* ``deepsearch/agent/contract_to_prompt.py`` imports the ``logging`` stdlib
  module and binds a module-level ``logger`` — m01.a's INFO log on
  success_criteria-fallback substitution is observable via the standard
  logging surface.
* ``_resolve_deliverables`` contains a literal reference to
  ``success_criteria`` (the fallback key) so the additive path cannot be
  silently regressed back to the pre-S81 empty-fallback behavior.
* ``deepsearch/agent/tools/completeness_check.py`` declares a
  ``_PARENTHETICAL_RE`` module constant — the regex the m01.b fix uses to
  strip parenthetical clauses before boundary-word truncation.
* ``_canonical_heading`` references ``_PARENTHETICAL_RE`` so the strip step
  cannot be removed without the AST guard catching it.

These guards fail loudly if a future sprint:

* Removes the ``import logging`` or ``logger`` binding from
  ``contract_to_prompt`` (which would silently delete the observability
  side-effect of m01.a even if the fallback path still returns a list).
* Reverts ``_resolve_deliverables`` to the pre-S81 short-form that does
  not look at ``success_criteria``.
* Deletes the ``_PARENTHETICAL_RE`` constant from
  ``completeness_check.py``.
* Removes the parens-strip call from ``_canonical_heading``.

Tests do not call any LLM, do not hit the network, and exercise only AST
inspection plus runtime calls on the helpers.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPSEARCH_ROOT = REPO_ROOT / "deepsearch"
CONTRACT_TO_PROMPT_PATH = DEEPSEARCH_ROOT / "agent" / "contract_to_prompt.py"
COMPLETENESS_CHECK_PATH = DEEPSEARCH_ROOT / "agent" / "tools" / "completeness_check.py"


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# m01.a — contract_to_prompt logging surface + success_criteria fallback.
# ---------------------------------------------------------------------------


def test_contract_to_prompt_imports_logging() -> None:
    """``contract_to_prompt.py`` must import the ``logging`` stdlib module.

    The m01.a fallback emits an INFO log on substitution; without the
    import the side-effect is gone and operators lose the cue that a
    mission supplied success_criteria-as-deliverables.
    """
    tree = _module_ast(CONTRACT_TO_PROMPT_PATH)
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
    assert "logging" in imported, (
        "deepsearch/agent/contract_to_prompt.py must import `logging` "
        "(S81 m01.a PD122) — required for the success_criteria fallback "
        "INFO log."
    )


def test_contract_to_prompt_binds_module_logger() -> None:
    """A top-level ``logger = logging.getLogger(__name__)`` binding must exist."""
    tree = _module_ast(CONTRACT_TO_PROMPT_PATH)
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id == "logger"):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "getLogger"
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "logging"
        ):
            return
    raise AssertionError(
        "deepsearch/agent/contract_to_prompt.py must bind "
        "`logger = logging.getLogger(__name__)` at module scope (S81 m01.a)."
    )


def test_resolve_deliverables_falls_back_to_success_criteria() -> None:
    """The function body must reference ``success_criteria`` literal."""
    tree = _module_ast(CONTRACT_TO_PROMPT_PATH)
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_resolve_deliverables"
        ):
            body_strings = {
                sub.value
                for sub in ast.walk(node)
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
            }
            assert "success_criteria" in body_strings, (
                "S81 m01.a (PD122) requires _resolve_deliverables to fall "
                "back to context['success_criteria'] when explicit "
                "deliverables are absent. Removing this literal regresses "
                "the Alexander-mission coverage-gate blindness."
            )
            assert "successCriteria" in body_strings, (
                "S81 m01.a (PD122) must also support the camelCase "
                "`successCriteria` key for parity with _resolve_objectives."
            )
            return
    raise AssertionError(
        "_resolve_deliverables function not found in "
        "deepsearch/agent/contract_to_prompt.py."
    )


def test_resolve_deliverables_runtime_explicit_takes_precedence() -> None:
    """Runtime: when BOTH explicit deliverables AND success_criteria exist,
    the explicit path wins."""
    from deepsearch.agent.contract_to_prompt import resolve_deliverables

    state = {
        "mission_id": "X",
        "mission_objectives": [],
        "mission_context": {
            "deliverables": ["Explicit A", "Explicit B"],
            "success_criteria": ["DO NOT USE"],
        },
    }
    assert resolve_deliverables(state) == ["Explicit A", "Explicit B"]


def test_resolve_deliverables_runtime_fallback_path() -> None:
    """Runtime: missing deliverables triggers the success_criteria fallback."""
    from deepsearch.agent.contract_to_prompt import resolve_deliverables

    state = {
        "mission_id": "X",
        "mission_objectives": [],
        "mission_context": {
            "success_criteria": ["Criterion 1", "Criterion 2"],
        },
    }
    assert resolve_deliverables(state) == ["Criterion 1", "Criterion 2"]


# ---------------------------------------------------------------------------
# m01.b — completeness_check parens-handling fix.
# ---------------------------------------------------------------------------


def test_completeness_check_declares_parenthetical_regex_constant() -> None:
    """``_PARENTHETICAL_RE`` must exist at module scope."""
    tree = _module_ast(COMPLETENESS_CHECK_PATH)
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "_PARENTHETICAL_RE":
            return
    raise AssertionError(
        "deepsearch/agent/tools/completeness_check.py must declare "
        "`_PARENTHETICAL_RE` at module scope (S81 m01.b PD116a). The "
        "constant is the regex used to strip parenthetical clauses "
        "BEFORE boundary-word truncation in _canonical_heading."
    )


def test_canonical_heading_invokes_parenthetical_strip() -> None:
    """The function body must call ``_PARENTHETICAL_RE.sub(...)``."""
    tree = _module_ast(COMPLETENESS_CHECK_PATH)
    for node in tree.body:
        if not (
            isinstance(node, ast.FunctionDef) and node.name == "_canonical_heading"
        ):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "sub"
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "_PARENTHETICAL_RE"
            ):
                return
        raise AssertionError(
            "_canonical_heading must invoke `_PARENTHETICAL_RE.sub(...)` "
            "to strip parenthetical clauses before boundary-word "
            "truncation (S81 m01.b PD116a)."
        )
    raise AssertionError(
        "_canonical_heading function not found in completeness_check.py."
    )


def test_canonical_heading_runtime_strips_parens_pre_truncation() -> None:
    """Runtime: the PLUGIN-BACKEND-AUTH-01 verbatim bullet canonicalises
    to ``Pattern catalogue`` (not the orphaned ``Pattern catalogue (...)``).
    """
    from deepsearch.agent.tools.completeness_check import _canonical_heading

    bullet = (
        "Pattern catalogue (persistent-account / ephemeral-session / "
        "hybrid with optional account upgrade)"
    )
    out = _canonical_heading(bullet)
    assert out == "Pattern catalogue", (
        f"Expected 'Pattern catalogue', got {out!r}. The open paren must "
        "be stripped before boundary-word truncation (S81 m01.b PD116a)."
    )
    assert "(" not in out, "No open-paren orphan permitted in canonical heading."
    assert ")" not in out


def test_canonical_heading_runtime_handles_multiple_parens() -> None:
    """Runtime: multiple disjoint parenthetical groups all strip."""
    from deepsearch.agent.tools.completeness_check import _canonical_heading

    out = _canonical_heading("Risk register (top 9) covering (a) cost (b) coverage")
    assert out == "Risk register"


def test_canonical_heading_runtime_no_double_whitespace() -> None:
    """Runtime: paren-strip never leaves doubled whitespace."""
    from deepsearch.agent.tools.completeness_check import _canonical_heading

    out = _canonical_heading("OAuth flows (PKCE / device code) comparison")
    assert "  " not in out
    assert out == "OAuth flows comparison"
