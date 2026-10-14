"""Static-AST regression guards for S81 m03 — Critique Output Cap Lift.

17th-consecutive-sprint RD2 layered-guard contribution. Pins the load-bearing
invariants from PD124:

* ``deepsearch/agent/codeagent_runner.py`` declares ``DEFAULT_MAX_TOKENS ==
  32768`` (lifted from S62-locked 16384 per user authorization 2026-05-14).
  This is the singular exception to the smolagents stack lock that holds
  on model / api_base / temperature / max_steps=40.
* The same module declares ``_FALLBACK_MAX_TOKENS == 16384`` (the
  defensive fallback constant the build_default_model retry path uses).
* ``deepsearch/agent/critique.py`` imports ``DEFAULT_MAX_TOKENS`` from
  ``codeagent_runner`` — so the critique LLM's per-call output cap
  inherits the lifted value uniformly without a duplicate constant.
* ``build_default_model`` catches construction-time exceptions and
  retries once with ``_FALLBACK_MAX_TOKENS`` when ``max_tokens >
  _FALLBACK_MAX_TOKENS``.

These guards fail loudly if a future sprint:

* Reverts ``DEFAULT_MAX_TOKENS`` to 16384 without an explicit
  authorisation event recorded in PD/CMOS.
* Removes ``_FALLBACK_MAX_TOKENS`` (or renames it) so the defensive
  fallback contract is silently disabled.
* Duplicates the cap as a literal in ``critique.py`` instead of importing
  from ``codeagent_runner`` (would create a drift surface where one cap
  changes and the other doesn't).
* Removes the try/except retry path from ``build_default_model``.

Tests do not call any LLM, do not hit the network, and exercise only AST
inspection plus a runtime attribute check on the module-level constants.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CODEAGENT_RUNNER_PATH = REPO_ROOT / "deepsearch" / "agent" / "codeagent_runner.py"
CRITIQUE_PATH = REPO_ROOT / "deepsearch" / "agent" / "critique.py"


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _module_constant_value(path: Path, name: str) -> int | None:
    """Return the int value of a top-level ``name = <int>`` assignment, or None."""
    tree = _module_ast(path)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id == name):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, int):
            return value.value
    return None


# ---------------------------------------------------------------------------
# m03 — DEFAULT_MAX_TOKENS lift to 32768.
# ---------------------------------------------------------------------------


def test_default_max_tokens_lifted_to_32768() -> None:
    """The constant must equal 32768 — user-authorized lift from 16384."""
    value = _module_constant_value(CODEAGENT_RUNNER_PATH, "DEFAULT_MAX_TOKENS")
    assert value == 32768, (
        "S81 m03 (PD124) — DEFAULT_MAX_TOKENS must equal 32768. The S62-"
        "locked 16384 value was lifted on 2026-05-14 per explicit user "
        "authorization. Reverting requires a fresh authorization decision "
        "recorded in CMOS (PD/sprint plan-doc)."
    )


def test_default_max_tokens_runtime_value() -> None:
    """Runtime: the module-imported symbol resolves to 32768."""
    from deepsearch.agent.codeagent_runner import DEFAULT_MAX_TOKENS

    assert DEFAULT_MAX_TOKENS == 32768


def test_fallback_max_tokens_constant_declared() -> None:
    """``_FALLBACK_MAX_TOKENS == 16384`` is the defensive fallback constant."""
    value = _module_constant_value(CODEAGENT_RUNNER_PATH, "_FALLBACK_MAX_TOKENS")
    assert value == 16384, (
        "S81 m03 (PD124) — _FALLBACK_MAX_TOKENS must equal 16384 (the "
        "previous locked cap). The build_default_model defensive retry "
        "path uses this when the model API rejects the new cap."
    )


def test_fallback_max_tokens_runtime_value() -> None:
    """Runtime: the module-imported symbol resolves to 16384."""
    from deepsearch.agent.codeagent_runner import _FALLBACK_MAX_TOKENS

    assert _FALLBACK_MAX_TOKENS == 16384


def test_build_default_model_carries_fallback_retry_path() -> None:
    """``build_default_model`` body references ``_FALLBACK_MAX_TOKENS``.

    The retry path inside the try/except is the load-bearing defensive
    contract — without it, a future model API rejection of the new cap
    crashes the worker instead of degrading gracefully.
    """
    tree = _module_ast(CODEAGENT_RUNNER_PATH)
    for node in tree.body:
        if not (
            isinstance(node, ast.FunctionDef) and node.name == "build_default_model"
        ):
            continue
        body_names = {
            sub.id
            for sub in ast.walk(node)
            if isinstance(sub, ast.Name)
        }
        assert "_FALLBACK_MAX_TOKENS" in body_names, (
            "S81 m03 (PD124, PD129 risk #6) — build_default_model must "
            "reference _FALLBACK_MAX_TOKENS inside its body so the "
            "defensive retry path remains wired in."
        )
        # Also confirm the function carries a try/except block.
        has_try = any(isinstance(sub, ast.Try) for sub in ast.walk(node))
        assert has_try, (
            "S81 m03 (PD124) — build_default_model must wrap its model "
            "construction in a try/except so the defensive retry can fire."
        )
        return
    raise AssertionError(
        "build_default_model function not found in codeagent_runner.py."
    )


# ---------------------------------------------------------------------------
# m03 — critique.py inherits the cap (no duplicate literal).
# ---------------------------------------------------------------------------


def test_critique_inherits_parameters_from_shared_factory() -> None:
    """S93 supersedes the fixed cap: critic inherits the selected model contract."""
    tree = _module_ast(CRITIQUE_PATH)
    imports = [alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.module == "codeagent_runner" for alias in node.names]
    assert "build_default_model" in imports
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "model"):
            assert not {"max_tokens", "temperature"} & {kw.arg for kw in node.keywords}


def test_critique_does_not_pin_literal_16384() -> None:
    """No literal 16384 max_tokens slot remains in critique.py."""
    source = CRITIQUE_PATH.read_text(encoding="utf-8")
    # The literal may legitimately appear in comments or docstrings as
    # historical context — but no ``max_tokens=16384`` call kwarg should
    # remain. The literal expression we forbid is the kwarg pattern.
    assert "max_tokens=16384" not in source, (
        "S81 m03 (PD124) — critique.py must NOT carry a literal "
        "`max_tokens=16384` call argument. The cap value lives in "
        "codeagent_runner.DEFAULT_MAX_TOKENS and the import "
        "above ensures the critique call inherits it."
    )
