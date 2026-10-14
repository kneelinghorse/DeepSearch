"""Static-AST + import-surface regression guards for the S77 m02 working_set surgery.

13th-consecutive-sprint RD2 layered-guard contribution (m02 half — m01 half lives
at ``tests/unit/test_s77_dead_code_invariants.py``).

S77 m02 deletes:

* ``deepsearch.agent.working_set`` — the bounded scored working-set + admission
  pipeline (pre-rebuild source-management substrate; replaced by the CodeAgent's
  live retrieval tools at S64).
* ``deepsearch.validation.acceptance_smoke`` — local diagnostic harness (no
  deployed runtime path; already broken since S75 m01's removal of
  ``agent/finalize_node`` per the file's own internal note at the deletion site).

Plus:

* ``slot_coverage.py`` simplified to ``extract_failing_slot_details`` only
  (``compute_slot_coverage`` / ``unfilled_slot_ids`` /
  ``format_slot_coverage_summary`` removed).
* ``codeagent_finalize_node.py:51,102,271`` working_set threading removed.
* ``checkpointer.py:11`` working_set import + the
  ``normalize_checkpoint_state`` legacy migration branch removed.

These guards fail loudly if a future sprint accidentally:

* re-introduces ``deepsearch/agent/working_set.py`` or
  ``deepsearch/validation/acceptance_smoke.py``,
* re-imports any working_set symbol from elsewhere
  (``WorkingSet`` / ``ensure_working_set`` / ``sync_state_sources`` /
  ``admit_sources`` / ``ScoredSource`` / ``DropRecord`` / ``AdmissionSummary``
  / ``academic_authority_bonus`` / ``lane_weight_for`` /
  ``resolve_working_set_capacity``),
* re-introduces any of the dropped slot_coverage helpers
  (``compute_slot_coverage`` / ``unfilled_slot_ids`` /
  ``format_slot_coverage_summary``),
* re-introduces the working_set migration branch in ``normalize_checkpoint_state``.

Tests do not call any LLM, do not hit any network, do not require pytest
fixtures, and do not import the deleted modules (verifying they are truly
gone via ``importlib`` ImportError).

State-field strip on ``AgentState.working_set`` / ``AgentState.sources_found``
was DEFERRED at m02 Phase A per PD71 (live readers exist; plan §5.4 condition
resolves negative). This file does NOT enforce field absence — that becomes
a future-sprint concern when the readers themselves are stripped.

Test layout follows the S66.1 convention: flat in ``tests/unit/``.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPSEARCH_ROOT = REPO_ROOT / "deepsearch"
TESTS_ROOT = REPO_ROOT / "tests"
AGENT_ROOT = DEEPSEARCH_ROOT / "agent"

DELETED_MODULES = (
    "deepsearch.agent.working_set",
    "deepsearch.validation.acceptance_smoke",
)

DELETED_RELATIVE_NAMES = ("working_set",)

DELETED_WORKING_SET_SYMBOLS = (
    "AdmissionSummary",
    "DropRecord",
    "ScoredSource",
    "WorkingSet",
    "academic_authority_bonus",
    "admit_sources",
    "ensure_working_set",
    "lane_weight_for",
    "resolve_working_set_capacity",
    "sync_state_sources",
)

DELETED_SLOT_COVERAGE_SYMBOLS = (
    "compute_slot_coverage",
    "format_slot_coverage_summary",
    "unfilled_slot_ids",
)

SLOT_COVERAGE_PATH = AGENT_ROOT / "slot_coverage.py"
CHECKPOINTER_PATH = AGENT_ROOT / "checkpointer.py"


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _iter_python_sources(root: Path):
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


# ---------------------------------------------------------------------------
# Whole-file deletions: the 2 modules must be unimportable.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", DELETED_MODULES)
def test_deleted_module_cannot_be_imported(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", DELETED_MODULES)
def test_deleted_module_path_does_not_exist(module_name: str) -> None:
    relative = module_name.removeprefix("deepsearch.").replace(".", "/")
    path = DEEPSEARCH_ROOT / f"{relative}.py"
    assert not path.exists(), f"deleted module re-introduced at {path}"


# ---------------------------------------------------------------------------
# No remaining absolute-form imports of the deleted modules anywhere in the
# tree.
# ---------------------------------------------------------------------------


def _imports_absolute_module(node: ast.AST, target: str) -> bool:
    parent, _, leaf = target.rpartition(".")
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module == target:
            return True
        if module == parent:
            for alias in node.names:
                if alias.name == leaf:
                    return True
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == target:
                return True
    return False


@pytest.mark.parametrize("target_module", DELETED_MODULES)
def test_no_consumer_imports_deleted_module(target_module: str) -> None:
    """No file under deepsearch/ or tests/ imports the deleted modules."""
    offenders: list[str] = []
    for path in (*_iter_python_sources(DEEPSEARCH_ROOT), *_iter_python_sources(TESTS_ROOT)):
        if path.name == "test_s77_working_set_invariants.py":
            continue
        try:
            tree = _module_ast(path)
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if _imports_absolute_module(node, target_module):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, (
        f"Re-imports of deleted module {target_module!r} detected:\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# No relative imports of working_set from inside ``deepsearch/agent``.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_relative", DELETED_RELATIVE_NAMES)
def test_no_relative_imports_of_deleted_modules(target_relative: str) -> None:
    offenders: list[str] = []
    for path in _iter_python_sources(AGENT_ROOT):
        try:
            tree = _module_ast(path)
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level >= 1 and node.module == target_relative:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, (
        f"Relative imports of {target_relative!r} detected in "
        f"deepsearch/agent:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# No imported-name reference to deleted working_set symbols anywhere
# (catches ``from deepsearch.agent.working_set import X`` re-introductions
# even when ``X`` would be a valid symbol on a sibling module).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", DELETED_WORKING_SET_SYMBOLS)
def test_no_consumer_imports_deleted_working_set_symbol(symbol: str) -> None:
    offenders: list[str] = []
    for path in (*_iter_python_sources(DEEPSEARCH_ROOT), *_iter_python_sources(TESTS_ROOT)):
        if path.name == "test_s77_working_set_invariants.py":
            continue
        try:
            tree = _module_ast(path)
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                # Match any from-import that lands inside the agent package
                # asking for the symbol — both absolute (deepsearch.agent.X)
                # and relative (.X). The check is intentionally loose: we
                # don't want a future re-export to silently re-introduce the
                # symbol via a sibling module.
                if module == "deepsearch.agent.working_set" or (
                    node.level >= 1 and module == "working_set"
                ):
                    for alias in node.names:
                        if alias.name == symbol:
                            offenders.append(
                                f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
                            )
    assert not offenders, (
        f"Imports of deleted working_set symbol {symbol!r} detected:\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# slot_coverage.py: __all__ shape + dropped function symbols not redefined.
# ---------------------------------------------------------------------------


def test_slot_coverage_all_exports_only_extract_failing_slot_details() -> None:
    tree = _module_ast(SLOT_COVERAGE_PATH)
    exported_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                exported_names.append(elt.value)
    assert exported_names == ["extract_failing_slot_details"], (
        f"slot_coverage.__all__ drifted: {exported_names!r}"
    )


@pytest.mark.parametrize("symbol", DELETED_SLOT_COVERAGE_SYMBOLS)
def test_slot_coverage_dropped_symbol_not_redefined(symbol: str) -> None:
    tree = _module_ast(SLOT_COVERAGE_PATH)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assert node.name != symbol, (
                f"slot_coverage.py re-introduced {symbol!r} (was dropped at S77 m02 "
                f"per PD62)"
            )


# ---------------------------------------------------------------------------
# checkpointer.normalize_checkpoint_state must NOT contain the legacy
# working_set / sources_found migration branch.
# ---------------------------------------------------------------------------


def test_checkpointer_normalize_state_drops_working_set_branch() -> None:
    source = CHECKPOINTER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "normalize_checkpoint_state":
            for sub in ast.walk(node):
                # Catch any reference to ``ensure_working_set`` /
                # ``sync_state_sources`` inside the function body.
                if isinstance(sub, ast.Name) and sub.id in {
                    "ensure_working_set",
                    "sync_state_sources",
                }:
                    raise AssertionError(
                        "checkpointer.normalize_checkpoint_state still references "
                        f"{sub.id!r} — the working_set migration branch was removed "
                        f"at S77 m02 (PD61.2)"
                    )
            return
    raise AssertionError(
        "checkpointer.normalize_checkpoint_state was not found — surgery may "
        "have removed the function entirely (unintended)"
    )
