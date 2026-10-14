"""Static-AST + import-surface regression guards for the S77 m01 dead-code trio.

13th-consecutive-sprint RD2 layered-guard contribution. Mirrors the layered-guard
pattern from S65 (synthesis-path bypass), S66 (critique single-call),
S67 (citation extractor), S68 (citation liveness), S68.5 (writer
persistence), S69 (factscore invariants), S71 (sources_collected
persistence), S74 (metadata truthfulness), S75 (deletion trio), and
S76 (depth + structural_validators).

S77 m01 deletes three modules that became true dead-code after the S76 m02
test-file deletions stripped their last consumers:

* ``deepsearch.agent.prompts_v2`` — pre-rebuild research-prompt builder.
* ``deepsearch.agent.reflection_node`` — pre-rebuild quantitative-reflection
  LangGraph node. Was registered but unreachable post-S64.
* ``deepsearch.agent.search_orchestrator`` — pre-rebuild research-loop
  orchestrator. Zero consumers anywhere at audit re-validation.

These guards fail loudly if a future sprint accidentally:

* re-introduces ``deepsearch/agent/prompts_v2.py`` /
  ``deepsearch/agent/reflection_node.py`` /
  ``deepsearch/agent/search_orchestrator.py``,
* re-imports any of the modules from elsewhere (absolute or relative form),
* re-exports a deleted-module submodule from ``agent/__init__.py``.

Tests do not call any LLM, do not hit any network, do not require pytest
fixtures, and do not import the deleted modules (verifying they are truly
gone via ``importlib`` ImportError).

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
    "deepsearch.agent.prompts_v2",
    "deepsearch.agent.reflection_node",
    "deepsearch.agent.search_orchestrator",
)

DELETED_RELATIVE_NAMES = (
    "prompts_v2",
    "reflection_node",
    "search_orchestrator",
)


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _iter_python_sources(root: Path):
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


# ---------------------------------------------------------------------------
# Whole-file deletions: the 3 modules must be unimportable.
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
# tree (``from deepsearch.agent.X import ...`` / ``import deepsearch.agent.X``
# / ``from deepsearch.agent import X``).
# ---------------------------------------------------------------------------


def _imports_absolute_module(node: ast.AST, target: str) -> bool:
    """Return True when an AST node imports `target` in absolute form.

    Catches three import shapes:
      * ``from deepsearch.agent.prompts_v2 import X``    (ImportFrom.module == target)
      * ``import deepsearch.agent.prompts_v2``           (Import.alias == target)
      * ``from deepsearch.agent import prompts_v2``      (ImportFrom.module == parent,
                                                         alias.name == leaf)
    """
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
        # Skip the guard file itself.
        if path.name == "test_s77_dead_code_invariants.py":
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
# No relative imports of the deleted modules from inside ``deepsearch/agent``
# (``from .prompts_v2 import ...`` etc.).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_relative", DELETED_RELATIVE_NAMES)
def test_no_relative_imports_of_deleted_modules(target_relative: str) -> None:
    offenders: list[str] = []
    for path in _iter_python_sources(AGENT_ROOT):
        if path.name == "test_s77_dead_code_invariants.py":
            continue
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
# agent/__init__.py must not re-export any of the deleted submodules.
# ---------------------------------------------------------------------------


def test_agent_init_does_not_reexport_deleted_modules() -> None:
    init_path = AGENT_ROOT / "__init__.py"
    source = init_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Pull __all__ values + _LAZY_EXPORTS keys statically.
    exported_names: set[str] = set()
    lazy_export_modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                exported_names.add(elt.value)
                if isinstance(target, ast.Name) and target.id == "_LAZY_EXPORTS":
                    if isinstance(node.value, ast.Dict):
                        for value in node.value.values:
                            if (
                                isinstance(value, ast.Tuple)
                                and value.elts
                                and isinstance(value.elts[0], ast.Constant)
                                and isinstance(value.elts[0].value, str)
                            ):
                                lazy_export_modules.add(value.elts[0].value)

    # __all__ must not name any deleted submodule.
    for relative in DELETED_RELATIVE_NAMES:
        assert relative not in exported_names, (
            f"agent/__init__.py __all__ re-exports deleted submodule {relative!r}"
        )

    # _LAZY_EXPORTS must not point at any deleted module path.
    for absolute in DELETED_MODULES:
        assert absolute not in lazy_export_modules, (
            f"agent/__init__.py _LAZY_EXPORTS resolves into deleted module {absolute!r}"
        )
