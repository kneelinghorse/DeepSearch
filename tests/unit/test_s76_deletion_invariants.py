"""Static-AST + import-surface regression guards for the S76 source-level deletion.

12th-consecutive-sprint RD2 layered-guard contribution. Mirrors the layered-guard
pattern from S65 (synthesis-path bypass), S66 (critique single-call),
S67 (citation extractor), S68 (citation liveness), S68.5 (writer
persistence), S69 (factscore invariants), S71 (sources_collected
persistence), S74 (metadata truthfulness), and S75 (deletion trio).

S76 = Option 3 hybrid slice of the Sprint 75 PD44(a) carry list. Two
modules deleted in this sprint:

* ``deepsearch.agent.depth`` (256 LOC) — the ResearchDepth enum + DepthConfig
  dataclass + parse_depth/get_depth_config helpers + DEPTH_CONFIGS registry.
  Replaced by plain string research_depth labels at the boundary.
* ``deepsearch.validation.structural_validators`` (1,282 LOC) — the
  pre-rebuild deliverable-schema validators. Replaced by an empty
  pass-through StructuralValidationSummary in validation/harness.py.

The remaining two PD49 entries (``agent.working_set`` + ``agent.source_quality``)
were deferred to S77 because the post-S64 live path
(``codeagent_finalize_node`` + ``checkpointer`` + ``slot_coverage``) still
calls ``ensure_working_set`` + ``sync_state_sources`` + ``WorkingSet``.
See ``cmos/reports/sprint_76_keep_side_audit.md`` for the audit + rationale.

If a future sprint accidentally:

* re-introduces ``deepsearch/agent/depth.py`` or
  ``deepsearch/validation/structural_validators.py``,
* re-imports any deleted symbol (``ResearchDepth`` / ``DepthConfig`` /
  ``parse_depth`` / ``get_depth_config`` / ``DEPTH_CONFIGS`` /
  ``run_validators`` / ``ValidationReport`` / ``ValidatorResult``),
* re-adds depth-tier dispatch (``get_depth_config(...)`` /
  ``parse_depth(...)``),
* re-adds the legacy ``_apply_depth_override`` business logic that
  promoted max_loops/min_loops from depth tier defaults,
* breaks module-level import sanity for any of the modules that survived
  the deletion event,

these guards fail loudly. Tests do not call any LLM, do not hit any
network, do not require pytest fixtures, and do not import the deleted
modules (verifying they are truly gone via ``importlib`` ImportError).

Test layout follows the S66.1 convention (decision #501): flat in
``tests/unit/``, not ``tests/unit/agent/``.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPSEARCH_ROOT = REPO_ROOT / "deepsearch"
TESTS_ROOT = REPO_ROOT / "tests"

DELETED_MODULES = (
    "deepsearch.agent.depth",
    "deepsearch.validation.structural_validators",
)

DELETED_DEPTH_SYMBOLS = (
    "DEPTH_CONFIGS",
    "DepthConfig",
    "ResearchDepth",
    "get_depth_config",
    "parse_depth",
)

DELETED_VALIDATOR_SYMBOLS = (
    "run_validators",
    "ValidationReport",
    "ValidatorResult",
)

# Boundary types where ``research_depth`` survives as a plain ``str`` field
# (TraceLab MCP / API request / YAML / contract / telemetry boundaries).
# Hard Rule 10 attribute-access scan tolerates these contexts.
BOUNDARY_RESEARCH_DEPTH_OWNERS = {
    "deepsearch/agent/evidence.py",
    "deepsearch/api/queue.py",
    "deepsearch/api/models/requests.py",
    "deepsearch/mission/contract.py",
    "deepsearch/mission/loader.py",
    "deepsearch/mission/schema.py",
    "deepsearch/runner.py",
    "deepsearch/worker/converter.py",
    "deepsearch/worker/poller.py",
}


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
# No remaining imports of the deleted modules anywhere in the tree.
# ---------------------------------------------------------------------------


def _imports_module(node: ast.AST, target: str) -> bool:
    """Return True when an AST node imports `target` (relative or absolute)."""
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module == target:
            return True
        # Catch absolute-form imports like `import deepsearch.agent.depth`.
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
        # Skip the guard file itself + the audit doc references.
        if path.name == "test_s76_deletion_invariants.py":
            continue
        try:
            tree = _module_ast(path)
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if _imports_module(node, target_module):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, (
        f"Re-imports of deleted module {target_module!r} detected:\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Relative imports of the deleted modules from inside their original packages
# (``from .depth import ...`` / ``from .structural_validators import ...``).
# ---------------------------------------------------------------------------


_RELATIVE_IMPORT_TARGETS = (
    ("depth", "deepsearch/agent"),
    ("structural_validators", "deepsearch/validation"),
)


@pytest.mark.parametrize("target_relative,package_dir", _RELATIVE_IMPORT_TARGETS)
def test_no_relative_imports_of_deleted_modules(
    target_relative: str, package_dir: str
) -> None:
    package_path = REPO_ROOT / package_dir
    offenders: list[str] = []
    for path in _iter_python_sources(package_path):
        if path.name == "test_s76_deletion_invariants.py":
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
        f"{package_dir}:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Symbols from the deleted modules must not be referenced as imported names
# in deepsearch/ source. Comment/docstring text mentions are tolerated.
# ---------------------------------------------------------------------------


def _aliased_imports(tree: ast.Module) -> set[str]:
    """Return the set of names introduced into a module by import statements."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
    return names


@pytest.mark.parametrize(
    "symbol", DELETED_DEPTH_SYMBOLS + DELETED_VALIDATOR_SYMBOLS
)
def test_no_consumer_imports_deleted_symbols(symbol: str) -> None:
    """No deepsearch/ module imports any deleted symbol."""
    offenders: list[str] = []
    for path in _iter_python_sources(DEEPSEARCH_ROOT):
        if path.name == "test_s76_deletion_invariants.py":
            continue
        try:
            tree = _module_ast(path)
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == symbol:
                        offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, (
        f"Re-imports of deleted symbol {symbol!r} detected:\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Legacy depth-driven business logic must not return.
# ---------------------------------------------------------------------------


RUNNER_PATH = DEEPSEARCH_ROOT / "runner.py"


def test_runner_apply_depth_override_has_no_get_depth_config_call() -> None:
    """`_apply_depth_override` must not re-introduce depth-config dispatch."""
    tree = _module_ast(RUNNER_PATH)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_apply_depth_override":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    fn = inner.func
                    if isinstance(fn, ast.Name) and fn.id in {"parse_depth", "get_depth_config"}:
                        raise AssertionError(
                            "_apply_depth_override re-introduced depth-config dispatch; "
                            "post-S76 the function must only normalize the boundary string"
                        )
            return  # function found and clean
    # If the function was deleted in a future sprint, that's also acceptable.


# ---------------------------------------------------------------------------
# Hard Rule 10 — `obj.research_depth` attribute access scan.
#
# `research_depth` survives as a plain ``str`` boundary field on MissionRow,
# MissionYAML, ExecutionBudget, EvidenceTelemetry, and ExecuteMissionRequest.
# Reading those fields is allowed, but only inside the small set of boundary
# files that own the data flow. Any new file under deepsearch/ that introduces
# `obj.research_depth` access without being whitelisted should fail this guard
# so future drift is visible.
# ---------------------------------------------------------------------------


def test_research_depth_attribute_access_remains_bounded() -> None:
    """Hard Rule 10: `obj.research_depth` access only at known boundary files."""
    offenders: list[str] = []
    for path in _iter_python_sources(DEEPSEARCH_ROOT):
        relative = str(path.relative_to(REPO_ROOT))
        if relative in BOUNDARY_RESEARCH_DEPTH_OWNERS:
            continue
        try:
            tree = _module_ast(path)
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "research_depth":
                offenders.append(f"{relative}:{node.lineno}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "research_depth"
            ):
                offenders.append(f"{relative}:{node.lineno}")
    assert not offenders, (
        "obj.research_depth / getattr(obj, 'research_depth') access detected "
        "outside the boundary whitelist. Either route the read through the "
        "owner module or extend BOUNDARY_RESEARCH_DEPTH_OWNERS with explicit "
        "rationale.\nOffenders:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Module-level import sanity: every retained agent/* + validation/* module
# imports cleanly post-deletion. Catches transitive-import regressions early.
# ---------------------------------------------------------------------------


def _retained_modules(package_relative: str) -> list[str]:
    package_path = DEEPSEARCH_ROOT / package_relative
    modules: list[str] = []
    for path in package_path.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if path.name == "__init__.py":
            relative = path.parent.relative_to(DEEPSEARCH_ROOT)
            module = "deepsearch." + ".".join(relative.parts)
        else:
            relative = path.relative_to(DEEPSEARCH_ROOT).with_suffix("")
            module = "deepsearch." + ".".join(relative.parts)
        modules.append(module)
    return sorted(set(modules))


@pytest.mark.parametrize("module_name", _retained_modules("agent"))
def test_retained_agent_modules_import_cleanly(module_name: str) -> None:
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", _retained_modules("validation"))
def test_retained_validation_modules_import_cleanly(module_name: str) -> None:
    importlib.import_module(module_name)
