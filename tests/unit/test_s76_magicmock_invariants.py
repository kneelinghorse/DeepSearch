"""Sprint 76 PD47 — bare-MagicMock-of-dataclass guard.

Codifies the S75 hot-fix `b9ff080` lesson: tests that fixture project dataclass
types with bare ``MagicMock(...)`` mask real-dataclass divergence, because
MagicMock auto-creates any attribute access. After ExecutionMetadata had
``loops_executed`` / ``sources_count`` stripped at S75, the test fixture
``MagicMock(loops_executed=N, sources_count=N)`` still passed — production code
accessed ``metadata.loops_executed``, MagicMock invented the attribute, and the
divergence only surfaced at deploy.

The fix landed in S75 hot-fix at three patch sites: real ``ExecutionMetadata(...)``
instances replace the bare MagicMock fixtures.

This guard enforces the rule project-wide so the same anti-pattern can't recur:

  Forbidden: ``MagicMock(loops_executed=N, sources_count=N)``
             (kwarg names match a whitelisted dataclass field; no ``spec=``)

  Allowed:   ``MagicMock(spec=ExecutionMetadata, loops_executed=N)``
             (``spec=`` constrains attribute access to the real type)

  Allowed:   ``MagicMock()``  (no kwargs — likely a non-dataclass mock)

  Allowed:   ``MagicMock(return_value="x")``  (mock-internal kwargs only)

  Allowed:   real ``ExecutionMetadata(total_tokens=0, duration_seconds=0.0)``

Whitelist scope is narrow by design (PD47): only the 8 dataclass types whose
field shape is load-bearing across the worker / agent / report boundary.
Non-dataclass mocks (HTTP clients, file handles, settings stubs) are
unaffected.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
from pathlib import Path
from typing import Iterable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"


# Whitelist: dataclass / pydantic / TypedDict types whose field-shape is
# load-bearing. Maps display name → import path. Field names are introspected
# at test collection time so the rule always tracks the real source.
WHITELISTED_TYPES: dict[str, str] = {
    "ExecutionMetadata": "deepsearch.worker.writer",
    "MissionRow": "deepsearch.worker.poller",
    "MissionProtocolComplete": "deepsearch.report.protocol",
    "AgentState": "deepsearch.agent.state",
    "Synthesis": "deepsearch.report.synthesis",
    "CitationAnnotation": "deepsearch.report.citation",
    "Evidence": "deepsearch.report.evidence",
    "ReportMetadata": "deepsearch.report.protocol",
}


# Mock-internal kwargs are not dataclass fields; ignore them when computing
# the suspicious-kwarg overlap.
MOCK_INTERNAL_KWARGS = frozenset(
    {
        "spec",
        "spec_set",
        "return_value",
        "side_effect",
        "wraps",
        "name",
        "_spec_class",
        "_spec_set",
        "unsafe",
    }
)


def _resolve_field_names() -> dict[str, set[str]]:
    """Resolve current field-name set for each whitelisted type.

    Handles three flavors: ``@dataclass`` (regular), ``BaseModel`` (pydantic),
    and ``TypedDict``. Skips types that fail to import in this environment.
    """

    out: dict[str, set[str]] = {}
    for type_name, module_path in WHITELISTED_TYPES.items():
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        type_obj = getattr(module, type_name, None)
        if type_obj is None:
            continue
        if dataclasses.is_dataclass(type_obj):
            out[type_name] = {f.name for f in dataclasses.fields(type_obj)}
            continue
        model_fields = getattr(type_obj, "model_fields", None)
        if isinstance(model_fields, dict):
            out[type_name] = set(model_fields.keys())
            continue
        annotations = getattr(type_obj, "__annotations__", None)
        if isinstance(annotations, dict):
            out[type_name] = set(annotations.keys())
    return out


def _imported_whitelist_names(tree: ast.AST) -> set[str]:
    """Return the subset of whitelisted type names imported by this AST."""

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in WHITELISTED_TYPES:
                    imported.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                # ``import deepsearch.worker.writer`` → no name match here;
                # the dataclass type itself is what counts.
                pass
    return imported


def _is_mock_call(node: ast.Call) -> bool:
    """True if the Call invokes ``MagicMock`` / ``Mock`` (any namespace)."""

    func = node.func
    if isinstance(func, ast.Name) and func.id in {"MagicMock", "Mock"}:
        return True
    if isinstance(func, ast.Attribute) and func.attr in {"MagicMock", "Mock"}:
        return True
    return False


def _has_spec_kwarg(node: ast.Call) -> bool:
    return any(kw.arg in {"spec", "spec_set"} for kw in node.keywords)


def _suspicious_kwargs(node: ast.Call) -> list[str]:
    """Return non-mock-internal kwarg names from this Call."""

    return [
        kw.arg
        for kw in node.keywords
        if kw.arg is not None and kw.arg not in MOCK_INTERNAL_KWARGS
    ]


def _iter_test_files() -> Iterable[Path]:
    for path in sorted(TESTS_DIR.rglob("*.py")):
        # Skip the guard module itself + __pycache__ etc.
        if path.name == Path(__file__).name:
            continue
        if "__pycache__" in path.parts:
            continue
        yield path


@pytest.fixture(scope="module")
def whitelist_fields_by_type() -> dict[str, set[str]]:
    """Per-type field-name sets for each whitelisted dataclass."""

    by_type = _resolve_field_names()
    assert by_type, "whitelisted type field sets must resolve at test time"
    return by_type


def _file_imported_whitelist_types(tree: ast.AST) -> set[str]:
    """Set of whitelisted type names this file imports by name."""

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in WHITELISTED_TYPES:
                    imported.add(alias.name)
    return imported


def _file_relevant_fields(
    tree: ast.AST, by_type: dict[str, set[str]]
) -> set[str]:
    """Field-name union over the whitelisted types this file imports.

    Narrows the rule per PD47: if the file doesn't import (e.g.,)
    ``CitationAnnotation``, kwargs / attribute names that happen to
    overlap with its fields are not flagged.
    """

    imported = _file_imported_whitelist_types(tree)
    union: set[str] = set()
    for type_name in imported:
        union |= by_type.get(type_name, set())
    return union


def test_whitelisted_types_resolve_field_sets() -> None:
    """Every whitelisted type must resolve to a non-empty field set.

    If a future rebuild renames or relocates one of these types, this test
    fails loud rather than letting the broader guard silently weaken.
    """

    by_type = _resolve_field_names()
    missing = sorted(set(WHITELISTED_TYPES) - set(by_type))
    assert not missing, (
        f"could not resolve field set for whitelisted types: {missing}. "
        "Update WHITELISTED_TYPES paths in this guard."
    )
    empty = [name for name, fields in by_type.items() if not fields]
    assert not empty, f"whitelisted types resolved to empty field sets: {empty}"


def test_no_bare_magicmock_with_dataclass_field_kwargs(
    whitelist_fields_by_type: dict[str, set[str]],
) -> None:
    """No bare ``MagicMock(...)``/``Mock(...)`` may carry dataclass-field kwargs.

    Catches the S75 hot-fix kwarg-shape — ``MagicMock(loops_executed=N)`` —
    where a kwarg name matches a real field on a whitelisted dataclass type
    that the test file imports, but the call lacks ``spec=`` to constrain
    attribute access. Such fixtures auto-create stripped attributes and
    mask production-side divergence.

    Fix: either pass ``spec=ExecutionMetadata`` so attribute access is
    constrained to the real shape, or construct a real instance.

    Narrowing (PD47): only fires when the test file imports the
    whitelisted type whose field is involved — kwarg overlap with an
    unimported type's field set (e.g., ``mock_segment.start_index`` in
    Google Search response tests) is not a violation.
    """

    violations: list[str] = []
    for path in _iter_test_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        relevant_fields = _file_relevant_fields(tree, whitelist_fields_by_type)
        if not relevant_fields:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not _is_mock_call(node):
                continue
            if _has_spec_kwarg(node):
                continue
            kwargs = _suspicious_kwargs(node)
            offending = sorted(set(kwargs) & relevant_fields)
            if not offending:
                continue
            rel = path.relative_to(REPO_ROOT)
            violations.append(
                f"{rel}:{node.lineno}: bare MagicMock with dataclass-field kwargs "
                f"{offending} (matches a whitelisted type imported by this file). "
                "Fix: use MagicMock(spec=DataclassType, ...) "
                "or construct a real instance."
            )

    assert not violations, (
        "PD47 violation — bare MagicMock cannot stand in for a whitelisted "
        "project dataclass.\n\n"
        + "\n".join(violations)
        + "\n\nWhy this rule exists: see S75 hot-fix b9ff080. Bare MagicMock "
        "auto-creates any attribute access, so when a dataclass field is "
        "stripped, fixtures keep passing while production raises AttributeError. "
        "spec= constrains attribute access to the real dataclass shape."
    )


def _bare_mock_bindings(func: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, int]:
    """Map variable name → first lineno of a bare ``MagicMock()`` bind.

    Catches both ``x = MagicMock()`` and ``x = Mock()``, ignoring spec'd
    calls. Used to track which locals carry a bare-mock value for the
    attribute-assignment pattern check.
    """

    bindings: dict[str, int] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        if not _is_mock_call(node.value):
            continue
        if _has_spec_kwarg(node.value):
            continue
        bindings.setdefault(target.id, node.lineno)
    return bindings


def test_no_bare_magicmock_followed_by_dataclass_field_assignment(
    whitelist_fields_by_type: dict[str, set[str]],
) -> None:
    """No bare ``MagicMock()`` may be followed by ``.<dataclass-field> = ...``.

    Catches the S75 hot-fix assignment-shape:

      mock_meta_instance = MagicMock()
      mock_meta_instance.loops_executed = 2  # auto-creates stripped attribute

    Fix: pass ``spec=ExecutionMetadata`` to the constructor or use a real
    instance.

    Detection is per-function with per-file narrowing: a bare-mock bind
    plus a same-function attribute assignment to a field name in the
    whitelist of types this file imports.
    """

    violations: list[str] = []
    for path in _iter_test_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        relevant_fields = _file_relevant_fields(tree, whitelist_fields_by_type)
        if not relevant_fields:
            continue
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            bare_mock_vars = _bare_mock_bindings(func)
            if not bare_mock_vars:
                continue
            for node in ast.walk(func):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if not isinstance(target, ast.Attribute):
                        continue
                    if not isinstance(target.value, ast.Name):
                        continue
                    var_name = target.value.id
                    if var_name not in bare_mock_vars:
                        continue
                    if target.attr not in relevant_fields:
                        continue
                    rel = path.relative_to(REPO_ROOT)
                    bind_line = bare_mock_vars[var_name]
                    violations.append(
                        f"{rel}:{node.lineno}: assignment to '{var_name}.{target.attr}' "
                        f"where '{var_name}' was bound to bare MagicMock at line "
                        f"{bind_line} ('{target.attr}' is a field on a whitelisted "
                        "type imported by this file). Fix: pass spec=DataclassType "
                        "to the MagicMock constructor or construct a real instance."
                    )

    assert not violations, (
        "PD47 violation — bare MagicMock followed by dataclass-field assignment.\n\n"
        + "\n".join(violations)
        + "\n\nWhy this rule exists: see S75 hot-fix b9ff080 (the second "
        "anti-pattern). MagicMock without spec= auto-creates any attribute, "
        "so stripped fields stay invisible to the test. spec=DataclassType "
        "constrains attribute access to the real shape."
    )
