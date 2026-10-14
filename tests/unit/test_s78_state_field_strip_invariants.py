"""Static-AST regression guards for the S78 m02 state-field strip.

14th-consecutive-sprint RD2 layered-guard contribution. Pins the two
load-bearing invariants from PD83 (conservative flavor):

* ``AgentState.working_set`` and ``AgentState.sources_found`` are NOT
  declared on the ``AgentState`` TypedDict.
* ``empty_state(...)`` does NOT seed either field via kwargs to
  ``AgentState(...)``.
* No production code under ``deepsearch/`` subscript-accesses these
  fields (``state["working_set"]`` / ``state["sources_found"]``); the
  graceful-handling contract is ``state.get(...)`` everywhere.

Aggressive flavor (strip the dead readers + retire the constant-``fail``
``evidence_links_gate`` / ``traceability_gate`` vestiges) LANDED at S83 m06
(PD83-b): the untested dead ``state.get("working_set")`` /
``state.get("sources_found")`` reads were stripped and the two gates repointed
onto ``citation_telemetry``. Four TESTED source/report utilities still read the
(dead-in-production) field rather than be gutted (decision B): ``runner.py``,
``agent/evidence.py:extract_sources_from_state``, ``report/generator.py``, and
``report/evidence.py:generate_evidence_list``. The reader-pattern guard below is
therefore INVERTED to a never-reappears shape over the OTHER readers, and
``test_s83_m06_invariants.py`` carries the module-wide AST scan (same carve-outs).

These guards fail loudly if a future sprint accidentally:

* re-declares either field on ``AgentState``,
* re-introduces a default for either field in ``empty_state()``,
* introduces a subscript-access pattern (``state["working_set"]``)
  inside ``deepsearch/`` that would crash on the post-S78 default
  state shape.

Tests do not call any LLM, do not hit any network, and do not import
runtime code paths — pure AST inspection over ``deepsearch/`` files.

Test layout follows the S66.1 convention: flat in ``tests/unit/``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPSEARCH_ROOT = REPO_ROOT / "deepsearch"
STATE_PATH = DEEPSEARCH_ROOT / "agent" / "state.py"

STRIPPED_FIELDS = ("working_set", "sources_found")


def _iter_python_sources(root: Path):
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# AgentState class declaration — neither field may appear as an annotation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_name", STRIPPED_FIELDS)
def test_agent_state_class_does_not_declare_field(field_name: str) -> None:
    tree = _module_ast(STATE_PATH)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AgentState":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    assert stmt.target.id != field_name, (
                        f"AgentState re-declared {field_name!r} — S78 m02 "
                        f"stripped this field per PD83 conservative flavor"
                    )
            return
    raise AssertionError(
        "AgentState class definition not found in deepsearch/agent/state.py"
    )


# ---------------------------------------------------------------------------
# empty_state() — neither field may be passed as a kwarg to AgentState(...).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_name", STRIPPED_FIELDS)
def test_empty_state_does_not_seed_field(field_name: str) -> None:
    tree = _module_ast(STATE_PATH)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "empty_state":
            for sub in ast.walk(node):
                if isinstance(sub, ast.keyword):
                    if sub.arg == field_name:
                        raise AssertionError(
                            f"empty_state() re-seeds {field_name!r} — the "
                            f"S78 m02 strip removed this default; readers "
                            f"degrade gracefully via state.get(...)"
                        )
            return
    raise AssertionError(
        "empty_state function not found in deepsearch/agent/state.py"
    )


# ---------------------------------------------------------------------------
# Runtime behavior — empty_state() output does not contain either key.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_name", STRIPPED_FIELDS)
def test_empty_state_runtime_omits_field(field_name: str) -> None:
    """End-to-end guard: actually calling ``empty_state()`` must NOT
    surface the stripped fields. Catches a regression where a future
    edit re-introduces a default via a different code path the AST
    scan above would miss.
    """

    from deepsearch.agent.state import empty_state

    state = empty_state(mission_id="S78.M02.GUARD")
    assert field_name not in state, (
        f"empty_state() runtime output still contains {field_name!r} — "
        f"S78 m02 conservative strip requires the field to be absent"
    )


# ---------------------------------------------------------------------------
# Production-code consumer scan — no subscript access on stripped fields.
# Catches a future writer that adds `state["working_set"] = ...` (which
# would technically work post-strip because TypedDict total=False allows
# arbitrary keys, BUT subscript READS would crash on absence). Subscript
# writes are tolerated; subscript reads are forbidden.
# ---------------------------------------------------------------------------


def _is_subscript_read_of_field(node: ast.AST, field_name: str) -> bool:
    """Return True iff ``node`` is a subscript expression reading ``field_name``.

    Catches the shape ``state["working_set"]`` and ``state.get(...)`` is
    NOT caught because ``state.get(...)`` is a Call, not a Subscript.
    Subscript writes (``state["working_set"] = ...``) are differentiated
    by checking the ast.Store context against ast.Load — only Load reads
    crash on absence.
    """
    if not isinstance(node, ast.Subscript):
        return False
    slice_node = node.slice
    if isinstance(slice_node, ast.Constant) and slice_node.value == field_name:
        ctx = getattr(node, "ctx", None)
        return isinstance(ctx, ast.Load)
    return False


# Test-side fixtures legitimately construct state dicts containing the
# stripped fields (to exercise reader graceful-handling). The guard
# scans deepsearch/ only — production code, not test inputs.
SCAN_ROOT = DEEPSEARCH_ROOT


@pytest.mark.parametrize("field_name", STRIPPED_FIELDS)
def test_no_production_subscript_read_on_stripped_field(field_name: str) -> None:
    offenders: list[str] = []
    for path in _iter_python_sources(SCAN_ROOT):
        try:
            tree = _module_ast(path)
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if _is_subscript_read_of_field(node, field_name):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, (
        f"Production code subscript-reads stripped field {field_name!r}:\n  "
        + "\n  ".join(offenders)
        + f"\n\nUse state.get({field_name!r}) instead — the S78 m02 "
        f"conservative strip means subscript reads crash on the post-S78 "
        f"default state shape."
    )


# ---------------------------------------------------------------------------
# Reader-side never-reappears guard — S83 m06 (PD83-b) INVERSION.
#
# Pre-m06 these files read the stripped fields via state.get(...) and degraded
# gracefully; this guard asserted they STILL did. The aggressive flavor (m06)
# removed all 13 dead reads + repointed the two protocol gates, so the guard is
# inverted: NONE of these files may reference state.get("working_set") /
# state.get("sources_found") any longer. A re-appearance would resurrect a
# dead-field dependency. (Module-wide AST scan: test_s83_m06_invariants.py.)
# ---------------------------------------------------------------------------


# NB report/evidence.py + agent/evidence.py are intentionally ABSENT — S83 m06
# decision B kept their generic source-extraction utilities
# (generate_evidence_list / extract_sources_from_state) as the two retained
# readers of the dead field rather than gut them + their test suites. Documented
# carve-outs; see test_s83_m06_invariants.py::_CARVE_OUT_RELPATHS.
FORMER_READER_FILES = (
    "agent/finding.py",
    "agent/coverage.py",
    "report/protocol.py",
    "validation/hybrid_evaluator.py",
)


def test_known_reader_files_no_longer_use_state_get_pattern() -> None:
    """S83 m06 inversion: the PD83 reader files must NO LONGER call
    ``state.get(...)`` (or ``final_state.get(...)``) on a stripped field — the
    aggressive strip removed every dead read. Docstring examples use the
    dict-literal ``"sources_found":`` form and never match this ``.get`` phrase.
    """

    forbidden_phrases = tuple(f'state.get("{f}")' for f in STRIPPED_FIELDS) + tuple(
        f'final_state.get("{f}")' for f in STRIPPED_FIELDS
    )
    offenders: list[str] = []
    for relative in FORMER_READER_FILES:
        path = DEEPSEARCH_ROOT / relative
        if not path.exists():
            continue  # a deleted reader file is fine — the read is gone with it
        source = path.read_text(encoding="utf-8")
        for phrase in forbidden_phrases:
            if phrase in source:
                offenders.append(f"{relative}: {phrase}")
    assert not offenders, (
        "S83 m06 stripped these reads; a re-appearance resurrects a dead-field "
        "dependency:\n  " + "\n  ".join(offenders)
    )
