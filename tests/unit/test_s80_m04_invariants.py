"""Static-AST regression guards for the S80 m04 filename uniqueness fix.

16th-consecutive-sprint RD2 layered-guard contribution. Pins the load-bearing
invariants from PD109:

* ``deepsearch/agent/state.py`` declares the ``mission_uuid: Optional[str]``
  field on ``AgentState`` and ``empty_state()`` accepts a ``mission_uuid``
  keyword argument.
* ``deepsearch/worker/converter.py`` propagates the TraceLab
  ``MissionRow.id`` UUID into ``empty_state`` via ``mission_uuid=row.id``.
  The boundary contract names ``MissionRow.id`` as the system-assigned
  per-row UUID; without this propagation, re-runs of the same mission_id
  slug silently overwrite each other on disk.
* ``deepsearch/report/generator.py`` exposes ``_filename_unique_suffix``
  AND both ``save_report`` and ``save_protocol`` accept a ``mission_uuid``
  keyword-only argument.
* ``deepsearch/agent/codeagent_finalize_node.py`` passes
  ``mission_uuid=mission_uuid`` to BOTH ``save_report`` and
  ``save_protocol``.

These guards fail loudly if a future sprint:

* Renames or removes ``mission_uuid`` from ``AgentState`` /
  ``empty_state``.
* Drops the worker propagation from ``MissionRow.id`` to
  ``mission_uuid``.
* Removes the ``mission_uuid`` parameter from the persistence layer or
  forgets to thread it from the finalize node call sites.

Tests do not call any LLM, do not hit the network, and exercise only AST
inspection plus runtime calls on the helper.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPSEARCH_ROOT = REPO_ROOT / "deepsearch"
STATE_PATH = DEEPSEARCH_ROOT / "agent" / "state.py"
WORKER_CONVERTER_PATH = DEEPSEARCH_ROOT / "worker" / "converter.py"
GENERATOR_PATH = DEEPSEARCH_ROOT / "report" / "generator.py"
FINALIZE_NODE_PATH = DEEPSEARCH_ROOT / "agent" / "codeagent_finalize_node.py"


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# AgentState — `mission_uuid` field.
# ---------------------------------------------------------------------------


def test_agent_state_typeddict_carries_mission_uuid_field() -> None:
    tree = _module_ast(STATE_PATH)
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == "AgentState"):
            continue
        annotated_fields = {
            stmt.target.id
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        }
        assert "mission_uuid" in annotated_fields, (
            "AgentState must declare a `mission_uuid: Optional[str]` field "
            "(S80 m04 PD109). Removing this field forces re-runs of the same "
            "mission_id slug back to the silent-overwrite behavior closed at "
            "S80 m04."
        )
        return
    raise AssertionError("AgentState class declaration not found.")


def test_empty_state_accepts_mission_uuid_kwarg() -> None:
    """The empty_state factory must accept and pin mission_uuid."""
    tree = _module_ast(STATE_PATH)
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name == "empty_state"):
            continue
        kwonly_names = {arg.arg for arg in node.args.kwonlyargs}
        assert "mission_uuid" in kwonly_names, (
            "empty_state must declare `mission_uuid` as a keyword-only "
            "argument (S80 m04 PD109)."
        )
        return
    raise AssertionError("empty_state function not found.")


# ---------------------------------------------------------------------------
# Worker converter — propagation from MissionRow.id.
# ---------------------------------------------------------------------------


def test_worker_converter_propagates_mission_uuid_from_row_id() -> None:
    """``convert_row_to_state`` must call ``empty_state(mission_uuid=row.id, ...)``.

    The TraceLab boundary contract names ``MissionRow.id`` as the
    system-assigned UUID and ``MissionRow.mission_id`` as the
    human-readable slug. Pre-S80 the worker converter only forwarded the
    slug, dropping the UUID at the boundary; this guard pins the
    propagation so the slot cannot silently regress.
    """
    tree = _module_ast(WORKER_CONVERTER_PATH)
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef) and node.name == "convert_row_to_state"
        ):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "empty_state"
            ):
                for kw in sub.keywords:
                    if kw.arg != "mission_uuid":
                        continue
                    value = kw.value
                    # Accept `row.id` directly OR `row.id or None` etc.
                    if isinstance(value, ast.Attribute):
                        if (
                            isinstance(value.value, ast.Name)
                            and value.value.id == "row"
                            and value.attr == "id"
                        ):
                            return
                    raise AssertionError(
                        "empty_state mission_uuid kwarg in convert_row_to_state "
                        "must read from `row.id` (S80 m04 PD109)."
                    )
        raise AssertionError(
            "convert_row_to_state must call empty_state with mission_uuid kwarg."
        )
    raise AssertionError("convert_row_to_state function not found.")


# ---------------------------------------------------------------------------
# Persistence layer — save_report / save_protocol signatures.
# ---------------------------------------------------------------------------


def _function_kwonly_names(path: Path, function_name: str) -> set[str]:
    tree = _module_ast(path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return {arg.arg for arg in node.args.kwonlyargs}
    raise AssertionError(f"{function_name} function not found in {path}.")


def test_save_report_carries_mission_uuid_kwonly_argument() -> None:
    kwonly = _function_kwonly_names(GENERATOR_PATH, "save_report")
    assert "mission_uuid" in kwonly, (
        "save_report must declare `mission_uuid` as a keyword-only argument "
        "(S80 m04 PD109)."
    )


def test_save_protocol_carries_mission_uuid_kwonly_argument() -> None:
    kwonly = _function_kwonly_names(GENERATOR_PATH, "save_protocol")
    assert "mission_uuid" in kwonly, (
        "save_protocol must declare `mission_uuid` as a keyword-only argument "
        "(S80 m04 PD109)."
    )


def test_filename_unique_suffix_helper_is_top_level() -> None:
    """The filename disambiguator helper must be a top-level function.

    Hiding the helper inside ``save_report`` / ``save_protocol`` would
    duplicate the suffix logic across both call sites — pinning the helper
    at module scope keeps the contract symmetric.
    """
    tree = _module_ast(GENERATOR_PATH)
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_filename_unique_suffix"
        ):
            return
    raise AssertionError(
        "_filename_unique_suffix must exist as a top-level function in "
        "deepsearch/report/generator.py (S80 m04 PD109)."
    )


# ---------------------------------------------------------------------------
# Finalize node — both save_report and save_protocol calls receive UUID.
# ---------------------------------------------------------------------------


def _save_call_kwarg_names(function_node: ast.FunctionDef | ast.AsyncFunctionDef,
                            target_name: str) -> set[str]:
    """Return the set of kwarg names passed to a Call(name=target_name)."""
    kwarg_names: set[str] = set()
    for sub in ast.walk(function_node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == target_name
        ):
            for kw in sub.keywords:
                if kw.arg:
                    kwarg_names.add(kw.arg)
    return kwarg_names


def test_codeagent_finalize_node_threads_uuid_to_save_report() -> None:
    tree = _module_ast(FINALIZE_NODE_PATH)
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "codeagent_finalize_node"
        ):
            continue
        kwargs = _save_call_kwarg_names(node, "save_report")
        assert "mission_uuid" in kwargs, (
            "codeagent_finalize_node must pass `mission_uuid=...` to "
            "save_report (S80 m04 PD109). Without the kwarg, the persistence "
            "layer falls back to a random suffix and the deterministic "
            "per-UUID filename property is lost."
        )
        return
    raise AssertionError("codeagent_finalize_node not found.")


def test_codeagent_finalize_node_threads_uuid_to_save_protocol() -> None:
    tree = _module_ast(FINALIZE_NODE_PATH)
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "codeagent_finalize_node"
        ):
            continue
        kwargs = _save_call_kwarg_names(node, "save_protocol")
        assert "mission_uuid" in kwargs, (
            "codeagent_finalize_node must pass `mission_uuid=...` to "
            "save_protocol (S80 m04 PD109)."
        )
        return
    raise AssertionError("codeagent_finalize_node not found.")


# ---------------------------------------------------------------------------
# Runtime — the suffix helper actually behaves.
# ---------------------------------------------------------------------------


def test_runtime_suffix_uses_first_8_hex_of_uuid() -> None:
    from deepsearch.report.generator import _filename_unique_suffix

    suffix = _filename_unique_suffix("93201955-71e8-47e1-8124-81244e892924")
    assert suffix == "93201955", (
        "When a UUID is present, the suffix must be the first 8 hex chars "
        "with hyphens stripped, lowercased."
    )


def test_runtime_suffix_handles_uppercase_uuid() -> None:
    from deepsearch.report.generator import _filename_unique_suffix

    suffix = _filename_unique_suffix("93201955-71E8-47E1-8124-81244E892924")
    assert suffix == "93201955"


def test_runtime_suffix_falls_back_to_random_when_uuid_absent() -> None:
    from deepsearch.report.generator import _filename_unique_suffix

    suffix_a = _filename_unique_suffix(None)
    suffix_b = _filename_unique_suffix(None)
    assert len(suffix_a) == 8
    # Two random suffixes will almost certainly differ (1 in 2^32 collision).
    assert suffix_a != suffix_b


def test_runtime_suffix_falls_back_when_uuid_empty_string() -> None:
    from deepsearch.report.generator import _filename_unique_suffix

    # Empty string is falsy → random fallback path.
    suffix = _filename_unique_suffix("")
    assert len(suffix) == 8
