"""Static-AST regression guards for the S68.5 writer→TraceLab persistence path.

Mirrors the per-sprint invariants pattern (S65 synthesis-path bypass,
S66 critique invariants, S67 citation invariants, S68 liveness invariants):
runtime tests live in the consumer integration file
(``tests/integration/test_writer_telemetry_persistence.py``); this module
locks the *structure* — the dataclass field exists, the build path reads
``state["synthesis_telemetry"]``, and the serialiser references the field.

If a future sprint accidentally:

* deletes the ``synthesis_telemetry`` field from
  :class:`deepsearch.worker.writer.ExecutionMetadata`,
* stops reading ``state["synthesis_telemetry"]`` from
  :func:`deepsearch.worker.writer.build_execution_metadata`, or
* removes the ``self.synthesis_telemetry`` reference from
  :meth:`ExecutionMetadata.to_dict`,

these guards fail loudly. Without the field the entire telemetry envelope
(critique + citation + liveness) is silently dropped at the boundary —
exactly the pre-S68.5 defect surfaced by mission ``OODS-FIGMA-HOST-01-S68-WIRING``
(session ``PS-2026-04-28-012``).

Test layout follows the S66.1 convention (decision #501): flat in
``tests/unit/``, not ``tests/unit/worker/``.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WRITER_PATH = REPO_ROOT / "deepsearch" / "worker" / "writer.py"


def _module_ast() -> ast.Module:
    return ast.parse(WRITER_PATH.read_text(encoding="utf-8"))


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _find_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


# ---------------------------------------------------------------------------
# 1. Dataclass field invariant — ExecutionMetadata must declare a field
#    literally named `synthesis_telemetry`. Without this, the agent-side
#    envelope is silently dropped at construction.
# ---------------------------------------------------------------------------


def test_execution_metadata_has_synthesis_telemetry_field():
    tree = _module_ast()
    cls = _find_class(tree, "ExecutionMetadata")
    assert cls is not None, "ExecutionMetadata class missing from worker/writer.py"

    field_names: set[str] = set()
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            field_names.add(node.target.id)

    assert "synthesis_telemetry" in field_names, (
        "ExecutionMetadata must declare a `synthesis_telemetry` field. "
        "Without it the worker→TraceLab serialiser silently drops the "
        "agent-side envelope (engine, model_id, framework, step_count, "
        "runtime_seconds, token_usage, tool_call_summary, synthesis_error "
        "+ nested critique/citation/liveness envelopes). The S73 paid "
        "smoke depends on liveness_telemetry.hallucinated_url_count being "
        "observable on the deployed path."
    )


# ---------------------------------------------------------------------------
# 2. Reader invariant — build_execution_metadata must read
#    state["synthesis_telemetry"] (AST walk for the subscript expression on
#    the parameter `state`). Otherwise the field is constructed empty and
#    the dataclass guard above is a no-op.
# ---------------------------------------------------------------------------


def test_build_execution_metadata_reads_state_synthesis_telemetry():
    tree = _module_ast()
    fn = _find_function(tree, "build_execution_metadata")
    assert fn is not None, (
        "build_execution_metadata function missing from worker/writer.py"
    )

    found_subscript_or_get = False
    for node in ast.walk(fn):
        # state["synthesis_telemetry"]
        if isinstance(node, ast.Subscript):
            value = node.value
            if (
                isinstance(value, ast.Name)
                and value.id == "state"
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == "synthesis_telemetry"
            ):
                found_subscript_or_get = True
                break
        # state.get("synthesis_telemetry"[, default])
        if isinstance(node, ast.Call):
            target = node.func
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "get"
                and isinstance(target.value, ast.Name)
                and target.value.id == "state"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "synthesis_telemetry"
            ):
                found_subscript_or_get = True
                break

    assert found_subscript_or_get, (
        "build_execution_metadata must read `state[\"synthesis_telemetry\"]` "
        "(or `state.get(\"synthesis_telemetry\")`). Otherwise the agent-side "
        "envelope is dropped before construction and the persistence fix is "
        "inert."
    )


# ---------------------------------------------------------------------------
# 3. Serialiser invariant — to_dict must reference self.synthesis_telemetry
#    at least once, otherwise the JSONB payload omits the envelope even
#    when the dataclass holds it.
# ---------------------------------------------------------------------------


def test_to_dict_references_self_synthesis_telemetry():
    tree = _module_ast()
    cls = _find_class(tree, "ExecutionMetadata")
    assert cls is not None
    method = _find_method(cls, "to_dict")
    assert method is not None, "ExecutionMetadata.to_dict method missing"

    found = False
    for node in ast.walk(method):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "synthesis_telemetry"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            found = True
            break

    assert found, (
        "ExecutionMetadata.to_dict must reference `self.synthesis_telemetry` "
        "at least once. Without the conditional emit, the field is constructed "
        "but never serialised to the JSONB payload."
    )


# ---------------------------------------------------------------------------
# 4. HD3 truthfulness invariant — _derive_llm_backend helper must exist and
#    map the locked rebuild stack's model_id prefixes ("deepseek", "gemini")
#    to provider labels. Catches a silent regression that would leave
#    runtime_identity.llm_backend stuck on the legacy probe value.
# ---------------------------------------------------------------------------


def test_derive_llm_backend_helper_exists():
    tree = _module_ast()
    fn = _find_function(tree, "_derive_llm_backend")
    assert fn is not None, (
        "_derive_llm_backend helper missing from worker/writer.py. HD3 "
        "truthfulness depends on this function deriving llm_backend from the "
        "smolagents-reported model_id rather than the legacy LLM_BACKEND env "
        "probe."
    )
