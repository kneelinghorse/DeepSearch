"""Static-AST regression guards for the S71 Path A persistence seam.

Mirrors the layered-guard pattern from S65 (synthesis-path bypass),
S66 (critique invariants), S67 (citation invariants), S68 (liveness
invariants), and S68.5 (writer persistence invariants): runtime
behaviour lives in
``tests/integration/test_sources_collected_persistence.py`` and the
extended ``tests/unit/test_codeagent_finalize_node.py``; this module
locks the *structure* — the field exists on the protocol model, the
finalize node assigns it, and the writer's NUL-byte sanitiser is the
recursive form that traverses lists of dicts.

If a future sprint accidentally:

* deletes the ``sources_collected`` field from
  :class:`deepsearch.report.protocol.MissionProtocolComplete`,
* removes the ``protocol.sources_collected = ...`` assignment in
  :func:`deepsearch.agent.codeagent_finalize_node.codeagent_finalize_node`, or
* downgrades :func:`deepsearch.worker.writer._strip_null_bytes` to a
  non-recursive shape (so list-of-dict sanitisation regresses),

these guards fail loudly. Without the field the FActScore harness has
no probe substrate (PD4 short-circuit), and the rebuild roadmap §S71
"calibration anchor" goal is structurally unreachable.

Eighth-consecutive RD2 sprint applying the layered-guard pattern.

Test layout follows the S66.1 convention (decision #501): flat in
``tests/unit/``, not ``tests/unit/report/`` or ``tests/unit/agent/``.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "deepsearch" / "report" / "protocol.py"
FINALIZE_NODE_PATH = REPO_ROOT / "deepsearch" / "agent" / "codeagent_finalize_node.py"
WRITER_PATH = REPO_ROOT / "deepsearch" / "worker" / "writer.py"


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


# ---------------------------------------------------------------------------
# 1. Pydantic field invariant — MissionProtocolComplete must declare a
#    field literally named ``sources_collected``. Without this, the
#    runner-side accumulator has no persisted home and the FActScore
#    harness keeps short-circuiting on the empty-substrate path that
#    produced ``factscore=0.000`` at S70 m02.
# ---------------------------------------------------------------------------


def test_protocol_has_sources_collected_field() -> None:
    tree = _module_ast(PROTOCOL_PATH)
    cls = _find_class(tree, "MissionProtocolComplete")
    assert cls is not None, (
        "MissionProtocolComplete class missing from deepsearch/report/protocol.py"
    )

    field_names: set[str] = set()
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            field_names.add(node.target.id)

    assert "sources_collected" in field_names, (
        "MissionProtocolComplete must declare a `sources_collected` field. "
        "Without it, the runner-side accumulator from "
        "deepsearch.agent.codeagent_runner.SynthesisResult.sources_collected "
        "has no persisted home, the FActScore harness short-circuits at PD4 "
        "(empty sources_collected[]), and S71 calibration substrate blocker "
        "(2) regresses."
    )


def test_protocol_sources_collected_is_additive_not_replacing_evidence() -> None:
    """PD9 carry-forward — additive-only schema. Both ``evidence`` and
    ``sources_collected`` must coexist; sources_collected complements
    evidence, it does not replace it.
    """

    tree = _module_ast(PROTOCOL_PATH)
    cls = _find_class(tree, "MissionProtocolComplete")
    assert cls is not None

    field_names: set[str] = set()
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            field_names.add(node.target.id)

    for required in ("evidence", "citations", "sources_collected"):
        assert required in field_names, (
            f"PD9 additive-only constraint: `{required}` must remain on "
            "MissionProtocolComplete alongside the other persisted-source "
            "surfaces. None of evidence/citations/sources_collected may "
            "displace another."
        )


# ---------------------------------------------------------------------------
# 2. Finalize-node assignment invariant — codeagent_finalize_node must
#    contain an attribute assignment ``protocol.sources_collected = ...``.
#    Otherwise the field is constructed empty and the Pydantic guard
#    above is a no-op.
# ---------------------------------------------------------------------------


def test_finalize_node_assigns_protocol_sources_collected() -> None:
    tree = _module_ast(FINALIZE_NODE_PATH)
    fn = _find_function(tree, "codeagent_finalize_node")
    assert fn is not None, (
        "codeagent_finalize_node function missing from "
        "deepsearch/agent/codeagent_finalize_node.py"
    )

    found = False
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "sources_collected"
            and isinstance(target.value, ast.Name)
            and target.value.id == "protocol"
        ):
            found = True
            break

    assert found, (
        "codeagent_finalize_node must assign "
        "`protocol.sources_collected = ...` after `generate_mission_protocol`. "
        "Without the assignment, the protocol model carries the default "
        "empty list at persistence time and the runner-side accumulator is "
        "silently dropped — exactly the substrate-blocker (2) the S71 sprint "
        "is closing."
    )


def test_finalize_node_assignment_reads_synthesis_result_accumulator() -> None:
    """The assignment must source from
    ``synthesis_result.sources_collected`` — that is the runtime carrier
    populated by ``run_codeagent_synthesis``. Locking the source side
    catches an accidental swap to an unrelated AgentState field.
    """

    tree = _module_ast(FINALIZE_NODE_PATH)
    fn = _find_function(tree, "codeagent_finalize_node")
    assert fn is not None

    found_synthesis_attribute = False
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and target.attr == "sources_collected"
            and isinstance(target.value, ast.Name)
            and target.value.id == "protocol"
        ):
            continue
        for descendant in ast.walk(node.value):
            if (
                isinstance(descendant, ast.Attribute)
                and descendant.attr == "sources_collected"
                and isinstance(descendant.value, ast.Name)
                and descendant.value.id == "synthesis_result"
            ):
                found_synthesis_attribute = True
                break
        if found_synthesis_attribute:
            break

    assert found_synthesis_attribute, (
        "The protocol.sources_collected assignment must read from "
        "`synthesis_result.sources_collected`. Any other source (e.g. "
        "state['sources_found'], working_set entries) loses the runner-side "
        "URL-deduplicated accumulator that web_search / source_fetch / "
        "url_liveness populate via _accumulate_source — the only shape "
        "the FActScore harness PD4 path expects."
    )


# ---------------------------------------------------------------------------
# 3. Worker writer NUL-byte sanitiser invariant — _strip_null_bytes must
#    recurse through both ``dict`` and ``list``, otherwise a list-of-dict
#    field like ``sources_collected`` carries embedded NULs into the
#    Postgres JSONB write and the insert fails (or, worse, the records
#    silently disappear depending on the driver).
# ---------------------------------------------------------------------------


def test_strip_null_bytes_recurses_through_lists_and_dicts() -> None:
    tree = _module_ast(WRITER_PATH)
    fn = _find_function(tree, "_strip_null_bytes")
    assert fn is not None, (
        "_strip_null_bytes helper missing from deepsearch/worker/writer.py"
    )

    seen_dict_branch = False
    seen_list_branch = False
    seen_self_recursion = False
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name) and target.id == "_strip_null_bytes":
                seen_self_recursion = True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "isinstance":
            if len(node.args) >= 2:
                second = node.args[1]
                if isinstance(second, ast.Name):
                    if second.id == "dict":
                        seen_dict_branch = True
                    elif second.id == "list":
                        seen_list_branch = True
                if isinstance(second, ast.Tuple):
                    for el in second.elts:
                        if isinstance(el, ast.Name):
                            if el.id == "dict":
                                seen_dict_branch = True
                            elif el.id == "list":
                                seen_list_branch = True

    assert seen_self_recursion, (
        "_strip_null_bytes must recurse on itself; without recursion the "
        "list-of-dict sources_collected payload is left un-sanitised on "
        "the JSONB write path."
    )
    assert seen_dict_branch and seen_list_branch, (
        "_strip_null_bytes must branch on both `dict` and `list` (the "
        "sources_collected field is a list of dicts). Losing either branch "
        "silently drops NUL-byte sanitisation for the new field shape."
    )
