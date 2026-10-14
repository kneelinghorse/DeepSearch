"""S72 PD22 / PD24 — static AST guards for the run-integrity surface.

RD2 ninth-consecutive sprint contribution. These guards are intentionally
the "skeleton" check against the m01 surface: the runtime tests in
``tests/unit/test_codeagent_finalize_node.py`` and the HD4 round-trip in
``tests/integration/test_report_metadata_persistence.py`` cover behaviour;
this file locks the structural shape so a future refactor cannot quietly
delete the contract.

What's locked:

1. :class:`deepsearch.report.protocol.ReportMetadata` exists, is a Pydantic
   ``BaseModel`` subclass, and exposes the 4 PD22 header fields plus
   :pyattr:`forensic`. PD22 explicitly held ``build_hash`` and
   ``schema_version`` out of the at-a-glance header — they live under
   :pyattr:`forensic`.
2. :class:`deepsearch.report.protocol.MissionProtocolComplete` carries an
   :pyattr:`report_metadata` field (Optional[ReportMetadata], default None).
3. :func:`deepsearch.report.protocol.MissionProtocolComplete.to_dict`
   serialises the nested block — drift here breaks the worker JSONB write.
4. ``deepsearch.agent.codeagent_finalize_node`` exposes both
   :func:`_build_report_metadata` and :func:`_render_report_metadata_preamble`
   helpers. The single-source-of-truth guarantee depends on both being
   reachable.
5. ``deepsearch.agent.codeagent_runner._accumulate_source`` carries the
   PD24 body_source ratchet — a textual marker (``ratchet`` / ``never
   downgrade``) plus the ``body_source`` field appears in the merge body.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures: parse the protocol module + finalize-node module + runner module
# once per session.
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "deepsearch" / "report" / "protocol.py"
FINALIZE_NODE_PATH = REPO_ROOT / "deepsearch" / "agent" / "codeagent_finalize_node.py"
RUNNER_PATH = REPO_ROOT / "deepsearch" / "agent" / "codeagent_runner.py"


@pytest.fixture(scope="module")
def protocol_tree() -> ast.AST:
    return ast.parse(PROTOCOL_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def finalize_tree() -> ast.AST:
    return ast.parse(FINALIZE_NODE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runner_tree() -> ast.AST:
    return ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))


def _find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _class_assigned_names(node: ast.ClassDef) -> set[str]:
    """Collect names assigned at the class body level (Pydantic field names)."""
    names: set[str] = set()
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _class_method_names(node: ast.ClassDef) -> set[str]:
    return {
        stmt.name
        for stmt in node.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


# ---------------------------------------------------------------------------
# 1. ReportMetadata exists with the PD22 contract.
# ---------------------------------------------------------------------------


def test_report_metadata_class_exists(protocol_tree: ast.AST) -> None:
    cls = _find_class(protocol_tree, "ReportMetadata")
    assert cls is not None, (
        "deepsearch.report.protocol.ReportMetadata must exist — it is the "
        "single truth surface that supersedes the legacy execution_metadata "
        "fields scheduled for deletion at s72-m02."
    )


def test_report_metadata_subclasses_pydantic_basemodel(protocol_tree: ast.AST) -> None:
    cls = _find_class(protocol_tree, "ReportMetadata")
    assert cls is not None
    base_names = []
    for base in cls.bases:
        if isinstance(base, ast.Name):
            base_names.append(base.id)
        elif isinstance(base, ast.Attribute):
            base_names.append(base.attr)
    assert "BaseModel" in base_names, (
        "ReportMetadata must subclass pydantic.BaseModel so the to_dict / "
        "model_dump_json round-trip works (HD4 round-trip relies on it)."
    )


def test_report_metadata_has_pd22_header_fields(protocol_tree: ast.AST) -> None:
    """PD22 — exactly these 4 at-a-glance fields are header surface."""

    cls = _find_class(protocol_tree, "ReportMetadata")
    assert cls is not None
    fields = _class_assigned_names(cls)
    expected = {
        "sources_collected",
        "references",
        "agent_steps_used",
        "agent_steps_max",
        "runtime_seconds",
        "forensic",
    }
    missing = expected - fields
    assert not missing, (
        f"ReportMetadata is missing PD22 fields: {sorted(missing)}. "
        f"PD22 locked the at-a-glance header to "
        f"{{sources_collected, references, agent_steps_used, agent_steps_max, runtime_seconds}} "
        f"plus a forensic dict; the deletion at s72-m02 assumes all six exist."
    )


def test_report_metadata_has_to_dict(protocol_tree: ast.AST) -> None:
    cls = _find_class(protocol_tree, "ReportMetadata")
    assert cls is not None
    assert "to_dict" in _class_method_names(cls), (
        "ReportMetadata must expose to_dict() — the parent "
        "MissionProtocolComplete.to_dict override calls it for nested "
        "serialisation."
    )


# ---------------------------------------------------------------------------
# 2. MissionProtocolComplete carries the report_metadata field.
# ---------------------------------------------------------------------------


def test_mission_protocol_carries_report_metadata_field(protocol_tree: ast.AST) -> None:
    cls = _find_class(protocol_tree, "MissionProtocolComplete")
    assert cls is not None
    assert "report_metadata" in _class_assigned_names(cls), (
        "MissionProtocolComplete must declare the report_metadata field — "
        "without it codeagent_finalize_node has nowhere to attach the new "
        "run-integrity block."
    )


def test_mission_protocol_to_dict_serialises_report_metadata(
    protocol_tree: ast.AST,
) -> None:
    """The custom to_dict() override must include a report_metadata write."""

    cls = _find_class(protocol_tree, "MissionProtocolComplete")
    assert cls is not None
    to_dict = next(
        (
            stmt
            for stmt in cls.body
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "to_dict"
        ),
        None,
    )
    assert to_dict is not None, "MissionProtocolComplete.to_dict must exist."

    report_metadata_handled = False
    for node in ast.walk(to_dict):
        if isinstance(node, ast.Subscript):
            slice_node = node.slice
            if (
                isinstance(slice_node, ast.Constant)
                and slice_node.value == "report_metadata"
            ):
                report_metadata_handled = True
                break
    assert report_metadata_handled, (
        "MissionProtocolComplete.to_dict must explicitly serialise "
        "report_metadata. Pydantic's model_dump alone produces a dict "
        "(fine), but the existing override pattern handles each nested "
        "model individually — the to_dict() round-trip relies on it."
    )


# ---------------------------------------------------------------------------
# 3. codeagent_finalize_node exposes _build_report_metadata +
#    _render_report_metadata_preamble.
# ---------------------------------------------------------------------------


def test_finalize_node_exposes_build_and_render_helpers(finalize_tree: ast.AST) -> None:
    function_names = {
        node.name
        for node in ast.walk(finalize_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for required in ("_build_report_metadata", "_render_report_metadata_preamble"):
        assert required in function_names, (
            f"deepsearch.agent.codeagent_finalize_node must expose "
            f"{required}() — the single-source-of-truth guarantee between "
            f"protocol.report_metadata and the markdown header preamble "
            f"depends on a single builder + a single renderer."
        )


def test_finalize_node_imports_report_metadata_from_protocol() -> None:
    """The finalize node must import ReportMetadata; otherwise it cannot
    populate ``protocol.report_metadata``.
    """

    from deepsearch.agent import codeagent_finalize_node as fnode

    source = inspect.getsource(fnode)
    assert "ReportMetadata" in source, (
        "codeagent_finalize_node must import ReportMetadata from "
        "deepsearch.report.protocol — drift here breaks the protocol "
        "wire-up."
    )


# ---------------------------------------------------------------------------
# 4. S87 m02 supersedes the PD24 revert with a bounded design: readable bodies
# stay internal, the model-facing fetch response is capped, and monotonic merge
# logic prevents later thin payloads from degrading richer evidence.
# ---------------------------------------------------------------------------


def test_runner_pairs_quality_discriminator_with_public_fetch_cap() -> None:
    """Full internal evidence is safe only with an explicit public payload cap."""

    runner_text = RUNNER_PATH.read_text(encoding="utf-8")
    assert "body_source" in runner_text
    assert "content_quality" in runner_text
    assert "SOURCE_FETCH_TOOL_TEXT_MAX_CHARS" in runner_text


def test_runner_content_merge_is_monotonic_by_quality_and_length() -> None:
    """A later thin or shorter payload cannot replace richer stored content."""

    runner_text = RUNNER_PATH.read_text(encoding="utf-8")
    assert "incoming_quality < existing_quality" in runner_text
    assert "incoming_content_chars < existing_content_chars" in runner_text
