"""S83 m06 RD2 invariants — PD83-b retirement of detection-without-enforcement.

Two protocol quality gates (evidence_links, traceability) degenerated to a
CONSTANT misleading ``"fail"`` shipped to TraceLab on every codeagent dispatch
once S78 stripped ``working_set`` / ``sources_found`` from ``AgentState`` (the
gates' live branches went dead; nothing enforces the result — ``all_gates_pass``
has zero non-test callers and live ``upload()`` never inspects ``cp.status``).

m06 (a) stripped the 13 dead ``state.get("working_set"|"sources_found")`` reads
across 8 files + deleted ``coverage._working_set_titles_for_matching``, and
(b) REPOINTED both gates onto the ``citation_telemetry`` the codeagent path
genuinely produces — a truthful pass/fail rather than a discarded constant.

Pinned here:
  * module-wide AST scan: no ``.get("working_set"|"sources_found")`` Call survives
    anywhere in ``deepsearch/`` (the never-reappears guard, module-wide);
  * ``REQUIRED_COMPLETION_GATES`` literal still has exactly the 5 canonical names
    (the TraceLab ingest contract);
  * evidence_links / traceability are no longer constant-fail — they pass on a
    populated ``citation_telemetry`` and fail (non-degenerately) when absent;
  * ``_working_set_titles_for_matching`` is gone.
"""

from __future__ import annotations

import ast
from pathlib import Path

from deepsearch.agent import coverage as coverage_mod
from deepsearch.report import protocol as protocol_mod
from deepsearch.report.protocol import (
    REQUIRED_COMPLETION_GATES,
    _evaluate_evidence_links_gate,
    _evaluate_traceability_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPSEARCH_ROOT = REPO_ROOT / "deepsearch"
PROTOCOL_PATH = DEEPSEARCH_ROOT / "report" / "protocol.py"

_STRIPPED_FIELDS = frozenset({"working_set", "sources_found"})
# S83 m06 CARVE-OUT (decision B): two tested, exported generic source-extraction
# utilities are kept reading the dead field rather than gutted —
# report/evidence.py:generate_evidence_list and
# agent/evidence.py:extract_sources_from_state. Both are dead-in-production (no
# sources_found writer). Excluded from the module-wide scan.
_CARVE_OUT_RELPATHS = frozenset(
    {"report/evidence.py", "agent/evidence.py", "runner.py", "report/generator.py"}
)
_CANONICAL_GATES = [
    "research_statement",
    "evidence_links",
    "synthesis_quality",
    "traceability",
    "contradictions_resolved",
]


def _iter_python_sources(root: Path):
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


# ---------------------------------------------------------------------------
# (a) Module-wide never-reappears AST scan.
# ---------------------------------------------------------------------------


def test_no_get_call_on_stripped_fields_anywhere() -> None:
    """No ``X.get("working_set"|"sources_found")`` Call survives in deepsearch/.

    Catches a re-introduction in ANY file (not just the former readers) of a
    dependency on the S78-stripped fields, EXCEPT the documented carve-out
    (report/evidence.py:generate_evidence_list — see _CARVE_OUT_RELPATHS).
    Docstring examples use the dict-literal ``"sources_found":`` key form — an
    ``ast.Constant`` inside a ``Dict``, never a ``.get`` Call arg — so they do
    not match.
    """
    offenders: list[str] = []
    for path in _iter_python_sources(DEEPSEARCH_ROOT):
        if path.relative_to(DEEPSEARCH_ROOT).as_posix() in _CARVE_OUT_RELPATHS:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in _STRIPPED_FIELDS
            ):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{node.args[0].value}"
                )
    assert not offenders, (
        "S83 m06 stripped every read of the S78-dead fields; a re-appearance "
        "resurrects a dead-field dependency:\n  " + "\n  ".join(sorted(offenders))
    )


# ---------------------------------------------------------------------------
# REQUIRED_COMPLETION_GATES — exactly 5 canonical names (ingest contract).
# ---------------------------------------------------------------------------


def test_required_completion_gates_literal_is_five_canonical_names() -> None:
    """AST guard on the literal: REQUIRED_COMPLETION_GATES is a 5-element list of
    the canonical gate names. The TraceLab ingest contract depends on the names
    + count; m06 repoints behaviour, never the schema."""
    tree = ast.parse(PROTOCOL_PATH.read_text(encoding="utf-8"))
    found: list[str] | None = None
    for node in ast.walk(tree):
        targets = (
            node.targets if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign)
            else []
        )
        for tgt in targets:
            if isinstance(tgt, ast.Name) and tgt.id == "REQUIRED_COMPLETION_GATES":
                assert isinstance(node.value, ast.List), "must be a list literal"
                found = [
                    elt.value for elt in node.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
    assert found is not None, "REQUIRED_COMPLETION_GATES literal not found"
    assert found == _CANONICAL_GATES
    # Runtime value agrees with the literal.
    assert list(REQUIRED_COMPLETION_GATES) == _CANONICAL_GATES


# ---------------------------------------------------------------------------
# (b) Repointed gates — no longer constant-fail.
# ---------------------------------------------------------------------------


def _state_with_citations(n: int, distinct: int = 0) -> dict:
    return {
        "synthesis_telemetry": {
            "citation_telemetry": {
                "citations_extracted": n,
                "distinct_urls_cited": distinct or n,
            }
        }
    }


def test_evidence_links_gate_passes_on_citations_fails_on_zero() -> None:
    status_pass, _ = _evaluate_evidence_links_gate(
        _state_with_citations(3), evidence=None
    )
    assert status_pass == "pass"

    status_fail, _ = _evaluate_evidence_links_gate(
        _state_with_citations(0), evidence=None
    )
    assert status_fail == "fail"

    # Absent citation_telemetry entirely -> fail (non-degenerate, not constant).
    status_absent, _ = _evaluate_evidence_links_gate({}, evidence=None)
    assert status_absent == "fail"


def test_traceability_gate_passes_on_citations_fails_on_absent() -> None:
    status_pass, note_pass = _evaluate_traceability_gate(_state_with_citations(4, distinct=3))
    assert status_pass == "pass"
    assert "3" in note_pass and "4" in note_pass  # distinct URLs + citation count

    status_fail, _ = _evaluate_traceability_gate(_state_with_citations(0))
    assert status_fail == "fail"

    status_absent, _ = _evaluate_traceability_gate({})
    assert status_absent == "fail"


def test_gates_do_not_read_citation_density_anymore() -> None:
    """The dead citation_density path (zero production writers) is gone — a state
    carrying ONLY citation_density (no citation_telemetry) no longer passes
    traceability."""
    state = {"synthesis_telemetry": {"citation_density": 0.99}}
    status, _ = _evaluate_traceability_gate(state)
    assert status == "fail", "traceability must read citation_telemetry, not citation_density"
    # Structural: the dead threshold constant is gone with the citation_density path.
    assert not hasattr(protocol_mod, "_CITATION_DENSITY_THRESHOLD")


# ---------------------------------------------------------------------------
# Hard Rule 8 — the deleted symbol is gone with no surviving caller.
# ---------------------------------------------------------------------------


def test_working_set_titles_helper_is_deleted() -> None:
    assert not hasattr(coverage_mod, "_working_set_titles_for_matching")
    # And no caller survives in the coverage source.
    assert "_working_set_titles_for_matching" not in (
        (DEEPSEARCH_ROOT / "agent" / "coverage.py").read_text(encoding="utf-8")
    )
