"""Static-AST regression guards for the S79 m02 pre-acceptance completeness gate.

15th-consecutive-sprint RD2 layered-guard contribution (continuation of
test_s79_m01_invariants.py). Pins the load-bearing invariants from PD97:

* ``deepsearch/agent/tools/completeness_check.py`` exists and defines a
  ``completeness_check`` callable (m02.a).
* ``codeagent_runner.build_codeagent_tools`` accepts a ``deliverable_spec``
  parameter and wires ``completeness_check`` into the bundle.tools list (m02.a).
* ``run_codeagent_synthesis`` accepts and threads ``deliverable_spec`` into
  ``build_codeagent_tools`` (m02.a).
* ``contract_to_prompt.resolve_deliverables`` is exported (m02.a wiring).
* ``contract_to_prompt`` system prompt template includes the MANDATORY
  completeness_check clause (m02.b).
* ``coverage.py`` exports the renamed source-coverage symbols
  (``SourceCoverageGateResult``, ``aggregate_source_coverage``,
  ``coerce_source_coverage_gate_result``, ``missing_source_coverage_gate_result``)
  and no longer exports the old names (m02.c).
* ``codeagent_finalize_node`` populates ``synthesis_telemetry["deliverable_coverage"]``
  via a ``completeness_check`` call (m02.c).

These guards fail loudly if a future sprint:

* deletes the completeness_check tool or its wiring,
* drops the deliverable_spec parameter from build_codeagent_tools or run_codeagent_synthesis,
* removes the MANDATORY clause from contract_to_prompt,
* re-introduces the pre-rename ``aggregate_coverage`` symbol,
* drops the ``deliverable_coverage`` telemetry surface.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPSEARCH_ROOT = REPO_ROOT / "deepsearch"
COMPLETENESS_CHECK_PATH = DEEPSEARCH_ROOT / "agent" / "tools" / "completeness_check.py"
CODEAGENT_RUNNER_PATH = DEEPSEARCH_ROOT / "agent" / "codeagent_runner.py"
CODEAGENT_FINALIZE_PATH = DEEPSEARCH_ROOT / "agent" / "codeagent_finalize_node.py"
CONTRACT_TO_PROMPT_PATH = DEEPSEARCH_ROOT / "agent" / "contract_to_prompt.py"
COVERAGE_PATH = DEEPSEARCH_ROOT / "agent" / "coverage.py"


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# m02.a — completeness_check tool exists and is wired through the runner.
# ---------------------------------------------------------------------------


def test_completeness_check_module_exists() -> None:
    assert COMPLETENESS_CHECK_PATH.exists(), (
        "deepsearch/agent/tools/completeness_check.py must exist (S79 m02.a NEW tool)"
    )


def test_completeness_check_defines_callable() -> None:
    tree = _module_ast(COMPLETENESS_CHECK_PATH)
    callables = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "completeness_check" in callables, (
        "completeness_check.py must define a `completeness_check` function (m02.a contract)"
    )


def test_build_codeagent_tools_accepts_deliverable_spec() -> None:
    """``build_codeagent_tools`` must take a ``deliverable_spec`` parameter (m02.a)."""
    tree = _module_ast(CODEAGENT_RUNNER_PATH)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_codeagent_tools":
            arg_names = [arg.arg for arg in node.args.args + node.args.kwonlyargs]
            assert "deliverable_spec" in arg_names, (
                "build_codeagent_tools must accept a `deliverable_spec` parameter "
                "(m02.a wiring contract)"
            )
            return
    raise AssertionError("build_codeagent_tools function not found")


def test_run_codeagent_synthesis_accepts_deliverable_spec() -> None:
    """``run_codeagent_synthesis`` must thread ``deliverable_spec`` to the factory."""
    tree = _module_ast(CODEAGENT_RUNNER_PATH)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_codeagent_synthesis":
            arg_names = [arg.arg for arg in node.args.args + node.args.kwonlyargs]
            assert "deliverable_spec" in arg_names, (
                "run_codeagent_synthesis must accept a `deliverable_spec` parameter"
            )
            return
    raise AssertionError("run_codeagent_synthesis function not found")


def test_completeness_check_in_tools_bundle() -> None:
    """The returned tools list must contain ``completeness_check``."""
    tree = _module_ast(CODEAGENT_RUNNER_PATH)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_codeagent_tools":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Call):
                    for kw in sub.value.keywords:
                        if kw.arg == "tools" and isinstance(kw.value, ast.List):
                            element_names = [
                                elt.id
                                for elt in kw.value.elts
                                if isinstance(elt, ast.Name)
                            ]
                            assert "completeness_check" in element_names, (
                                "build_codeagent_tools must include completeness_check "
                                "in its returned tools list (m02.a wiring)"
                            )
                            return
    raise AssertionError(
        "build_codeagent_tools function or its tools-list Return not found"
    )


def test_resolve_deliverables_exported_from_contract_to_prompt() -> None:
    """``resolve_deliverables`` is the public hook m02.a uses to feed the spec
    to the agent's tool factory."""
    from deepsearch.agent import contract_to_prompt

    assert hasattr(contract_to_prompt, "resolve_deliverables"), (
        "contract_to_prompt.resolve_deliverables must be public (m02.a)"
    )
    assert "resolve_deliverables" in contract_to_prompt.__all__, (
        "resolve_deliverables must be on contract_to_prompt.__all__"
    )


def test_codeagent_finalize_node_imports_resolve_deliverables() -> None:
    """The finalize node must import the public resolver to feed the spec into
    the runner."""
    source = CODEAGENT_FINALIZE_PATH.read_text(encoding="utf-8")
    assert "resolve_deliverables" in source, (
        "codeagent_finalize_node must import resolve_deliverables (m02.a wiring)"
    )


# ---------------------------------------------------------------------------
# m02.b — MANDATORY completeness_check clause in the system prompt.
# ---------------------------------------------------------------------------


def test_contract_to_prompt_contains_mandatory_completeness_clause() -> None:
    """The MANDATORY clause is the load-bearing behavior-shaping nudge for m02.b."""
    source = CONTRACT_TO_PROMPT_PATH.read_text(encoding="utf-8")
    assert "PRE-ACCEPTANCE COMPLETENESS CHECK" in source, (
        "contract_to_prompt template must include the PRE-ACCEPTANCE COMPLETENESS "
        "CHECK section header (m02.b)"
    )
    assert "completeness_check(draft_markdown" in source, (
        "MANDATORY clause must instruct the agent to call completeness_check(draft_markdown=...)"
    )
    assert "MANDATORY" in source, (
        "MANDATORY language must remain in the prompt (consistent with the "
        "[cite:URL] section's voice)"
    )


def test_contract_to_prompt_lists_completeness_check_in_tools() -> None:
    """The Tools Available section must enumerate completeness_check so the
    agent knows to call it."""
    source = CONTRACT_TO_PROMPT_PATH.read_text(encoding="utf-8")
    assert "`completeness_check(" in source, (
        "Tools Available section must enumerate completeness_check (m02.b discoverability)"
    )


def test_contract_to_prompt_lists_scholar_search_in_tools() -> None:
    """S79 m01.a wiring: the prompt should also surface scholar_search now that
    the tool is in the agent's bag (joint m01.a + m02.b coverage)."""
    source = CONTRACT_TO_PROMPT_PATH.read_text(encoding="utf-8")
    assert "`scholar_search(" in source, (
        "Tools Available section must enumerate scholar_search (m01.a discoverability)"
    )


# ---------------------------------------------------------------------------
# m02.c — coverage_gate honest renaming + deliverable_coverage telemetry.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "SourceCoverageGateResult",
        "aggregate_source_coverage",
        "missing_source_coverage_gate_result",
        "coerce_source_coverage_gate_result",
    ],
)
def test_coverage_module_exports_renamed_symbol(name: str) -> None:
    from deepsearch.agent import coverage

    assert hasattr(coverage, name), (
        f"coverage.py must export {name} after S79 m02.c rename (PD97 Option B)"
    )


@pytest.mark.parametrize(
    "old_name",
    [
        "CoverageGateResult",
        "aggregate_coverage",
        "missing_coverage_gate_result",
        "coerce_coverage_gate_result",
    ],
)
def test_coverage_module_no_longer_exports_pre_rename_symbol(old_name: str) -> None:
    """Belt-and-suspenders: the pre-rename symbols must be gone (no aliases per
    feedback_no_hacky_fallbacks.md "one config name everywhere")."""
    from deepsearch.agent import coverage

    assert not hasattr(coverage, old_name), (
        f"coverage.py must NOT export {old_name} — m02.c is a clean rename "
        f"(no backwards-compat aliases per feedback_no_hacky_fallbacks)"
    )


def test_synthesis_telemetry_includes_deliverable_coverage_field() -> None:
    """The codeagent_finalize_node must wire ``deliverable_coverage`` onto
    synthesis_telemetry (m02.c). PD102 verification expectation #10."""
    source = CODEAGENT_FINALIZE_PATH.read_text(encoding="utf-8")
    assert 'synthesis_telemetry["deliverable_coverage"]' in source, (
        "codeagent_finalize_node must populate "
        'synthesis_telemetry["deliverable_coverage"] (m02.c)'
    )
    assert "completeness_check(" in source, (
        "synthesis_telemetry deliverable_coverage must be computed via completeness_check"
    )


def test_coverage_gate_failures_field_name_preserved_on_routing_decision() -> None:
    """PD102 #13 — the public ``coverage_gate_failures`` field name on
    ``deepsearch.agent.evaluation.RoutingDecision`` is the S75 carry; m02.c
    rename only touches internal coverage.py symbols."""
    from deepsearch.agent.evaluation import RoutingDecision

    assert "coverage_gate_failures" in RoutingDecision.__dataclass_fields__, (
        "RoutingDecision.coverage_gate_failures must remain (S75 carry / PD102 #13)"
    )
