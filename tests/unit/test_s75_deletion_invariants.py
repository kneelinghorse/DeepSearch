"""Static-AST + import-surface regression guards for the S75 source-level deletion.

Mirrors the layered-guard pattern from S65 (synthesis-path bypass), S66
(critique single-call), S67 (citation extractor), S68 (citation liveness),
S68.5 (writer persistence), S69 (factscore invariants), S71
(sources_collected persistence), and S74 (metadata truthfulness): runtime
tests live in the consumer test files; this module locks the *structure* of
the post-deletion codebase.

If a future sprint accidentally:

* re-introduces any of the 8 whole-file modules deleted at S75 m01
  (``_graph_builder``, ``research_node``, ``finding_dedup``, ``prompts``,
  ``finalize_node`` legacy, ``nodes``, ``synthesis_two_pass``,
  ``synthesis_evidence_packet``),
* re-introduces the ``_run_degraded_workflow`` fallback path in
  ``runner.py``,
* re-renders a ``## Telemetry`` table in
  ``deepsearch/report/generator.py``,
* re-emits the ``coverage_not_computed`` sentinel from
  ``agent/state.py`` initial state or ``agent/coverage.py`` fallback,
* re-adds ``loops_executed`` / ``sources_count`` / ``quality_gates_passed``
  to ``worker/writer.py`` ``ExecutionMetadata``,
* re-inserts the ``max_count=5`` cap into ``report/synthesis.py``
  ``_select_key_insights`` call,
* re-adds ``research_depth`` to ``worker/poller.py`` ``CLAIM_MISSION_SQL``
  (lockstep with TraceLab PR #213 alembic 028 DROP COLUMN), or
* drops the ``url_utils.resolve_source_url`` / ``extract_domain`` carve-out,

these guards fail loudly. Tests do not call any LLM, do not hit any network,
do not require pytest fixtures, and do not import the deleted modules
(verifying they are truly gone via ``importlib`` ImportError).

Test layout follows the S66.1 convention (decision #501): flat in
``tests/unit/``, not ``tests/unit/agent/``.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = REPO_ROOT / "deepsearch" / "report" / "generator.py"
SYNTHESIS_PATH = REPO_ROOT / "deepsearch" / "report" / "synthesis.py"
RUNNER_PATH = REPO_ROOT / "deepsearch" / "runner.py"
GRAPH_PATH = REPO_ROOT / "deepsearch" / "agent" / "graph.py"
URL_UTILS_PATH = REPO_ROOT / "deepsearch" / "agent" / "url_utils.py"
STATE_PATH = REPO_ROOT / "deepsearch" / "agent" / "state.py"
COVERAGE_PATH = REPO_ROOT / "deepsearch" / "agent" / "coverage.py"
WRITER_PATH = REPO_ROOT / "deepsearch" / "worker" / "writer.py"
WORKER_MAIN_PATH = REPO_ROOT / "deepsearch" / "worker" / "main.py"
POLLER_PATH = REPO_ROOT / "deepsearch" / "worker" / "poller.py"


DELETED_MODULES = (
    "deepsearch.agent._graph_builder",
    "deepsearch.agent.research_node",
    "deepsearch.agent.finding_dedup",
    "deepsearch.agent.prompts",
    "deepsearch.agent.finalize_node",
    "deepsearch.agent.nodes",
    "deepsearch.agent.synthesis_two_pass",
    "deepsearch.agent.synthesis_evidence_packet",
)


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Whole-file deletions: the 8 modules must be unimportable.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", DELETED_MODULES)
def test_deleted_module_cannot_be_imported(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", DELETED_MODULES)
def test_deleted_module_path_does_not_exist(module_name: str) -> None:
    relative = module_name.removeprefix("deepsearch.").replace(".", "/")
    path = REPO_ROOT / "deepsearch" / f"{relative.split('/', 1)[1]}.py"
    assert not path.exists(), f"deleted module re-introduced at {path}"


# ---------------------------------------------------------------------------
# `_run_degraded_workflow` fallback removed from runner.py.
# ---------------------------------------------------------------------------


def test_runner_has_no_run_degraded_workflow_function() -> None:
    tree = _module_ast(RUNNER_PATH)
    func_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_run_degraded_workflow" not in func_names, (
        "_run_degraded_workflow was deleted at S75 m01 (post-S64 codeagent path "
        "is the live path; the legacy sequential fallback ran the now-deleted "
        "research_node + reflection_node + legacy finalize_node)"
    )


def test_runner_does_not_import_deleted_modules() -> None:
    tree = _module_ast(RUNNER_PATH)
    forbidden = {
        "deepsearch.agent.research_node",
        "deepsearch.agent.finalize_node",
        "deepsearch.agent.nodes",
        "deepsearch.agent.prompts",
        "deepsearch.agent.synthesis_two_pass",
        "deepsearch.agent.synthesis_evidence_packet",
        "deepsearch.agent._graph_builder",
        "deepsearch.agent.finding_dedup",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in forbidden:
            raise AssertionError(
                f"runner.py re-introduced import from deleted module: {node.module}"
            )


# ---------------------------------------------------------------------------
# `## Telemetry` table + `_build_telemetry_section` removed from generator.py.
# ---------------------------------------------------------------------------


def test_generator_has_no_telemetry_section_function() -> None:
    tree = _module_ast(GENERATOR_PATH)
    func_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_build_telemetry_section" not in func_names, (
        "_build_telemetry_section was deleted at S75 m01 (PD34 carry from S74); "
        "the lying ## Telemetry table is replaced by the truthful "
        "_render_report_metadata_preamble + protocol.report_metadata surfaces"
    )


def test_generator_does_not_emit_telemetry_heading() -> None:
    src = GENERATOR_PATH.read_text(encoding="utf-8")
    assert "## Telemetry" not in src, (
        "deepsearch/report/generator.py emits a ## Telemetry heading; the "
        "S75 m01 deletion was supposed to strip the section entirely"
    )


# ---------------------------------------------------------------------------
# `coverage_not_computed` sentinel killed at both writer sites.
# ---------------------------------------------------------------------------


def test_state_has_no_coverage_not_computed_sentinel() -> None:
    src = STATE_PATH.read_text(encoding="utf-8")
    assert "coverage_not_computed" not in src, (
        "agent/state.py initial-state sentinel re-introduced; PD43 verification "
        "expects execution_metadata.final_outcome.coverage_gate_failures == [] "
        "on the deployed codeagent path"
    )


def test_coverage_module_has_no_sentinel() -> None:
    src = COVERAGE_PATH.read_text(encoding="utf-8")
    assert "coverage_not_computed" not in src, (
        "agent/coverage.py missing_source_coverage_gate_result sentinel re-introduced; "
        "post-S64 codeagent path doesn't run coverage gates so [] is the truth"
    )


# ---------------------------------------------------------------------------
# `quality_gates_passed` / `loops_executed` / `sources_count` stripped from
# ExecutionMetadata.
# ---------------------------------------------------------------------------


def test_execution_metadata_stripped_of_lying_fields() -> None:
    from deepsearch.worker.writer import ExecutionMetadata
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(ExecutionMetadata)}
    forbidden = {"loops_executed", "sources_count", "quality_gates_passed"}
    leaked = field_names & forbidden
    assert not leaked, (
        f"ExecutionMetadata re-introduced lying fields: {sorted(leaked)}. "
        "T42.3 confirmed zero TraceLab readers; truth surface is "
        "protocol.report_metadata"
    )


def test_execution_metadata_to_dict_emits_no_lying_fields() -> None:
    from deepsearch.worker.writer import ExecutionMetadata

    em = ExecutionMetadata(total_tokens=0, duration_seconds=0.0)
    payload = em.to_dict()
    forbidden = {"loops_executed", "sources_count", "quality_gates_passed"}
    leaked = set(payload.keys()) & forbidden
    assert not leaked, (
        f"ExecutionMetadata.to_dict() re-emits lying fields: {sorted(leaked)}"
    )


def test_writer_does_not_import_check_quality_gates() -> None:
    tree = _module_ast(WRITER_PATH)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "check_quality_gates":
                    raise AssertionError(
                        "worker/writer.py re-introduced check_quality_gates "
                        "import; the S75 strip removed the fallback gate "
                        "evaluation that produced the lying "
                        "quality_gates_passed=False on the codeagent path"
                    )


def test_worker_main_does_not_access_stripped_metadata_fields() -> None:
    """Closes the PD44(a) audit gap surfaced by CMOS-CONTEXT-RETRIEVAL-01
    (UUID d1f23330-b2b3-4191-a453-11436ed78411, blocked 2026-04-29). The S75
    m01 frozen audit caught the dataclass-side strip but missed the runtime
    attribute-access surface in worker/main.py — three log statements still
    read ``metadata.loops_executed`` and ``metadata.sources_count``, which
    raised AttributeError at upload time.
    """
    tree = _module_ast(WORKER_MAIN_PATH)
    forbidden = {"loops_executed", "sources_count", "quality_gates_passed"}
    leaked: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            target = ast.unparse(node.value)
            leaked.append(f"{target}.{node.attr} at line {node.lineno}")
    assert not leaked, (
        "worker/main.py accesses stripped ExecutionMetadata fields: "
        f"{leaked}. These attributes were removed at S75 m01; runtime "
        "access raises AttributeError on the deployed codeagent path."
    )


# ---------------------------------------------------------------------------
# `max_count=5` cap removed from synthesis.py `_select_key_insights`.
# ---------------------------------------------------------------------------


def test_synthesis_select_key_insights_has_no_max_count() -> None:
    tree = _module_ast(SYNTHESIS_PATH)
    fn_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_select_key_insights":
            fn_def = node
            break
    assert fn_def is not None, "_select_key_insights function missing from synthesis.py"
    arg_names = {a.arg for a in fn_def.args.args}
    assert "max_count" not in arg_names, (
        "_select_key_insights re-introduced max_count parameter; the S75 "
        "5-cap kill removed the legacy hard-cap that truncated CodeAgent "
        "synthesis output"
    )


def test_synthesis_call_site_does_not_pass_max_count() -> None:
    tree = _module_ast(SYNTHESIS_PATH)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_select_key_insights"
        ):
            for kw in node.keywords:
                assert kw.arg != "max_count", (
                    "_select_key_insights call site re-introduced max_count="
                    f"{ast.unparse(kw.value)} kwarg"
                )


# ---------------------------------------------------------------------------
# `research_depth` dropped from poller.py CLAIM_MISSION_SQL (T42.2 lockstep).
# ---------------------------------------------------------------------------


def test_poller_claim_sql_does_not_select_research_depth() -> None:
    src = POLLER_PATH.read_text(encoding="utf-8")
    # Find the CLAIM_MISSION_SQL block and verify research_depth not in SELECT
    start = src.find("CLAIM_MISSION_SQL")
    assert start != -1, "CLAIM_MISSION_SQL constant missing from poller.py"
    # Get the triple-quoted SQL body after the constant
    sql_start = src.find('"""', start)
    sql_end = src.find('"""', sql_start + 3)
    assert sql_start != -1 and sql_end != -1, "CLAIM_MISSION_SQL body not found"
    sql_body = src[sql_start + 3 : sql_end]
    assert "research_depth" not in sql_body, (
        "worker/poller.py CLAIM_MISSION_SQL re-introduced research_depth column "
        "(T42.2 lockstep — TraceLab PR #213 alembic 028 DROPped the column)"
    )


def test_poller_does_not_parse_research_depth() -> None:
    """The :249-254 parse path was stripped at S75 m01."""
    tree = _module_ast(POLLER_PATH)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and any(
                isinstance(a, ast.Constant) and a.value == "research_depth"
                for a in node.args
            )
        ):
            raise AssertionError(
                "worker/poller.py re-introduced record.get('research_depth') "
                "parse-path (T42.2 lockstep violation)"
            )


# ---------------------------------------------------------------------------
# `agent/url_utils.py` carve-out present.
# ---------------------------------------------------------------------------


def test_url_utils_module_exists() -> None:
    assert URL_UTILS_PATH.exists(), (
        "deepsearch/agent/url_utils.py missing — the S75 m01 Phase D Option A "
        "refactor split resolve_source_url + extract_domain into url_utils.py"
    )


def test_url_utils_exports_required_symbols() -> None:
    from deepsearch.agent import url_utils

    assert hasattr(url_utils, "resolve_source_url"), (
        "url_utils.resolve_source_url missing — Phase D refactor incomplete"
    )
    assert hasattr(url_utils, "extract_domain"), (
        "url_utils.extract_domain missing — Phase D refactor incomplete"
    )


def test_url_utils_consumers_use_url_utils_not_source_quality() -> None:
    """The 3 KEEP-side import sites must import from url_utils, not source_quality."""
    sites = (
        REPO_ROOT / "deepsearch" / "report" / "templates.py",
        REPO_ROOT / "deepsearch" / "report" / "evidence.py",
        REPO_ROOT / "deepsearch" / "agent" / "slot_coverage.py",
    )
    for site in sites:
        src = site.read_text(encoding="utf-8")
        assert "from deepsearch.agent.source_quality import resolve_source_url" not in src, (
            f"{site.name} still imports resolve_source_url from source_quality "
            "(should be url_utils)"
        )
        assert "from .source_quality import extract_domain" not in src, (
            f"{site.name} still imports extract_domain from source_quality "
            "(should be url_utils)"
        )


# ---------------------------------------------------------------------------
# `agent/graph.py` simplified to minimal build_workflow (no should_continue,
# no working_set/concept_clustering/coverage/evaluation imports).
# ---------------------------------------------------------------------------


def test_graph_module_minimal_post_deletion() -> None:
    tree = _module_ast(GRAPH_PATH)
    func_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "should_continue" not in func_names, (
        "agent/graph.py re-introduced should_continue; the S75 m01 deletion "
        "removed all routing-helper logic since the only consumer was the "
        "deleted _graph_builder.py"
    )


def test_graph_module_does_not_import_helpers() -> None:
    tree = _module_ast(GRAPH_PATH)
    forbidden = {
        "concept_clustering",
        "coverage",
        "evaluation",
        "working_set",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_tail = (node.module or "").rsplit(".", 1)[-1]
            if module_tail in forbidden:
                raise AssertionError(
                    f"agent/graph.py re-introduced import from .{module_tail} "
                    "— the simplified build_workflow only needs codeagent_finalize_node + state"
                )


# ---------------------------------------------------------------------------
# `agent/__init__.py` source_quality re-exports stripped.
# ---------------------------------------------------------------------------


def test_agent_init_no_source_quality_lazy_reexports() -> None:
    from deepsearch import agent

    forbidden = {
        "SourceQualityResult",
        "estimate_source_quality",
        "filter_sources_by_quality",
        "get_unique_domains",
    }
    leaked = forbidden & set(agent.__all__)
    assert not leaked, (
        f"deepsearch.agent.__all__ re-introduced source_quality re-exports: "
        f"{sorted(leaked)}. Phase D scope strips them — direct imports from "
        f"deepsearch.agent.source_quality still work via test files."
    )
