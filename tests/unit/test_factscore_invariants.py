"""Static-AST regression guards for the S69 FActScore harness.

Mirrors the layered-guard pattern from S65 (synthesis-path bypass), S66
(critique single-call), S67 (citation extractor), S68 (citation
liveness), and S68.5 (writer persistence). This module locks the
*structure* of ``deepsearch.eval.factscore``: signature, dataclass shape,
and the PD10 local-only contract.

If a future sprint accidentally:

* makes ``score_mission`` synchronous (breaks PD5 budget enforcement),
* drops a PD7 field from ``FActScoreReport``,
* drops or renames the load-bearing ``unsupported_count_per_citation``
  reference in ``to_dict`` (breaks the S68-axis vs FActScore-axis parity
  surface), or
* imports any production-runtime module into the harness (breaks PD10
  local-only — runtime FActScore wiring is post-S70 territory at the
  earliest),

these guards fail loudly. Tests do not call any LLM, do not hit any
network, and do not require ``deepeval`` to be installed.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FACTSCORE_PATH = REPO_ROOT / "deepsearch" / "eval" / "factscore.py"
FACTSCORE_CLI_PATH = REPO_ROOT / "deepsearch" / "eval" / "__main__.py"
PROTOCOL_EXTRACT_PATH = REPO_ROOT / "deepsearch" / "eval" / "protocol_extract.py"
DEEPEVAL_ADAPTER_PATH = REPO_ROOT / "deepsearch" / "eval" / "deepeval_adapter.py"


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _find_async_def(tree: ast.AST, name: str) -> ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return node
    return None


def _find_class_def(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _import_targets(tree: ast.AST) -> set[str]:
    """Return all dotted import names (``import X.Y`` and ``from X.Y import``)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


# ---------------------------------------------------------------------------
# 1. ``score_mission`` is async — PD5 budget enforcement runs through asyncio.
# ---------------------------------------------------------------------------


def test_score_mission_is_async():
    tree = _module_ast(FACTSCORE_PATH)
    fn = _find_async_def(tree, "score_mission")
    assert fn is not None, (
        "score_mission must be an `async def`; PD5 budget enforcement uses "
        "asyncio.wait + asyncio.to_thread."
    )


# ---------------------------------------------------------------------------
# 2. Signature carries the PD-mandated keyword-only and positional args.
# ---------------------------------------------------------------------------


def test_score_mission_signature_carries_locked_kwargs():
    tree = _module_ast(FACTSCORE_PATH)
    fn = _find_async_def(tree, "score_mission")
    assert fn is not None

    kwonly_names = {a.arg for a in fn.args.kwonlyargs}
    pos_names = {a.arg for a in fn.args.args}

    required_kwonly = {
        "budget_seconds",
        "decomposition_model",
        "verification_model",
        "threshold",
    }
    missing_kwonly = required_kwonly - kwonly_names
    assert not missing_kwonly, (
        f"score_mission keyword-only args missing: {sorted(missing_kwonly)}. "
        f"PD2 + PD3 + PD5 lock these as configurable per call."
    )

    required_positional = {"mission_id", "protocol_path", "sources_path"}
    missing_positional = required_positional - pos_names
    assert not missing_positional, (
        f"score_mission positional args missing: {sorted(missing_positional)}."
    )


# ---------------------------------------------------------------------------
# 3. ``FActScoreReport`` declares every PD7 field as an annotated assignment.
# ---------------------------------------------------------------------------


def test_factscore_report_carries_all_pd7_fields():
    tree = _module_ast(FACTSCORE_PATH)
    cls = _find_class_def(tree, "FActScoreReport")
    assert cls is not None, "FActScoreReport dataclass must exist"

    field_names: set[str] = set()
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            field_names.add(node.target.id)

    pd7_fields = {
        "mission_id",
        "factscore",
        "decomposed_claim_count",
        "supported_count",
        "unsupported_count",
        "unknown_count",
        "unsupported_count_per_citation",
        "wall_clock_seconds",
        "budget_exhausted",
        "decomposition_model_id",
        "verification_model_id",
        "threshold",
        "harness_version",
    }
    missing = pd7_fields - field_names
    assert not missing, (
        f"FActScoreReport missing PD7 fields: {sorted(missing)}. The locked "
        f"plan-doc PD7 enumerates these fields explicitly; any deletion is a "
        f"schema regression that would silently drop signal needed for S70 "
        f"calibration."
    )


# ---------------------------------------------------------------------------
# 4. PD10 — no production-runtime imports leak in. The harness consumes
#    cached outputs; runtime FActScore wiring is S70 territory at earliest.
# ---------------------------------------------------------------------------


def test_factscore_module_does_not_import_production_runtime():
    tree = _module_ast(FACTSCORE_PATH)
    targets = _import_targets(tree)

    forbidden = {
        "deepsearch.agent.codeagent_finalize_node",
        "deepsearch.agent.citation_liveness",
        "deepsearch.agent.critique",
        "deepsearch.agent.contract_to_prompt",
        "deepsearch.agent.codeagent_runner",
        "deepsearch.worker.writer",
        "deepsearch.agent.tools.web_search",
        "deepsearch.agent.tools.source_fetch",
    }
    leaked = targets & forbidden
    assert not leaked, (
        f"deepsearch.eval.factscore must stay LOCAL-ONLY (PD10). Forbidden "
        f"imports leaked: {sorted(leaked)}. The harness reads cached "
        f"artifacts; any runtime wiring (codeagent_finalize_node, critique, "
        f"writer, etc.) is S70 territory at earliest."
    )


# ---------------------------------------------------------------------------
# 5. ``to_dict`` references the load-bearing PD7 field by attribute name.
#    Without this, a refactor could silently drop the FActScore-axis
#    counterpart to S68's ``hallucinated_url_count``.
# ---------------------------------------------------------------------------


def test_to_dict_references_unsupported_count_per_citation():
    tree = _module_ast(FACTSCORE_PATH)
    cls = _find_class_def(tree, "FActScoreReport")
    assert cls is not None

    to_dict_fn = None
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "to_dict":
            to_dict_fn = node
            break
    assert to_dict_fn is not None, "FActScoreReport.to_dict must exist"

    referenced = False
    for node in ast.walk(to_dict_fn):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "unsupported_count_per_citation"
        ):
            referenced = True
            break

    assert referenced, (
        "FActScoreReport.to_dict must reference "
        "self.unsupported_count_per_citation. PD7 — load-bearing FActScore-"
        "axis equivalent of S68's hallucinated_url_count; this guard catches "
        "an accidental drop in any future schema-touching sprint."
    )


# ---------------------------------------------------------------------------
# 6. (m02) The CLI module stays LOCAL-ONLY — same forbidden-imports set as
#    the harness module. The CLI is the only other surface allowed in
#    deepsearch/eval/, and any future tendency to wire production-runtime
#    helpers into it (codeagent_finalize_node, critique, writer, etc.)
#    must fail this guard before it lands.
# ---------------------------------------------------------------------------


def test_factscore_cli_module_does_not_import_production_runtime():
    tree = _module_ast(FACTSCORE_CLI_PATH)
    targets = _import_targets(tree)

    forbidden = {
        "deepsearch.agent.codeagent_finalize_node",
        "deepsearch.agent.citation_liveness",
        "deepsearch.agent.critique",
        "deepsearch.agent.contract_to_prompt",
        "deepsearch.agent.codeagent_runner",
        "deepsearch.worker.writer",
        "deepsearch.agent.tools.web_search",
        "deepsearch.agent.tools.source_fetch",
    }
    leaked = targets & forbidden
    assert not leaked, (
        f"deepsearch.eval.__main__ must stay LOCAL-ONLY (PD10). Forbidden "
        f"imports leaked: {sorted(leaked)}. The CLI is allowed to import "
        f"only deepsearch.eval.factscore + stdlib."
    )

    # The CLI's only deepsearch.* import targets are the harness modules
    # (FActScore + S85 m02 benchmark) plus the S70 m01 read-only
    # protocol_extract utility — a tighter constraint than the forbidden-set
    # guard. All are LOCAL-ONLY eval modules (PD10-clean themselves).
    deepsearch_imports = {t for t in targets if t.startswith("deepsearch")}
    allowed = {
        "deepsearch.eval",
        "deepsearch.eval.benchmark",
        "deepsearch.eval.comparison",
        "deepsearch.eval.factscore",
        "deepsearch.eval.protocol_extract",
    }
    assert deepsearch_imports.issubset(allowed), (
        f"deepsearch.eval.__main__ must only import from deepsearch.eval.*; "
        f"observed {sorted(deepsearch_imports)}, allowed {sorted(allowed)}."
    )


# ---------------------------------------------------------------------------
# 7. (S70 m01) The protocol_extract module stays LOCAL-ONLY. PD10 forbidden-
#    imports set carries verbatim from S69. The shared async asyncpg wrapper
#    in deepsearch.worker.database is the ONLY production-runtime-adjacent
#    import allowed — it is a bare DB client, not mission-runtime processing
#    logic.
# ---------------------------------------------------------------------------


def test_protocol_extract_module_does_not_import_production_runtime():
    tree = _module_ast(PROTOCOL_EXTRACT_PATH)
    targets = _import_targets(tree)

    forbidden = {
        "deepsearch.agent.codeagent_finalize_node",
        "deepsearch.agent.citation_liveness",
        "deepsearch.agent.critique",
        "deepsearch.agent.contract_to_prompt",
        "deepsearch.agent.codeagent_runner",
        "deepsearch.worker.writer",
        "deepsearch.agent.tools.web_search",
        "deepsearch.agent.tools.source_fetch",
    }
    leaked = targets & forbidden
    assert not leaked, (
        f"deepsearch.eval.protocol_extract must stay LOCAL-ONLY (PD10). "
        f"Forbidden imports leaked: {sorted(leaked)}. The S70 m01 extractor "
        f"is a read-only Postgres SELECT utility; any runtime wiring is "
        f"explicitly out-of-scope per the locked plan-doc."
    )

    # Only the bare asyncpg wrapper and the S97 standard-library path helper
    # are allowed. The helper's own imports stay guarded so it cannot become
    # an indirect route into paid production machinery.
    deepsearch_imports = {t for t in targets if t.startswith("deepsearch")}
    allowed = {"deepsearch.worker.database", "deepsearch.data_paths"}
    assert deepsearch_imports.issubset(allowed), (
        f"Extractor imports must stay within {sorted(allowed)}; "
        f"observed {sorted(deepsearch_imports)}."
    )
    helper = PROTOCOL_EXTRACT_PATH.parents[1] / "data_paths.py"
    assert _import_targets(_module_ast(helper)).issubset({"__future__", "os", "pathlib"})


# ---------------------------------------------------------------------------
# 8. (S70 m01) extract_protocol is async — it runs through the asyncpg pool
#    in deepsearch.worker.database which is async-only.
# ---------------------------------------------------------------------------


def test_extract_protocol_is_async():
    tree = _module_ast(PROTOCOL_EXTRACT_PATH)
    fn = _find_async_def(tree, "extract_protocol")
    assert fn is not None, (
        "extract_protocol must be an `async def` — the underlying asyncpg "
        "DatabaseClient is async-only."
    )


# ---------------------------------------------------------------------------
# 9. (S70 m01) extract_protocol signature carries the locked kwarg-only API
#    (conn_str + out_path), so a future refactor cannot silently rename or
#    drop them and break the m02 calibration runner that piggybacks on this
#    surface.
# ---------------------------------------------------------------------------


def test_extract_protocol_signature_carries_locked_kwargs():
    tree = _module_ast(PROTOCOL_EXTRACT_PATH)
    fn = _find_async_def(tree, "extract_protocol")
    assert fn is not None

    pos_names = {a.arg for a in fn.args.args}
    kwonly_names = {a.arg for a in fn.args.kwonlyargs}

    assert "mission_id" in pos_names, (
        "extract_protocol must keep `mission_id` as a positional arg."
    )
    required_kwonly = {"conn_str", "out_path"}
    missing = required_kwonly - kwonly_names
    assert not missing, (
        f"extract_protocol keyword-only args missing: {sorted(missing)}. "
        f"PD11 (S70) mission objective lists conn_str + out_path explicitly."
    )


# ---------------------------------------------------------------------------
# 10. (S71 m02) The DeepEvalBaseLLM adapter module stays LOCAL-ONLY. PD10
#     forbidden-imports set extends here verbatim. The adapter wraps
#     third-party SDKs (``openai``, ``google.genai``) directly; it must
#     not couple back to any production-runtime surface.
# ---------------------------------------------------------------------------


def test_deepeval_adapter_module_does_not_import_production_runtime():
    tree = _module_ast(DEEPEVAL_ADAPTER_PATH)
    targets = _import_targets(tree)

    forbidden = {
        "deepsearch.agent.codeagent_finalize_node",
        "deepsearch.agent.citation_liveness",
        "deepsearch.agent.critique",
        "deepsearch.agent.contract_to_prompt",
        "deepsearch.agent.codeagent_runner",
        "deepsearch.worker.writer",
        "deepsearch.agent.tools.web_search",
        "deepsearch.agent.tools.source_fetch",
    }
    leaked = targets & forbidden
    assert not leaked, (
        f"deepsearch.eval.deepeval_adapter must stay LOCAL-ONLY (PD10). "
        f"Forbidden imports leaked: {sorted(leaked)}. The adapter wraps "
        f"third-party SDKs directly and is consumed only by "
        f"deepsearch.eval.factscore — any runtime wiring couples eval/ to "
        f"the agent surface and breaks the local-only contract that lets "
        f"FActScore stay advisory (PD14)."
    )

    # Tighter constraint: the adapter has zero deepsearch.* imports.
    # It is the bottom-of-stack eval-side module.
    deepsearch_imports = {t for t in targets if t.startswith("deepsearch")}
    assert deepsearch_imports == set(), (
        f"deepsearch.eval.deepeval_adapter must have ZERO deepsearch.* "
        f"imports — it is consumed by factscore.py, not the other way "
        f"around. Observed: {sorted(deepsearch_imports)}."
    )


# ---------------------------------------------------------------------------
# 11. (S71 m02) The adapter module exposes a `resolve_deepeval_llm` factory
#     that the harness wires at the FaithfulnessMetric construction site.
#     Renaming the factory silently breaks the calibration path; this guard
#     locks the public surface.
# ---------------------------------------------------------------------------


def test_deepeval_adapter_module_exposes_resolve_factory():
    tree = _module_ast(DEEPEVAL_ADAPTER_PATH)

    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_deepeval_llm":
            found = True
            break
    assert found, (
        "deepsearch.eval.deepeval_adapter must define `resolve_deepeval_llm`. "
        "It is the single entry-point consumed by deepsearch.eval.factscore "
        "and the public surface tests + S73 paid-smoke wiring depend on."
    )


# ---------------------------------------------------------------------------
# 12. (S71 m02) factscore.py wires the adapter factory at the
#     FaithfulnessMetric construction site through the local indirection
#     `_resolve_verification_llm`. Catching this name change here lets the
#     test_deepeval_adapter monkeypatch surface stay stable across refactors.
# ---------------------------------------------------------------------------


def test_factscore_module_defines_resolve_verification_llm():
    tree = _module_ast(FACTSCORE_PATH)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_verification_llm":
            found = True
            break
    assert found, (
        "deepsearch.eval.factscore must expose `_resolve_verification_llm` "
        "as the single indirection between the harness and the S71 Path B "
        "adapter factory. The unit tests for the adapter monkeypatch this "
        "symbol; renaming it silently breaks the test seam."
    )
