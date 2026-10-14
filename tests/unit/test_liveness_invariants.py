"""Static-AST regression guards for the S68 citation liveness gate.

Mirrors the layered-guard pattern from S65 (synthesis-path bypass), S66
(critique single-call invariants), and S67 (citation extractor invariants):
runtime tests live in the consumer test files; this module locks the
*structure* — what imports what, where the gate is invoked, and that the
gate stays HTTP-only with no LLM in the loop.

If a future sprint accidentally:

* adds a retrieval tool import to :mod:`citation_liveness` (re-retrieval
  triggered by extraction is roadmap §3.7),
* swaps the gate for an LLM-driven check (changes critique's no-retrieval
  contract from S66),
* moves the gate's callsite outside the seam between the title resolver
  and the critique LLM call, or
* introduces module-level cache state (cross-mission state leak),

these guards fail loudly. Tests do not call any LLM, do not hit any
network, and do not require pytest fixtures.

Test layout follows the S66.1 convention (decision #501): flat in
``tests/unit/``, not ``tests/unit/agent/``.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CITATION_LIVENESS_PATH = REPO_ROOT / "deepsearch" / "agent" / "citation_liveness.py"
FINALIZE_NODE_PATH = REPO_ROOT / "deepsearch" / "agent" / "codeagent_finalize_node.py"


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


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


def _imported_symbols(tree: ast.AST) -> set[tuple[str, str]]:
    """Return ``(module, symbol)`` pairs for every ``from … import …`` form."""

    pairs: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                pairs.add((node.module, alias.name))
    return pairs


def _find_call_lineno(tree: ast.AST, func_name: str) -> int | None:
    """Return the line number of the FIRST call to ``func_name`` (any callable form)."""

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            name = (
                target.id
                if isinstance(target, ast.Name)
                else target.attr
                if isinstance(target, ast.Attribute)
                else None
            )
            if name == func_name:
                return node.lineno
            # Offloaded form: ``asyncio.to_thread(func_name, ...)`` still invokes
            # func_name (just off the event-loop thread). Recognize it so the
            # seam-ordering invariant holds across the S87-m07 critique offload.
            if name == "to_thread" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Name) and first.id == func_name:
                    return node.lineno
    return None


# ---------------------------------------------------------------------------
# 1. Tool-surface invariant — the gate imports only `url_liveness` from the
#    `deepsearch.agent.tools.*` package. No web_search, source_fetch,
#    scratchpad, or rerank — those would imply re-retrieval inside the gate
#    (roadmap §3.7 anti-pattern).
# ---------------------------------------------------------------------------


def test_citation_liveness_imports_only_url_liveness_from_tools():
    tree = _module_ast(CITATION_LIVENESS_PATH)
    pairs = _imported_symbols(tree)
    tool_imports = {
        (mod, sym) for (mod, sym) in pairs if mod.startswith("deepsearch.agent.tools")
    }
    assert tool_imports == {("deepsearch.agent.tools.url_liveness", "url_liveness")}, (
        f"citation_liveness must import only url_liveness from "
        f"deepsearch.agent.tools.*; observed {sorted(tool_imports)}. "
        f"Importing web_search / source_fetch / scratchpad / rerank into "
        f"the gate would imply re-retrieval triggered by extraction, which "
        f"is the roadmap §3.7 anti-pattern."
    )


# ---------------------------------------------------------------------------
# 2. No-LLM invariant — the gate is HTTP HEAD only, never a model call.
# ---------------------------------------------------------------------------


def test_citation_liveness_has_no_llm_imports():
    tree = _module_ast(CITATION_LIVENESS_PATH)
    targets = _import_targets(tree)
    forbidden = {
        "smolagents",
        "openai",
        "anthropic",
        "deepsearch.agent.codeagent_runner",
        "deepsearch.agent.critique",
    }
    bad = targets & forbidden
    assert not bad, (
        f"citation_liveness must stay LLM-free; observed forbidden imports: "
        f"{sorted(bad)}. The gate is an HTTP HEAD probe + URL-set membership "
        f"check; introducing an LLM here would change critique's "
        f"no-retrieval contract from S66."
    )


# ---------------------------------------------------------------------------
# 3. Seam-ordering invariant — `check_citations_liveness` is invoked AFTER
#    `_resolve_citation_titles` (or the `_extract_and_resolve_citations`
#    helper that wraps it) and BEFORE `run_critique`. The seam moved at S68
#    so extraction precedes critique; this guard locks the new ordering.
# ---------------------------------------------------------------------------


def test_finalize_node_calls_check_liveness_between_extract_and_critique():
    tree = _module_ast(FINALIZE_NODE_PATH)
    extract_line = _find_call_lineno(tree, "_extract_and_resolve_citations")
    liveness_line = _find_call_lineno(tree, "_run_liveness_check")
    critique_line = _find_call_lineno(tree, "_invoke_critique")

    assert extract_line is not None, (
        "codeagent_finalize_node must call _extract_and_resolve_citations"
    )
    assert liveness_line is not None, (
        "codeagent_finalize_node must call _run_liveness_check (S68 wiring)"
    )
    assert critique_line is not None, (
        "codeagent_finalize_node must call _invoke_critique"
    )
    assert extract_line < liveness_line < critique_line, (
        f"S68 seam ordering violated: extract={extract_line}, "
        f"liveness={liveness_line}, critique={critique_line}. "
        f"The PD2-locked order is "
        f"extract → resolve titles → liveness check → critique."
    )


def test_finalize_node_calls_check_citations_liveness_exactly_once():
    """Static call-count guard mirrors the S66 single-critique invariant.

    The gate must fan out concurrent probes inside ONE batch wrapper call,
    not be invoked iteratively from the finalize node. A second callsite
    would imply we're re-running liveness on the same annotations.
    """

    tree = _module_ast(FINALIZE_NODE_PATH)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            name = (
                target.id
                if isinstance(target, ast.Name)
                else target.attr
                if isinstance(target, ast.Attribute)
                else None
            )
            if name == "check_citations_liveness":
                count += 1
    assert count == 1, (
        f"Expected exactly one static call to `check_citations_liveness` "
        f"in codeagent_finalize_node.py, found {count}."
    )


# ---------------------------------------------------------------------------
# 4. No-module-level-cache invariant — cache MUST live on a per-call
#    instance. A module-level dict would leak liveness state across missions
#    and silently break PD4's per-mission cache contract.
# ---------------------------------------------------------------------------


def test_citation_liveness_module_has_no_module_level_cache():
    tree = _module_ast(CITATION_LIVENESS_PATH)
    forbidden_names = {"_CACHE", "_LIVENESS_CACHE", "_GLOBAL_CACHE", "_URL_CACHE", "CACHE"}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in forbidden_names:
                    raise AssertionError(
                        f"citation_liveness has a module-level cache "
                        f"binding `{target.id}`. Cache must live on a "
                        f"per-call dict so liveness state never leaks "
                        f"across missions (PD4)."
                    )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in forbidden_names:
                raise AssertionError(
                    f"citation_liveness has a module-level cache annotation "
                    f"`{node.target.id}`. Cache must live on a per-call "
                    f"dict (PD4)."
                )
