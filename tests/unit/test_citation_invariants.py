"""Static-AST + import-surface regression guards for the S67 citation seam.

Mirrors the layered-guard pattern from S65 (synthesis-path bypass) and S66
(critique single-call invariants): runtime tests live in the consumer test
files; this module locks the *structure* — what imports what, where the
extractor is invoked, and that the extractor module stays I/O-free.

If a future sprint accidentally:

* adds a second extractor callsite in :mod:`codeagent_finalize_node`,
* drops the import from :mod:`citation_extraction`, or
* introduces re-retrieval inside the extractor (roadmap §3.7 anti-pattern),

these guards fail loudly. Tests do not call any LLM, do not hit any network,
and do not require pytest fixtures.

Test layout follows the S66.1 convention (decision #501): flat in tests/unit/,
not tests/unit/agent/.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from deepsearch.agent import citation_extraction, codeagent_finalize_node

REPO_ROOT = Path(__file__).resolve().parents[2]
FINALIZE_NODE_PATH = REPO_ROOT / "deepsearch" / "agent" / "codeagent_finalize_node.py"
CITATION_EXTRACTION_PATH = REPO_ROOT / "deepsearch" / "agent" / "citation_extraction.py"


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _count_calls(tree: ast.AST, func_name: str) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name) and target.id == func_name:
                count += 1
            elif isinstance(target, ast.Attribute) and target.attr == func_name:
                count += 1
    return count


# ---------------------------------------------------------------------------
# 1. Static import surface — `extract_citations` is imported into the
#    finalize node from `.citation_extraction`.
# ---------------------------------------------------------------------------


def test_finalize_node_imports_extract_citations():
    tree = _module_ast(FINALIZE_NODE_PATH)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[-1]
            names = {alias.name for alias in node.names}
            if module == "citation_extraction" and "extract_citations" in names:
                found = True
                break
    assert found, (
        "codeagent_finalize_node must import `extract_citations` from "
        "`.citation_extraction`. If this fails, the m02 wiring has been "
        "regressed."
    )


def test_finalize_node_imports_citation_annotation_type():
    tree = _module_ast(FINALIZE_NODE_PATH)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = {alias.name for alias in node.names}
            if module.endswith("citation") and "CitationAnnotation" in names:
                found = True
                break
    assert found, (
        "codeagent_finalize_node must import `CitationAnnotation` so the "
        "type annotation on `_extract_and_resolve_citations` stays accurate."
    )


# ---------------------------------------------------------------------------
# 2. Canonical extraction has one callsite. S96's advisory pre-finalization
#    word preview reuses the pure extractor to count the same marker-stripped
#    text without publishing annotations, and S101 m02's reader-number
#    renderer reads it to decide where numbers go. Permit only those named
#    helpers; the runtime guard still confirms canonical extraction once per
#    completion.
# ---------------------------------------------------------------------------


def test_extraction_calls_are_only_canonical_helper_and_readonly_consumers():
    tree = _module_ast(FINALIZE_NODE_PATH)
    calls = {
        node.name: _count_calls(node, "extract_citations")
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _count_calls(node, "extract_citations")
    }
    assert calls == {
        "_extract_and_resolve_citations": 1,
        "_preview_report_word_budget": 1,
        "_place_reader_numbers": 1,
    }
    assert _count_calls(tree, "extract_citations") == 3


def test_finalize_node_extract_citations_callsite_is_inside_helper():
    """The single callsite belongs to `_extract_and_resolve_citations`.

    Locating it in the helper (rather than inline in `codeagent_finalize_node`)
    keeps the seam testable in isolation and matches the S66 critique helper
    pattern.
    """

    src = inspect.getsource(codeagent_finalize_node)
    assert "def _extract_and_resolve_citations" in src
    helper_src = inspect.getsource(codeagent_finalize_node._extract_and_resolve_citations)
    assert "extract_citations(" in helper_src


# ---------------------------------------------------------------------------
# 3. The extractor module remains I/O-free.
#    Pure-function guarantee from the plan doc — re-retrieval triggered by
#    extraction is an explicit roadmap §3.7 anti-pattern.
# ---------------------------------------------------------------------------


_FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "requests",
    "httpx",
    "urllib",  # urlopen, urllib.request — both live under this top-level name
    "aiohttp",
    "smolagents",  # the LLM framework
    "openai",
    "anthropic",
}

_FORBIDDEN_MODULE_IMPORTS = {
    "deepsearch.agent.tools.web_search",
    "deepsearch.agent.tools.source_fetch",
    "deepsearch.agent.tools.url_liveness",
    "deepsearch.agent.codeagent_runner",
    "deepsearch.agent.critique",
}


def test_extractor_module_has_no_io_imports():
    tree = _module_ast(CITATION_EXTRACTION_PATH)
    seen_top: set[str] = set()
    seen_full: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                seen_top.add(alias.name.split(".")[0])
                seen_full.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            seen_top.add(mod.split(".")[0])
            seen_full.add(mod)

    bad_top = seen_top & _FORBIDDEN_TOP_LEVEL_IMPORTS
    bad_full = seen_full & _FORBIDDEN_MODULE_IMPORTS
    assert not bad_top, (
        f"citation_extraction must remain I/O-free. Forbidden top-level "
        f"imports detected: {sorted(bad_top)}. The extractor is a pure "
        f"function — re-retrieval triggered by extraction is roadmap §3.7."
    )
    assert not bad_full, (
        f"citation_extraction must not import retrieval / LLM modules. "
        f"Forbidden modules: {sorted(bad_full)}."
    )


def test_extractor_module_signature_returns_tuple():
    """The contract `extract_citations(str) -> tuple[str, list[Annotation]]` is locked."""

    sig = inspect.signature(citation_extraction.extract_citations)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["markdown", "telemetry"]
    assert params[0].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params[1].kind == inspect.Parameter.KEYWORD_ONLY
    assert params[1].default is None
    # S96's optional output counter is not a model/source/I/O dependency.



# ---------------------------------------------------------------------------
# 4. Critique fallthrough — extract_citations must NOT itself call run_critique
#    or run_codeagent_synthesis. Pure function only.
# ---------------------------------------------------------------------------


def test_extractor_does_not_call_critique_or_synthesis():
    tree = _module_ast(CITATION_EXTRACTION_PATH)
    forbidden = {"run_critique", "run_codeagent_synthesis", "build_default_model"}
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
            assert name not in forbidden, (
                f"citation_extraction must not call `{name}`. Pure-function "
                f"contract violated."
            )
