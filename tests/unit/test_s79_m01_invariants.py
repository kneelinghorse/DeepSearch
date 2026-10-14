"""Static-AST regression guards for the S79 m01 retrieval-substrate widening.

15th-consecutive-sprint RD2 layered-guard contribution. Pins the load-bearing
invariants from PD92–PD96:

* ``deepsearch/agent/tools/scholar_search.py`` exists and defines a
  ``scholar_search`` callable (m01.a).
* ``codeagent_runner.build_codeagent_tools`` imports ``scholar_search`` from
  ``.tools.scholar_search`` and wires it into the returned tools list (m01.a).
* ``_HIGH_HOST_PATTERNS`` in ``codeagent_runner.py`` contains the 25 academic
  and standards-body hosts mandated by PD94 (m01.b).
* ``_HIGH_HOST_TLD_SUFFIXES`` is defined and contains ``".edu"`` (m01.b).
* ``DEFAULT_MAX_STEPS == 40`` both at AST constant-assignment level and at
  runtime (m01.d / PD96).
* ``citation_extraction.py`` defines ``_UNICODE_SUPER_MARKER_RE`` and
  ``_UNICODE_SUPER_REFERENCE_ENTRY_RE`` (m01.c / PD95).

These guards fail loudly if a future sprint:

* deletes the scholar_search wrapper or its wiring in build_codeagent_tools,
* removes any of the academic/standards host patterns added at S79,
* removes the ``.edu`` TLD suffix entry,
* downgrades ``DEFAULT_MAX_STEPS`` without replacing the ``m01.d`` rationale,
* removes the Unicode superscript regex extension.

Tests do not call any LLM, do not hit the network, and do not exercise
runtime code paths — pure AST inspection over ``deepsearch/`` files plus a
final runtime constant check.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPSEARCH_ROOT = REPO_ROOT / "deepsearch"
SCHOLAR_SEARCH_PATH = DEEPSEARCH_ROOT / "agent" / "tools" / "scholar_search.py"
CODEAGENT_RUNNER_PATH = DEEPSEARCH_ROOT / "agent" / "codeagent_runner.py"
CITATION_EXTRACTION_PATH = DEEPSEARCH_ROOT / "agent" / "citation_extraction.py"


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# m01.a — scholar_search tool exists and exposes the callable.
# ---------------------------------------------------------------------------


def test_scholar_search_module_exists() -> None:
    assert SCHOLAR_SEARCH_PATH.exists(), (
        "deepsearch/agent/tools/scholar_search.py must exist (S79 m01.a NEW tool)"
    )


def test_scholar_search_defines_callable() -> None:
    tree = _module_ast(SCHOLAR_SEARCH_PATH)
    callables = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "scholar_search" in callables, (
        "scholar_search.py must define a `scholar_search` function (m01.a contract)"
    )


def test_scholar_search_imports_both_backends() -> None:
    """Primary OpenAlex + fallback Semantic Scholar — both are required wiring."""
    tree = _module_ast(SCHOLAR_SEARCH_PATH)
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.name)
    for required in ("OpenAlexClient", "OpenAlexError",
                     "SemanticScholarClient", "SemanticScholarError"):
        assert required in imported_names, (
            f"scholar_search.py must import {required} — wiring contract for m01.a"
        )


def test_codeagent_runner_imports_scholar_search() -> None:
    """``build_codeagent_tools`` must import scholar_search from the tools package."""
    source = CODEAGENT_RUNNER_PATH.read_text(encoding="utf-8")
    assert "from .tools.scholar_search import scholar_search" in source, (
        "codeagent_runner must import scholar_search from .tools.scholar_search "
        "(m01.a wiring contract)"
    )


def test_codeagent_runner_wires_scholar_search_into_tools_list() -> None:
    """The returned ``CodeAgentToolBundle.tools`` list must contain ``scholar_search``."""
    tree = _module_ast(CODEAGENT_RUNNER_PATH)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_codeagent_tools":
            # Look for `scholar_search` referenced inside a Return statement
            # that constructs the CodeAgentToolBundle. AST scan: any Return
            # node whose value is a Call to CodeAgentToolBundle with a tools
            # kwarg that references scholar_search.
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Call):
                    for kw in sub.value.keywords:
                        if kw.arg == "tools" and isinstance(kw.value, ast.List):
                            element_names = [
                                elt.id
                                for elt in kw.value.elts
                                if isinstance(elt, ast.Name)
                            ]
                            assert "scholar_search" in element_names, (
                                "build_codeagent_tools must include scholar_search "
                                "in its returned tools list (m01.a wiring)"
                            )
                            return
    raise AssertionError(
        "build_codeagent_tools function or its tools-list Return not found"
    )


# ---------------------------------------------------------------------------
# m01.b — authority-tier patches.
# ---------------------------------------------------------------------------


REQUIRED_ACADEMIC_HOSTS = (
    "arxiv.org",
    "acm.org",
    "dl.acm.org",
    "ieee.org",
    "ieeexplore.ieee.org",
    "springer.com",
    "link.springer.com",
    "nature.com",
    "sciencedirect.com",
    "openalex.org",
    "api.openalex.org",
    "semanticscholar.org",
    "api.semanticscholar.org",
    "openreview.net",
    "plos.org",
    "biorxiv.org",
)

REQUIRED_STANDARDS_HOSTS = (
    "ietf.org",
    "tools.ietf.org",
    "datatracker.ietf.org",
    "rfc-editor.org",
    "w3.org",
    "oauth.net",
    "openid.net",
    "nist.gov",
    "iso.org",
)


@pytest.mark.parametrize("host", REQUIRED_ACADEMIC_HOSTS + REQUIRED_STANDARDS_HOSTS)
def test_high_host_patterns_contains_academic_and_standards(host: str) -> None:
    """PD94 KEEP guard — the 25 academic/standards hosts stay on the allowlist."""
    from deepsearch.agent.codeagent_runner import _HIGH_HOST_PATTERNS

    pattern_hosts = {h for h, _p in _HIGH_HOST_PATTERNS}
    assert host in pattern_hosts, (
        f"_HIGH_HOST_PATTERNS must contain {host!r} (S79 m01.b PD94 KEEP guard)"
    )


def test_high_host_tld_suffixes_contains_edu() -> None:
    """PD94 — the generic .edu TLD shape is the second high-tier vector."""
    from deepsearch.agent.codeagent_runner import _HIGH_HOST_TLD_SUFFIXES

    assert ".edu" in _HIGH_HOST_TLD_SUFFIXES, (
        "_HIGH_HOST_TLD_SUFFIXES must contain '.edu' (m01.b TLD shape)"
    )


# ---------------------------------------------------------------------------
# m01.d — DEFAULT_MAX_STEPS bumped 25 → 40.
# ---------------------------------------------------------------------------


def test_default_max_steps_constant_is_forty_at_runtime() -> None:
    from deepsearch.agent import codeagent_runner

    assert codeagent_runner.DEFAULT_MAX_STEPS == 40, (
        "DEFAULT_MAX_STEPS must equal 40 (S79 m01.d / PD96 — Type A failure-mode "
        "headroom; revert is one-line per the rationale comment in the source)"
    )


def test_default_max_steps_assignment_is_forty_in_source() -> None:
    """Belt + suspenders: the AST constant assignment also reads 40."""
    tree = _module_ast(CODEAGENT_RUNNER_PATH)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEFAULT_MAX_STEPS":
                    assert isinstance(node.value, ast.Constant)
                    assert node.value.value == 40
                    return
    raise AssertionError(
        "DEFAULT_MAX_STEPS top-level assignment not found in codeagent_runner.py"
    )


# ---------------------------------------------------------------------------
# m01.c — Unicode superscript regex extension.
# ---------------------------------------------------------------------------


def test_citation_extraction_defines_unicode_superscript_regexes() -> None:
    """PD95 — both refs-entry and body-marker regexes for Unicode superscripts
    must be present in citation_extraction.py."""
    source = CITATION_EXTRACTION_PATH.read_text(encoding="utf-8")
    for symbol in (
        "_UNICODE_SUPER_REFERENCE_ENTRY_RE",
        "_UNICODE_SUPER_MARKER_RE",
        "_UNICODE_SUPER_TO_DIGIT",
    ):
        assert symbol in source, (
            f"{symbol} must be defined in citation_extraction.py (S79 m01.c PD95)"
        )


def test_unicode_super_marker_re_compiles_and_matches() -> None:
    """End-to-end runtime check that the regex is well-formed and covers ⁰–⁹."""
    from deepsearch.agent.citation_extraction import _UNICODE_SUPER_MARKER_RE

    # The full superscript range must match.
    for digit in "⁰¹²³⁴⁵⁶⁷⁸⁹":
        # Preceded by a non-letter so the lookbehind doesn't reject it.
        sample = f"text {digit}"
        assert _UNICODE_SUPER_MARKER_RE.search(sample) is not None, (
            f"_UNICODE_SUPER_MARKER_RE must match the superscript {digit!r}"
        )

    # Uppercase-letter prefix (math-variable shape) must NOT match.
    assert _UNICODE_SUPER_MARKER_RE.search("T¹⁰") is None, (
        "Math-variable exponent T¹⁰ must NOT match (PD95 negative case)"
    )
