"""Static-AST + import-surface regression guards for the S77 m03 source_quality finale.

13th-consecutive-sprint RD2 layered-guard contribution (m03 half — m01 lives at
``test_s77_dead_code_invariants.py``, m02 at ``test_s77_working_set_invariants.py``).

S77 m03 deletes ``deepsearch.agent.source_quality`` — the pre-rebuild authority +
relevance + diversity scoring substrate. Sole production importer was
``deepsearch.agent.working_set`` (deleted at m02), so post-m02 this file had
zero production consumers and zero test consumers (test_source_quality.py was
deleted at m02 per PD73; test_authority_biasing.py + test_embedding_rate_limited.py
likewise — all forced earlier than the plan §7 schedule by working_set's
disappearance).

These guards fail loudly if a future sprint accidentally:

* re-introduces ``deepsearch/agent/source_quality.py``,
* re-imports any source_quality public symbol from elsewhere
  (``estimate_source_quality`` / ``estimate_source_relevance`` /
  ``estimate_source_coherence`` / ``enforce_source_diversity`` /
  ``extract_domain`` / ``resolve_source_url`` / ``SourceQualityResult`` /
  ``DiversityFilterResult`` / ``EmbeddingRateLimitedError`` /
  ``filter_sources_by_quality`` / ``filter_sources_by_relevance`` /
  ``get_objective_embedding`` / ``get_unique_domains`` /
  ``HIGH_AUTHORITY_DOMAINS`` / ``MEDIUM_AUTHORITY_DOMAINS`` /
  ``LOW_AUTHORITY_DOMAINS`` / ``BLOG_LISTICLE_DOMAINS`` /
  ``HIGH_AUTHORITY_TLDS``).

Note on ``extract_domain``: a same-named helper survives at
``deepsearch/agent/url_utils.py`` (the post-S75 carry-forward home for URL
parsing helpers — see the module-level docstring there). The guard targets
imports from ``deepsearch.agent.source_quality`` specifically, not the
sibling ``url_utils`` re-introduction.

Note on the ``source_quality_floor`` config field: that's a depth-config field
name (a string parameter at ``report/protocol.py`` + ``report/generator.py``),
not a reference to the deleted module. The guards target module-level imports,
not field-name overlap.

S75 ``test_s75_deletion_invariants.py:356,360`` already string-asserts absence
of two specific source_quality import patterns; those guards remain valid and
strengthen with this file.

Tests do not call any LLM, do not hit any network, do not require pytest
fixtures, and do not import the deleted module (verifying it is truly gone
via ``importlib`` ImportError).

Test layout follows the S66.1 convention: flat in ``tests/unit/``.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPSEARCH_ROOT = REPO_ROOT / "deepsearch"
TESTS_ROOT = REPO_ROOT / "tests"
AGENT_ROOT = DEEPSEARCH_ROOT / "agent"

DELETED_MODULE = "deepsearch.agent.source_quality"
DELETED_RELATIVE_NAME = "source_quality"

DELETED_SOURCE_QUALITY_SYMBOLS = (
    "BLOG_LISTICLE_DOMAINS",
    "DiversityFilterResult",
    "EmbeddingRateLimitedError",
    "HIGH_AUTHORITY_DOMAINS",
    "HIGH_AUTHORITY_TLDS",
    "LOW_AUTHORITY_DOMAINS",
    "MEDIUM_AUTHORITY_DOMAINS",
    "SourceQualityResult",
    "cosine_similarity",
    "embed_text",
    "enforce_source_diversity",
    "estimate_source_coherence",
    "estimate_source_quality",
    "estimate_source_relevance",
    "filter_sources_by_quality",
    "filter_sources_by_relevance",
    "get_objective_embedding",
    "get_unique_domains",
    "resolve_source_url",
)


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _iter_python_sources(root: Path):
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


# ---------------------------------------------------------------------------
# Whole-file deletion: source_quality must be unimportable.
# ---------------------------------------------------------------------------


def test_source_quality_cannot_be_imported() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(DELETED_MODULE)


def test_source_quality_path_does_not_exist() -> None:
    path = AGENT_ROOT / "source_quality.py"
    assert not path.exists(), f"source_quality.py re-introduced at {path}"


# ---------------------------------------------------------------------------
# No remaining absolute-form imports of the deleted module anywhere.
# ---------------------------------------------------------------------------


def _imports_absolute_module(node: ast.AST, target: str) -> bool:
    parent, _, leaf = target.rpartition(".")
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module == target:
            return True
        if module == parent:
            for alias in node.names:
                if alias.name == leaf:
                    return True
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == target:
                return True
    return False


def test_no_consumer_imports_source_quality_module() -> None:
    """No file under deepsearch/ or tests/ imports the deleted module."""
    offenders: list[str] = []
    for path in (*_iter_python_sources(DEEPSEARCH_ROOT), *_iter_python_sources(TESTS_ROOT)):
        if path.name == "test_s77_source_quality_invariants.py":
            continue
        try:
            tree = _module_ast(path)
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if _imports_absolute_module(node, DELETED_MODULE):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, (
        f"Re-imports of deleted module {DELETED_MODULE!r} detected:\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# No relative imports of source_quality from inside ``deepsearch/agent``.
# ---------------------------------------------------------------------------


def test_no_relative_imports_of_source_quality() -> None:
    offenders: list[str] = []
    for path in _iter_python_sources(AGENT_ROOT):
        try:
            tree = _module_ast(path)
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level >= 1 and node.module == DELETED_RELATIVE_NAME:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, (
        f"Relative imports of {DELETED_RELATIVE_NAME!r} detected in "
        f"deepsearch/agent:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# No imported-name reference to deleted source_quality symbols anywhere.
# Catches ``from deepsearch.agent.source_quality import X`` re-introductions
# even when ``X`` would be a valid symbol on a sibling module.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", DELETED_SOURCE_QUALITY_SYMBOLS)
def test_no_consumer_imports_deleted_source_quality_symbol(symbol: str) -> None:
    offenders: list[str] = []
    for path in (*_iter_python_sources(DEEPSEARCH_ROOT), *_iter_python_sources(TESTS_ROOT)):
        if path.name == "test_s77_source_quality_invariants.py":
            continue
        try:
            tree = _module_ast(path)
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == DELETED_MODULE or (
                    node.level >= 1 and module == DELETED_RELATIVE_NAME
                ):
                    for alias in node.names:
                        if alias.name == symbol:
                            offenders.append(
                                f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
                            )
    assert not offenders, (
        f"Imports of deleted source_quality symbol {symbol!r} detected:\n  "
        + "\n  ".join(offenders)
    )
