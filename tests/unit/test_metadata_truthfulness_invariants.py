"""S74 m01 — static AST guards for the metadata-truthfulness fixes.

RD2 tenth-consecutive sprint contribution. Runtime tests in
``tests/unit/test_report_generator.py``, ``tests/unit/test_citation_extraction.py``,
and ``tests/unit/test_contract_to_prompt.py`` cover behaviour; this module
locks the *structural* shape of the Bug A + Bug B fixes so a future refactor
cannot quietly regress them.

What's locked:

1. ``deepsearch.report.generator._build_telemetry_section`` does NOT carry
   the three S74-stripped row literals (``Research loops``, ``Sources
   gathered``, ``Findings recorded``). The rows read from legacy
   ``state["research_loop_count"]`` / ``state["sources_found"]`` /
   ``state["findings"]`` which post-S64 entry-point flip stop populating —
   re-introducing them re-opens the S73-WIRING-02 lying-table bug.
2. ``deepsearch.agent.citation_extraction`` carries both the primary
   ``[cite:`` extractor surface and the S74 numbered-references fallback
   surface. S96 reconciles both passes, deduplicating each claim/URL pair, and
   resolves inline ``[N]`` markers via a ``## References`` / ``## Sources``
   parse — without it the S73-WIRING-02 ``References: 0`` preamble bug
   stays unfixed for the all-numbered failure mode.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = REPO_ROOT / "deepsearch" / "report" / "generator.py"
CITATION_EXTRACTION_PATH = REPO_ROOT / "deepsearch" / "agent" / "citation_extraction.py"


# ---------------------------------------------------------------------------
# Bug A — telemetry section row strip (PD31).
# ---------------------------------------------------------------------------


# NOTE: S74 m01 telemetry-section row-strip guards (Research loops, Sources
# gathered, Findings recorded, kept-rows) are obsolete after S75 m01: the
# whole `_build_telemetry_section` function (and the `## Telemetry` heading
# it built) was deleted in m01 Phase C.1. The structural guard against
# re-introduction lives in
# `tests/unit/test_s75_deletion_invariants.py::test_generator_has_no_telemetry_section_function`
# and `test_generator_does_not_emit_telemetry_heading`.


# ---------------------------------------------------------------------------
# Bug B — citation_extraction primary + fallback path coexistence (PD32).
# ---------------------------------------------------------------------------


def _extractor_module_source() -> str:
    return CITATION_EXTRACTION_PATH.read_text(encoding="utf-8")


def _extractor_function_names() -> set[str]:
    tree = ast.parse(_extractor_module_source())
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_extractor_module_carries_primary_cite_marker_surface() -> None:
    """The `[cite:` primary path must remain present.

    PD32 keeps the inline `[cite:URL]` marker as the primary path; the
    numbered fallback is a safety net, not a replacement.
    """

    source = _extractor_module_source()
    assert "[cite:" in source, (
        "citation_extraction must carry the `[cite:` primary marker surface — "
        "removing it makes the numbered fallback the sole path, breaking "
        "S67/S68 grounding contracts."
    )
    assert "_CITE_MARKER_RE" in source, (
        "citation_extraction must continue to expose `_CITE_MARKER_RE` "
        "(the primary `[cite:URL]` regex)."
    )


def test_extractor_module_carries_numbered_fallback_surface() -> None:
    """The S74 numbered-references fallback helper must be present.

    Required helper symbols for the fallback:
      * `_extract_numbered_fallback` — the fallback orchestrator
      * `_build_references_map` — parses `## References` / `## Sources`
        section into `{N: URL}`
      * `_NUMBERED_MARKER_RE` — inline `[N]` regex
      * `_REFERENCES_HEADING_RE` — `## References` / `## Sources` heading
        regex

    Drift here regresses the S73-WIRING-02 `References: 0` failure mode for
    the all-numbered case.
    """

    source = _extractor_module_source()
    function_names = _extractor_function_names()

    assert "_extract_numbered_fallback" in function_names, (
        "citation_extraction must expose `_extract_numbered_fallback` — the "
        "numbered-references pass reconciled with the primary `[cite:URL]` "
        "pass."
    )
    assert "_build_references_map" in function_names, (
        "citation_extraction must expose `_build_references_map` — the "
        "`## References` / `## Sources` section parser used by the fallback."
    )
    assert "_NUMBERED_MARKER_RE" in source, (
        "citation_extraction must expose `_NUMBERED_MARKER_RE` — the inline "
        "`[N]` regex used by the fallback."
    )
    assert "_REFERENCES_HEADING_RE" in source, (
        "citation_extraction must expose `_REFERENCES_HEADING_RE` — the "
        "`## References` / `## Sources` heading regex used by the fallback."
    )


def test_mixed_extraction_keeps_numbered_evidence_without_double_credit() -> None:
    """S96 repairs undercount while preserving the original deduplication intent."""
    from deepsearch.agent.citation_extraction import extract_citations

    raw = (
        "Shared claim [1] [cite:https://example.org/a][2].\n\n"
        "## References\n[1] A — https://example.org/a\n[2] B — https://example.org/b\n"
    )
    _, annotations = extract_citations(raw)
    assert [annotation.url for annotation in annotations] == [
        "https://example.org/a", "https://example.org/b",
    ]


def test_extractor_module_remains_io_free_after_fallback_addition() -> None:
    """The fallback must not introduce I/O imports.

    Sister guard to test_citation_invariants.test_extractor_module_has_no_io_imports.
    Restated here so the S74 fallback addition's pure-function status has a
    dedicated regression check tied to the S74 sprint memo. The fallback is
    pure-function — `## References` parsing happens against the in-memory
    markdown; no network / disk I/O is permitted.
    """

    source = _extractor_module_source()
    forbidden_substrings = (
        "import requests",
        "import httpx",
        "import urllib.request",
        "from urllib import request",
        "import aiohttp",
        "import smolagents",
        "import openai",
    )
    for forbidden in forbidden_substrings:
        assert forbidden not in source, (
            f"citation_extraction must remain I/O-free after the S74 fallback "
            f"addition. Forbidden substring detected: '{forbidden}'."
        )
