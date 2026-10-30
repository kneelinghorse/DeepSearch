"""S82 m05 (PD135) RD2 invariants — fabrication-metric split + reconciliation.

Three guarantees:
  1. hallucinated_url_count is NO LONGER a verbatim alias for the set-membership
     orphan count — it counts only orphans whose HTTP probe actually FAILED
     (NXDOMAIN/404). Orphans are now PROBED (the short-circuit is gone), so a
     not-in-sources URL that RESOLVES is an orphan-live, not fabrication.
  2. ONE shared normalisation+classification helper (classify_citation +
     build_source_url_set) is used by the liveness gate AND critique, so the
     two previously-divergent orphan universes reconcile (a tracking-param
     variant of an in-sources URL is no longer mislabelled orphan).
  3. The report header surfaces VERIFIED references (live-only); the raw
     distinct-cited count stays in the forensic block.
"""

from __future__ import annotations

import asyncio
import inspect

from deepsearch.agent import citation_liveness, critique
from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.citation_liveness import (
    CLASSIFICATION_DEAD,
    CLASSIFICATION_HALLUCINATED,
    CLASSIFICATION_LIVE,
    CLASSIFICATION_ORPHAN,
    CLASSIFICATION_UNKNOWN,
    build_source_url_set,
    check_citations_liveness,
    classify_citation,
)
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.report.citation import CitationAnnotation


def _ann(url: str, live: bool | None = None) -> CitationAnnotation:
    return CitationAnnotation(
        type="url_citation", start_index=0, end_index=1, url=url, title=None, live=live
    )


def _url_aware_probe(dead_substrings: tuple[str, ...]):
    def probe(url: str, timeout_s: float = 5.0) -> dict:
        alive = not any(s in url for s in dead_substrings)
        return {"alive": alive, "status_code": 200 if alive else 404, "final_url": url}

    return probe


# ---------------------------------------------------------------------------
# Shared 5-state classifier.
# ---------------------------------------------------------------------------


def test_classify_citation_five_state() -> None:
    s = build_source_url_set([{"url": "https://x.org/a"}])

    def c(live: bool | None, url: str) -> str:
        return classify_citation(live=live, url=url, source_url_set=s)

    assert c(True, "https://x.org/a") == CLASSIFICATION_LIVE  # in sources, alive
    assert c(True, "https://y.org/b") == CLASSIFICATION_ORPHAN  # not in sources, alive
    assert c(False, "https://x.org/a") == CLASSIFICATION_DEAD  # in sources, dead
    assert c(False, "https://y.org/b") == CLASSIFICATION_HALLUCINATED  # not in sources, dead
    assert c(None, "https://x.org/a") == CLASSIFICATION_UNKNOWN  # unprobed


def test_classify_normalises_tracking_params() -> None:
    # A tracking-param variant of an in-sources URL must classify as in-sources.
    s = build_source_url_set([{"url": "https://x.org/a"}])
    assert (
        classify_citation(live=True, url="https://x.org/a?utm_source=feed#frag", source_url_set=s)
        == CLASSIFICATION_LIVE
    )


# ---------------------------------------------------------------------------
# Metric split — orphan (set-membership) vs hallucinated (probe-failed).
# ---------------------------------------------------------------------------


def test_hallucinated_no_longer_aliases_orphan() -> None:
    """Mix of orphan-live + hallucinated: orphan counts both, hallucinated only
    the probe-failed one."""

    import unittest.mock as mock

    citations = [
        _ann("https://in.example/a"),  # in sources, alive -> live
        _ann("https://orphan-live.example/b"),  # not in sources, alive -> orphan
        _ann("https://dead.example/c"),  # not in sources, dead -> hallucinated
    ]
    sources = [{"url": "https://in.example/a"}]

    with mock.patch.object(citation_liveness, "url_liveness", _url_aware_probe(("dead.example",))):
        _, tel = asyncio.run(check_citations_liveness(citations, sources))

    assert tel.live == 1
    assert tel.dead == 0
    assert tel.orphan == 2  # set-membership: both not-in-sources URLs
    assert tel.hallucinated_url_count == 1  # only the probe-failed orphan
    assert tel.hallucinated_url_count != tel.orphan, "must not be an alias"


def test_genuinely_retrieved_orphan_not_counted_hallucinated() -> None:
    """A not-in-sources URL that RESOLVES is orphan-live, never hallucinated."""

    import unittest.mock as mock

    citations = [_ann("https://reachable.example/x")]
    with mock.patch.object(citation_liveness, "url_liveness", _url_aware_probe(())):
        annotations, tel = asyncio.run(check_citations_liveness(citations, []))

    assert tel.orphan == 1
    assert tel.hallucinated_url_count == 0
    assert annotations[0].live is True  # probed and resolved


# ---------------------------------------------------------------------------
# Reconciliation — gate + critique agree via the shared helper.
# ---------------------------------------------------------------------------


def test_gate_and_critique_reconcile_on_normalised_urls() -> None:
    """Pre-m05 critique used RAW URLs, so a tracking-param variant of an
    in-sources URL was mislabelled 'orphan'. Now critique builds its set via the
    shared build_source_url_set (normalised) and delegates to classify_citation,
    so the gate and critique never disagree."""

    sources = [{"url": "https://x.org/a"}]
    source_set = build_source_url_set(sources)
    ann = _ann("https://x.org/a?utm_source=feed", live=True)

    gate_cls = classify_citation(live=ann.live, url=ann.url, source_url_set=source_set)
    critique_cls = critique._classify_citation_for_prompt(ann, source_set, set())
    assert gate_cls == critique_cls == CLASSIFICATION_LIVE

    # End-to-end: the critique liveness-context block (which builds its own set)
    # renders this marker as live, not orphan.
    messages = critique._build_critique_messages(
        "# Draft", sources, {"title": "T", "objective": "O"}, citations=[ann]
    )
    block = messages[1]["content"]
    assert "[cite:https://x.org/a?utm_source=feed]  →  live" in block


def test_critique_builds_set_via_shared_builder() -> None:
    # Guard: critique must use the shared normalised builder (source-level).
    src = inspect.getsource(critique._build_critique_messages)
    assert "build_source_url_set(" in src


# ---------------------------------------------------------------------------
# Verified-references header (live-only); raw distinct-cited stays forensic.
# ---------------------------------------------------------------------------


def _synthesis() -> SynthesisResult:
    return SynthesisResult(
        output_markdown="# R",
        step_count=4,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=10.0,
        token_usage={"input": 0, "output": 0, "total": 0},
        tool_call_summary={"total_calls": 1},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[],
        sources_collected=[{"url": "https://x.org/a"}],
    )


def test_header_references_is_verified_live_count() -> None:
    synthesis_telemetry = {
        "citation_telemetry": {"distinct_urls_cited": 10, "citations_extracted": 12},
        "liveness_telemetry": {"checked": 10, "live": 3, "dead": 5, "orphan": 2},
    }
    meta = fnode._build_report_metadata(
        synthesis_result=_synthesis(),
        synthesis_telemetry=synthesis_telemetry,
        citation_annotations=[],
        telemetry={},
    )
    assert meta.references == 3, "header references must be the live-only count"
    # Raw distinct-cited count is retained in the forensic block, not headline.
    assert meta.forensic["citation_telemetry"]["distinct_urls_cited"] == 10

    preamble = fnode._render_report_metadata_preamble(meta)
    assert "**Verified references:** 3" in preamble


def test_header_falls_back_when_liveness_absent() -> None:
    # No liveness_telemetry -> fall back to distinct_urls_cited (legacy path).
    meta = fnode._build_report_metadata(
        synthesis_result=_synthesis(),
        synthesis_telemetry={"citation_telemetry": {"distinct_urls_cited": 7}},
        citation_annotations=[],
        telemetry={},
    )
    assert meta.references == 7


# ---------------------------------------------------------------------------
# Budget — configurable + citation-heavy observability.
# ---------------------------------------------------------------------------


def test_budget_is_configurable_and_heavy_threshold_pinned() -> None:
    assert "budget_seconds" in inspect.signature(check_citations_liveness).parameters
    assert citation_liveness._CITATION_HEAVY_THRESHOLD == 80
