"""S82 m02 (PD135) RD2 invariants — citation quarantine partition.

S84 m02 REWROTE the guarantees pinned here (decision #823): the S82-era
quarantine deleted DEAD citations, which made it the amplifier that turned
false-dead verdicts into gutted reports (45-91 markers stripped per CAT run).
The partition now DOWNGRADES instead of deleting.

Guarantees pinned here (post-S84-m02):
  * DEAD (live=False AND URL in collected sources) SURVIVES the quarantine —
    downgraded to unverified, kept in ``protocol.citations`` with live=False.
  * ORPHAN (live=True, URL NOT in sources) is PRESERVED (#792).
  * UNKNOWN (live=None) and LIVE (live=True) are PRESERVED.
  * Runs on both the critiqued and critique_skipped paths.
  * A leaked ``[cite:URL]`` marker is silently stripped — the gap note is
    NEVER inserted for a claim whose URL has a collected source (s84-m02
    criterion 1); only hallucinated leaked markers gap-note (see
    test_s83_m03_invariants.py).
  * markers_quarantined telemetry keeps its name and int shape but now
    counts DOWNGRADED dead citations (acted-upon count, not deletions).
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import patch

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.citation_extraction import extract_citations
from deepsearch.agent.citation_liveness import (
    LivenessTelemetry,
    quarantine_dead_citations,
    strip_dead_cite_markers,
)
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import CritiqueResult
from deepsearch.report.citation import CitationAnnotation


def _ann(url: str, live: bool | None) -> CitationAnnotation:
    return CitationAnnotation(
        type="url_citation", start_index=0, end_index=1, url=url, title=None, live=live
    )


_SOURCES = [
    {"url": "https://dead.example/a", "title": "A"},
    {"url": "https://live.example/c", "title": "C"},
]


# ---------------------------------------------------------------------------
# quarantine_dead_citations — the deterministic partition.
# ---------------------------------------------------------------------------


def test_dead_citation_is_downgraded_not_quarantined() -> None:
    # S84 m02: dead-with-collected-source is DOWNGRADED (kept, live=False =
    # unverified), never deleted — deleting it was the CAT damage amplifier.
    dead = _ann("https://dead.example/a", live=False)  # in sources + dead
    surviving, quarantined = quarantine_dead_citations([dead], _SOURCES)
    assert surviving == [dead]
    assert quarantined == []


def test_orphan_citation_is_preserved() -> None:
    # S83 m03: a TRUE orphan is live=True + URL NOT in sources (reachable but
    # uncollected) -> PRESERVED (orphan removal deferred per #792). A live=False
    # not-in-sources URL is now HALLUCINATED and quarantined — that case moved to
    # test_s83_m03_invariants.py.
    orphan = _ann("https://orphan.example/b", live=True)
    surviving, quarantined = quarantine_dead_citations([orphan], _SOURCES)
    assert surviving == [orphan]
    assert quarantined == []


def test_unknown_and_live_are_preserved() -> None:
    unknown = _ann("https://dead.example/a", live=None)  # in sources but unprobed
    live = _ann("https://live.example/c", live=True)
    surviving, quarantined = quarantine_dead_citations([unknown, live], _SOURCES)
    assert surviving == [unknown, live]
    assert quarantined == []


def test_mixed_set_partitions_correctly() -> None:
    # S84 m02: dead survives (downgraded); only HALLUCINATED is quarantined
    # (see test_s83_m03_invariants.py for that class).
    dead = _ann("https://dead.example/a", live=False)
    orphan = _ann("https://orphan.example/b", live=True)
    live = _ann("https://live.example/c", live=True)
    unknown = _ann("https://dead.example/a", live=None)
    surviving, quarantined = quarantine_dead_citations(
        [dead, orphan, live, unknown], _SOURCES
    )
    assert quarantined == []
    assert surviving == [dead, orphan, live, unknown]


def test_quarantine_does_not_mutate_input_annotations() -> None:
    dead = _ann("https://dead.example/a", live=False)
    quarantine_dead_citations([dead], _SOURCES)
    assert dead.live is False  # untouched — pure partition


def test_url_normalisation_matches_liveness_gate() -> None:
    # A tracking-param / fragment variant of a COLLECTED dead URL must still
    # classify as DEAD (downgraded, kept) — a normalisation miss would call
    # it HALLUCINATED and hard-drop it, deleting a collected-source citation.
    dead = _ann("https://dead.example/a?utm_source=x#frag", live=False)
    surviving, quarantined = quarantine_dead_citations([dead], _SOURCES)
    assert surviving == [dead], "normalised URL must match the source set"
    assert quarantined == [], "a collected-source variant must never hard-drop"


# ---------------------------------------------------------------------------
# strip_dead_cite_markers — defensive body guard + re-extraction invariant.
# ---------------------------------------------------------------------------


def test_leaked_dead_marker_is_silently_stripped_never_gap_noted() -> None:
    # S84 m02 criterion 1: even when a DEAD annotation is (wrongly) passed in
    # the quarantined set, a collected-source URL can NEVER be gap-noted —
    # the class guard inside strip_dead_cite_markers is structural. Leaked
    # markers strip silently, exactly like the pre-critique extractor.
    dead = _ann("https://dead.example/a", live=False)
    markdown = "Claim A [cite:https://dead.example/a]. Claim B [cite:https://live.example/c]."
    stripped, count = strip_dead_cite_markers(markdown, [dead], _SOURCES)
    assert count == 2  # every leaked marker is acted on (stripped)
    assert "[cite:" not in stripped
    assert "[Data unavailable in primary sources]" not in stripped, (
        "gap-noting a collected-source claim is the outlawed CAT amplifier"
    )
    # The claim text itself survives untouched.
    assert "Claim A" in stripped and "Claim B" in stripped


def test_strip_is_noop_without_markers() -> None:
    dead = _ann("https://dead.example/a", live=False)
    markdown = "Claim A. Claim B."  # markers already stripped at extraction
    stripped, count = strip_dead_cite_markers(markdown, [dead], _SOURCES)
    assert count == 0
    assert stripped == markdown


def test_post_strip_reextraction_has_no_markers_at_all() -> None:
    """Re-extraction finds zero surviving markers of ANY class — annotate
    mode strips every leaked marker (silently for collected-source URLs),
    so a raw [cite:URL] can never ship in the persisted body."""
    dead = _ann("https://dead.example/a", live=False)
    markdown = "Claim A [cite:https://dead.example/a]. Claim C [cite:https://live.example/c]."
    stripped, _ = strip_dead_cite_markers(markdown, [dead], _SOURCES)
    _, reextracted = extract_citations(stripped)
    assert reextracted == []


# ---------------------------------------------------------------------------
# Wiring — quarantine runs in finalize before protocol generation.
# ---------------------------------------------------------------------------


def test_finalize_calls_quarantine_before_protocol() -> None:
    src = inspect.getsource(fnode)
    assert "quarantine_dead_citations(" in src
    # The reassignment must happen so the surviving list flows to protocol.
    assert "citation_annotations, quarantined_dead = quarantine_dead_citations(" in src


def _state() -> dict:
    return {
        "mission_id": "S82-M02-WIRING",
        "mission_objectives": ["c1"],
        "mission_context": {
            "title": "Quarantine Mission",
            "objective": "verify dead-link quarantine",
            "deliverables": ["Comparison Matrix"],
        },
        "sources_found": [],
        "telemetry": {},
    }


def _synthesis_with_markers() -> SynthesisResult:
    return SynthesisResult(
        output_markdown=(
            "# Report\n\nClaim A [cite:https://dead.example/a] is grounded in evidence "
            "with enough words to clear any substance floor comfortably here. "
            "Claim B [cite:https://orphan.example/b] adds more. "
            "Claim C [cite:https://live.example/c] closes it out with detail."
        ),
        step_count=5,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=10.0,
        token_usage={"input": 100, "output": 100, "total": 200},
        tool_call_summary={"total_calls": 3},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[{"step_index": 0, "type": "ActionStep"}],
        sources_collected=[
            {"url": "https://dead.example/a", "title": "A", "snippet": "x"},
            {"url": "https://live.example/c", "title": "C", "snippet": "z"},
        ],
        error=None,
    )


def _fake_liveness():
    """Return an async stand-in for check_citations_liveness that marks
    dead.example dead, orphan.example a live orphan, live.example live."""

    async def _run(citations, sources_collected, budget_seconds=45.0, scope_admission=None):
        for ann in citations:
            if "dead.example" in ann.url:
                ann.live = False  # in sources + dead -> DEAD (quarantined)
            elif "orphan.example" in ann.url:
                # S83 m03: a TRUE orphan is reachable-but-uncollected (live=True)
                # and must SURVIVE. (A live=False not-in-sources URL is
                # hallucinated and now quarantined — see test_s83_m03.)
                ann.live = True
            else:
                ann.live = True
        telemetry = LivenessTelemetry(
            checked=len(citations), live=2, dead=1, orphan=1, unknown=0,
            hallucinated_url_count=0, distinct_urls_checked=len(citations),
        )
        return list(citations), telemetry

    return _run


def test_finalize_downgrades_dead_in_shipped_protocol(tmp_path, monkeypatch) -> None:
    """End-to-end (S84 m02): a probe-dead citation SURVIVES into
    protocol.citations carrying live=False (unverified) and is counted in
    markers_quarantined as the downgraded count; orphan + live survive."""

    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))
    captured: dict = {}

    def _capture_protocol(protocol, **kwargs):
        captured["citations"] = list(protocol.citations)
        return str(tmp_path / "protocol.json")

    critique = CritiqueResult(
        output_markdown="",  # force critique_skipped? no — passthrough below
        claim_count=0,
        gap_notes_inserted=0,
        runtime_seconds=0.0,
        token_usage={"input": 0, "output": 0, "total": 0},
        error=None,
    )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=_synthesis_with_markers()
        ), patch.object(
            fnode, "run_critique", return_value=critique
        ), patch.object(
            fnode, "check_citations_liveness", new=_fake_liveness()
        ), patch.object(
            fnode, "save_protocol", side_effect=_capture_protocol
        ):
            return await fnode.codeagent_finalize_node(_state())

    result = asyncio.run(run())

    shipped_urls = {c.url for c in captured["citations"]}
    assert "https://dead.example/a" in shipped_urls, (
        "dead citation must be downgraded-not-deleted (S84 m02)"
    )
    dead_shipped = [
        c for c in captured["citations"] if c.url == "https://dead.example/a"
    ]
    assert all(c.live is False for c in dead_shipped), (
        "the downgraded citation carries live=False — the unverified flag"
    )
    assert "https://orphan.example/b" in shipped_urls, "live orphan preserved (removal deferred #792)"
    assert "https://live.example/c" in shipped_urls, "live citation preserved"
    # markers_quarantined keeps name + int shape; it now counts DOWNGRADED dead.
    assert result["synthesis_telemetry"]["markers_quarantined"] == 1
    # liveness_telemetry keeps its PD7 shape — markers_quarantined is NOT in it.
    assert "markers_quarantined" not in result["synthesis_telemetry"]["liveness_telemetry"]
