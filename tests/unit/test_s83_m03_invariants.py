"""S83 m03 RD2 invariants — safe orphan/hallucinated citation quarantine.

Extends the S82 m02 deterministic dead-citation quarantine to ALSO drop the
HALLUCINATED case (``live is False`` AND URL not-in-sources — a probe-failed
NXDOMAIN/404 the model invented) while PRESERVING reachable orphans
(``live is True``, not-in-sources) and unprobed-at-budget citations
(``live is None``). The explicit unblocked follow-on to decision #792 — orphan
REMOVAL stays deferred; m03 only makes the dead+hallucinated half of the
``contract_to_prompt`` removal promise true in code.

Pinned here:
  * the 4-way partition (dead+hallucinated quarantined; orphan+unknown survive);
  * orphan (live, not-in-sources) and unknown (live=None) PRESERVED;
  * the telemetry split is ADDITIVE — markers_quarantined STAYS an int == the
    dead count; hallucinated_quarantined is a NEW synthesis_telemetry sibling;
  * PD7: hallucinated_quarantined is NOT inside liveness_telemetry;
  * the gate's orphan signal is NOT decremented by hallucinated quarantine;
  * an orphan still ships in protocol.citations (the #792 guard).
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import patch

import pytest

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.citation_extraction import extract_citations
from deepsearch.agent.citation_liveness import (
    CLASSIFICATION_HALLUCINATED,
    LivenessTelemetry,
    build_source_url_set,
    classify_citation,
    quarantine_dead_citations,
    strip_dead_cite_markers,
)
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import CritiqueResult
from deepsearch.agent.outcome import SOFT_ORPHAN, evaluate_outcome_gate
from deepsearch.report.citation import CitationAnnotation


def _ann(url: str, live: bool | None) -> CitationAnnotation:
    return CitationAnnotation(
        type="url_citation", start_index=0, end_index=1, url=url, title=None, live=live
    )


# Two collected sources — the membership universe for the classifier.
_SOURCES = [
    {"url": "https://dead.example/a", "title": "A"},
    {"url": "https://live.example/c", "title": "C"},
]


# ---------------------------------------------------------------------------
# quarantine_dead_citations — the broadened 4-way partition.
# ---------------------------------------------------------------------------


def test_partition_drops_only_hallucinated_keeps_dead_orphan_and_unknown() -> None:
    # S84 m02: the drop set narrowed to HALLUCINATED only — DEAD survives
    # downgraded (kept with live=False = unverified). Hard quarantine of
    # fabricated URLs is unchanged.
    dead = _ann("https://dead.example/a", live=False)  # in sources, dead -> DEAD
    hallucinated = _ann("https://fabricated.example/x", live=False)  # not-in, dead
    orphan = _ann("https://orphan.example/o", live=True)  # not-in, alive -> ORPHAN
    unknown = _ann("https://live.example/c", live=None)  # unprobed -> UNKNOWN

    surviving, quarantined = quarantine_dead_citations(
        [dead, hallucinated, orphan, unknown], _SOURCES
    )
    assert quarantined == [hallucinated]
    assert surviving == [dead, orphan, unknown]


def test_orphan_live_not_in_sources_is_preserved() -> None:
    """The load-bearing decision-#792 guard: a reachable, uncollected URL
    (live=True, not-in-sources) SURVIVES — orphan removal stays deferred."""
    orphan = _ann("https://orphan.example/o", live=True)
    surviving, quarantined = quarantine_dead_citations([orphan], _SOURCES)
    assert surviving == [orphan]
    assert quarantined == []


def test_unknown_is_preserved_regardless_of_source_membership() -> None:
    """live=None never drops, in-sources or not (budget-exhausted, not probed)."""
    in_sources = _ann("https://dead.example/a", live=None)
    not_in_sources = _ann("https://fabricated.example/x", live=None)
    surviving, quarantined = quarantine_dead_citations(
        [in_sources, not_in_sources], _SOURCES
    )
    assert surviving == [in_sources, not_in_sources]
    assert quarantined == []


def test_quarantine_does_not_mutate_input_annotations() -> None:
    hallucinated = _ann("https://fabricated.example/x", live=False)
    quarantine_dead_citations([hallucinated], _SOURCES)
    assert hallucinated.live is False  # pure partition — untouched


def test_post_strip_reextraction_has_no_hallucinated_marker() -> None:
    """Re-extraction finds zero surviving markers after strip: the leaked
    hallucinated marker is gap-noted (no collected source — the one legal
    gap-note case), and every other leaked marker strips silently (S84 m02
    annotate mode)."""
    hallucinated = _ann("https://fabricated.example/x", live=False)
    markdown = (
        "Claim A [cite:https://fabricated.example/x]. "
        "Claim C [cite:https://live.example/c]."
    )
    stripped, count = strip_dead_cite_markers(markdown, [hallucinated], _SOURCES)
    assert count == 2  # gap-noted fabricated + silently stripped live marker
    assert "[Data unavailable in primary sources]" in stripped
    _, reextracted = extract_citations(stripped)
    assert reextracted == []
    # And nothing re-extracted classifies as hallucinated against the sources.
    source_set = build_source_url_set(_SOURCES)
    assert all(
        classify_citation(live=False, url=a.url, source_url_set=source_set)
        != CLASSIFICATION_HALLUCINATED
        or a.url != "https://fabricated.example/x"
        for a in reextracted
    )


# ---------------------------------------------------------------------------
# Gate non-regression — quarantine never touches the gate's orphan signal.
# ---------------------------------------------------------------------------


def test_gate_still_flags_orphans_independent_of_quarantine() -> None:
    """liveness_telemetry.orphan stays the set-membership total (orphan +
    hallucinated, computed once in the liveness gate). With 4 orphan / 0
    hallucinated the gate's quality list still carries 'orphan_citations_4'."""
    synthesis_telemetry = {
        "hit_max_steps": False,
        "self_terminated": True,
        "citation_telemetry": {"citations_extracted": 10},
        "deliverable_coverage": {"shortfall_score": 0.0, "satisfied": ["x"], "missing": []},
        "liveness_telemetry": {
            "checked": 10,
            "live": 6,
            "dead": 0,
            "orphan": 4,
            "unknown": 0,
            "hallucinated_url_count": 0,
        },
    }
    assert SOFT_ORPHAN == 3  # boundary the fixture clears
    gate = evaluate_outcome_gate(synthesis_telemetry, report_word_count=1000)
    assert "orphan_citations_4" in gate.quality_gate_failures


# ---------------------------------------------------------------------------
# Finalize wiring — additive split + PD7 + orphan-promise on both paths.
# ---------------------------------------------------------------------------


def _state() -> dict:
    return {
        "mission_id": "S83-M03-WIRING",
        "mission_objectives": ["c1"],
        "mission_context": {
            "title": "Quarantine Mission",
            "objective": "verify hallucinated quarantine",
            "deliverables": ["Comparison Matrix"],
        },
        "sources_found": [],
        "telemetry": {},
    }


def _synthesis_with_four_markers() -> SynthesisResult:
    return SynthesisResult(
        output_markdown=(
            "# Report\n\nClaim A [cite:https://dead.example/a] is grounded in evidence "
            "with enough words to clear any substance floor comfortably here. "
            "Claim H [cite:https://fabricated.example/h] is invented. "
            "Claim O [cite:https://orphan.example/o] is reachable but uncollected. "
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


def _fake_liveness_one_dead_one_hallucinated():
    """dead.example -> dead (in sources), fabricated.example -> dead (not in
    sources = hallucinated), orphan.example -> live (not in sources = orphan),
    live.example -> live (in sources). orphan telemetry stays the set-membership
    total (orphan + hallucinated == 2)."""

    async def _run(citations, sources_collected, budget_seconds=45.0, scope_admission=None):
        for ann in citations:
            if "dead.example" in ann.url or "fabricated.example" in ann.url:
                ann.live = False
            else:
                ann.live = True
        telemetry = LivenessTelemetry(
            checked=len(citations), live=2, dead=1, orphan=2, unknown=0,
            hallucinated_url_count=1, distinct_urls_checked=len(citations),
        )
        return list(citations), telemetry

    return _run


@pytest.mark.parametrize("critique_skipped", [False, True])
def test_finalize_quarantines_hallucinated_and_splits_counts(
    tmp_path, monkeypatch, critique_skipped: bool
) -> None:
    """End-to-end on BOTH critique paths (S84 m02): the hallucinated URL is
    dropped from protocol.citations while dead (downgraded, live=False) and
    orphan SURVIVE; markers_quarantined==1 (now the DOWNGRADED dead count),
    hallucinated_quarantined==1, and the PD7 liveness shape is untouched."""

    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))
    captured: dict = {}

    def _capture_protocol(protocol, **kwargs):
        captured["citations"] = list(protocol.citations)
        return str(tmp_path / "protocol.json")

    # The successful critique must preserve cited claim text so the persisted
    # coordinate rebase can prove each surviving citation exactly. The
    # critique_skipped path forces an error and uses the same stripped draft.
    stripped_draft, _ = extract_citations(_synthesis_with_four_markers().output_markdown)
    critique = CritiqueResult(
        output_markdown=stripped_draft,
        claim_count=0,
        gap_notes_inserted=0,
        runtime_seconds=0.0,
        token_usage={"input": 0, "output": 0, "total": 0},
        error="forced skip" if critique_skipped else None,
    )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=_synthesis_with_four_markers()
        ), patch.object(
            fnode, "run_critique", return_value=critique
        ), patch.object(
            fnode, "check_citations_liveness", new=_fake_liveness_one_dead_one_hallucinated()
        ), patch.object(
            fnode, "save_protocol", side_effect=_capture_protocol
        ):
            return await fnode.codeagent_finalize_node(_state())

    result = asyncio.run(run())

    shipped_urls = {c.url for c in captured["citations"]}
    assert "https://dead.example/a" in shipped_urls, (
        "dead downgraded-not-deleted (S84 m02)"
    )
    assert "https://fabricated.example/h" not in shipped_urls, "hallucinated quarantined"
    assert "https://orphan.example/o" in shipped_urls, "live orphan preserved (#792)"
    assert "https://live.example/c" in shipped_urls, "live preserved"

    s_tel = result["synthesis_telemetry"]
    # markers_quarantined keeps its int shape; post-m02 it counts the
    # DOWNGRADED dead citations. hallucinated_quarantined counts the drop.
    assert s_tel["markers_quarantined"] == 1
    assert isinstance(s_tel["markers_quarantined"], int)
    assert s_tel["hallucinated_quarantined"] == 1
    # PD7 shape pin: hallucinated_quarantined lives at the synthesis_telemetry
    # top level, NEVER inside liveness_telemetry; orphan signal not decremented.
    assert "hallucinated_quarantined" in s_tel
    assert "hallucinated_quarantined" not in s_tel["liveness_telemetry"]
    assert "markers_quarantined" not in s_tel["liveness_telemetry"]
    assert s_tel["liveness_telemetry"]["orphan"] == 2, (
        "quarantining the hallucinated citation must not decrement the gate's "
        "orphan signal (set-membership total, computed once in the liveness gate)"
    )


# ---------------------------------------------------------------------------
# Structural guard — quarantine reassignment precedes the outcome gate.
# ---------------------------------------------------------------------------


def test_quarantine_precedes_outcome_gate_in_finalize() -> None:
    src = inspect.getsource(fnode)
    assert "quarantine_dead_citations(" in src
    assert "evaluate_outcome_gate(" in src
    assert src.index("quarantine_dead_citations(") < src.index("evaluate_outcome_gate("), (
        "the quarantine must run before the outcome gate so a result emptied of "
        "fabricated citations is still gated"
    )
    # The m02 wiring guard's exact reassignment line is preserved.
    assert "citation_annotations, quarantined_dead = quarantine_dead_citations(" in src
