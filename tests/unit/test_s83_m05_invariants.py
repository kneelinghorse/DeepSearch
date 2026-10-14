"""S83 m05 RD2 invariants — quality-telemetry series + gate-boundary hardening.

Four observability/hardening instruments:

(a) QUALITY-TELEMETRY SERIES (decision #803) — a pure ``build_quality_record``
    over the finalize synthesis_telemetry + the now-persisted report word count,
    emitted by the worker as a sibling of ``telemetry['final_outcome']`` and
    seeded into ``cmos/telemetry/series/quality_series.jsonl`` with the PD137 row.
(b) PD116c FORENSIC DUAL-COUNT — ``numbered_refs_parsed`` surfaces how many
    numbered refs the body carries WITHOUT changing what ``extract_citations``
    ships (byte-identical); it never feeds the gate.
(c) GATE-BOUNDARY FIXTURES — the sub-200-word 0-citation CLEAN escape, the
    ``Verified references: 0`` header on a live=0 mission, and the
    dead_rate(PRE-quarantine) vs markers_quarantined(POST-quarantine) epoch pin.
(d) CI ASSERTION pinning the S82 invariant COLLECTED-test count == 70.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.citation_extraction import extract_citations
from deepsearch.agent.codeagent_runner import DEFAULT_MAX_STEPS, SynthesisResult
from deepsearch.agent.critique import CritiqueResult
from deepsearch.agent.outcome import (
    DELIVERY_QUALITY_FAILED,
    DELIVERY_QUALITY_USABLE_WITH_WARNING,
    MIN_REPORT_WORDS,
    MissionOutcome,
    evaluate_outcome_gate,
)
from deepsearch.agent.quality_series import (
    _STEP_BUDGET,
    build_quality_record,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# Real calibration-series seed (provenance: tests/fixtures/PROVENANCE.md).


# PD137 (62eab1b0) seed numbers — the cross-store degraded-tier datapoint.
PD137_SYNTHESIS_TELEMETRY = {
    "citation_telemetry": {"citations_extracted": 149, "distinct_urls_cited": 35},
    "liveness_telemetry": {"live": 30, "dead": 1, "orphan": 4, "hallucinated_url_count": 0},
    "delivery_quality": "degraded",
    "markers_quarantined": 2,
    "self_terminated": True,
    "hit_max_steps": False,
    "step_count": 19,
}
PD137_WORD_COUNT = 9150


# ===========================================================================
# (a) build_quality_record + persistence + emission.
# ===========================================================================


def test_build_quality_record_pd137_shape_and_values() -> None:
    record = build_quality_record(PD137_SYNTHESIS_TELEMETRY, PD137_WORD_COUNT)
    assert record["word_count"] == 9150
    assert record["citations_extracted"] == 149
    assert record["distinct_urls"] == 35
    assert record["delivery_quality"] == "degraded"
    assert record["markers_quarantined"] == 2
    assert record["liveness"] == {
        "live": 30,
        "dead": 1,
        "orphan": 4,
        "unknown": 0,
        "hallucinated": 0,
        "available": True,
    }
    assert record["critique"]["status"] == "unknown"
    assert record["critique"]["applied"] is None
    assert record["retrieval_failures"]["total"] is None
    assert record["recovery"]["outcome"] == "unknown"
    assert record["self_terminated"] is True
    assert record["hit_max_steps"] is False
    assert record["terminating_step"] == 19
    assert record["budget"] == 40


def test_build_quality_record_maps_hallucinated_url_count() -> None:
    """The builder must map the REAL to_dict key ``hallucinated_url_count`` ->
    record ``hallucinated``; a non-zero seed makes a mis-map fail loudly."""
    st = {
        **PD137_SYNTHESIS_TELEMETRY,
        "liveness_telemetry": {"live": 5, "dead": 2, "orphan": 7, "hallucinated_url_count": 3},
    }
    record = build_quality_record(st, 100)
    assert record["liveness"]["hallucinated"] == 3, (
        "builder must read hallucinated_url_count, not a 'hallucinated' key"
    )


def test_build_quality_record_defensive_on_empty_and_malformed() -> None:
    """Empty / malformed telemetry coerces to 0/False with no raise."""
    empty = build_quality_record({}, None)
    assert empty["word_count"] == 0
    assert empty["citations_extracted"] == 0
    assert empty["liveness"] == {
        "live": 0,
        "dead": 0,
        "orphan": 0,
        "unknown": 0,
        "hallucinated": 0,
        "available": False,
    }
    assert empty["critique"]["status"] == "unknown"
    assert empty["critique"]["applied"] is None
    assert empty["recovery"]["passes"] is None
    assert empty["self_terminated"] is False
    assert empty["budget"] == _STEP_BUDGET

    malformed = build_quality_record(
        {"citation_telemetry": "nope", "liveness_telemetry": [1, 2, 3], "step_count": "x"},
        "not-an-int",
    )
    assert malformed["citations_extracted"] == 0
    assert malformed["liveness"]["dead"] == 0
    assert malformed["terminating_step"] == 0
    assert malformed["word_count"] == 0


def test_terminating_step_equals_step_count_no_off_by_one() -> None:
    """terminating_step is step_count verbatim (codeagent_runner step_count is a
    1-based len(action_steps)); PD137 step_count 19 -> terminating_step 19."""
    record = build_quality_record({"step_count": 19}, 0)
    assert record["terminating_step"] == 19


def test_step_budget_matches_runner_default() -> None:
    """Drift guard: the leaf module's budget constant mirrors the locked
    runner DEFAULT_MAX_STEPS."""
    assert _STEP_BUDGET == DEFAULT_MAX_STEPS




def test_worker_emits_quality_record_sibling_of_final_outcome() -> None:
    """Structural guard: the worker builds the quality record and emits it as a
    sibling of telemetry['final_outcome'] (after the final_outcome assignment)."""
    src = (REPO_ROOT / "deepsearch" / "worker" / "main.py").read_text()
    assert "build_quality_record(" in src
    assert 'telemetry["quality_record"]' in src
    assert src.index('telemetry["final_outcome"]') < src.index('telemetry["quality_record"]')


# --- finalize drives the persistence so live emission is never word_count=0 ---


def _state() -> dict:
    return {
        "mission_id": "S83-M05-WIRING",
        "mission_objectives": ["c1"],
        "mission_context": {
            "title": "Series Mission",
            "objective": "verify quality record plumbing",
            "deliverables": ["Comparison Matrix"],
        },
        "sources_found": [],
        "telemetry": {},
    }


def _synthesis(markdown: str, sources: list[dict] | None = None) -> SynthesisResult:
    return SynthesisResult(
        output_markdown=markdown,
        step_count=5,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=10.0,
        token_usage={"input": 100, "output": 100, "total": 200},
        tool_call_summary={"total_calls": 3},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[{"step_index": 0, "type": "ActionStep"}],
        sources_collected=sources if sources is not None else [],
        error=None,
    )


def _passthrough_critique(markdown: str) -> CritiqueResult:
    return CritiqueResult(
        output_markdown=markdown,
        claim_count=0,
        gap_notes_inserted=0,
        runtime_seconds=0.0,
        token_usage={"input": 0, "output": 0, "total": 0},
        error=None,
    )


def _run_finalize(state: dict, synthesis: SynthesisResult, fake_liveness=None):
    critique = _passthrough_critique(synthesis.output_markdown)

    async def run():
        patches = [
            patch.object(fnode, "run_codeagent_synthesis", return_value=synthesis),
            patch.object(fnode, "run_critique", return_value=critique),
        ]
        if fake_liveness is not None:
            patches.append(patch.object(fnode, "check_citations_liveness", new=fake_liveness))
        from contextlib import ExitStack

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            return await fnode.codeagent_finalize_node(state)

    return asyncio.run(run())


def test_finalize_persists_report_word_count(tmp_path, monkeypatch) -> None:
    """report_word_count is persisted onto synthesis_telemetry so the worker's
    final_state carries it (not a silent word_count=0)."""
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))
    body = "# Report\n\n" + ("word " * 120)  # >0 words, no citations
    result = _run_finalize(_state(), _synthesis(body))
    s_tel = result["synthesis_telemetry"]
    assert "report_word_count" in s_tel
    assert s_tel["report_word_count"] > 0
    # And the builder over the live final_state yields a non-zero word_count.
    record = build_quality_record(s_tel, s_tel["report_word_count"])
    assert record["word_count"] == s_tel["report_word_count"] > 0


# ===========================================================================
# (b) PD116c forensic dual-count.
# ===========================================================================


_MIXED_MARKDOWN = (
    "Claim X [cite:https://x.example/a]. "
    "Fact one [1]. Fact two [2]. Fact three [3].\n\n"
    "## References\n"
    "[1] https://r1.example\n"
    "[2] https://r2.example\n"
    "[3] https://r3.example\n"
)

_PURE_NUMBERED_MARKDOWN = (
    "Fact one [1]. Fact two [2].\n\n"
    "## References\n"
    "[1] https://r1.example\n"
    "[2] https://r2.example\n"
)


def test_numbered_refs_parsed_and_reconciled_counts_are_distinct() -> None:
    """S96 resolves PD116c: numbered claims enter the annotation path."""
    _, annotations = extract_citations(_MIXED_MARKDOWN)
    assert len(annotations) == 4
    st: dict = {}
    fnode._extract_and_resolve_citations(
        _MIXED_MARKDOWN, sources_collected=[], synthesis_telemetry=st
    )
    ct = st["citation_telemetry"]
    assert ct["citations_extracted"] == 4
    assert ct["numbered_refs_parsed"] == 3
    assert ct["reconciliation"]["primary_annotations"] == 1


def test_pure_numbered_citations_extracted_equals_numbered_refs() -> None:
    st: dict = {}
    fnode._extract_and_resolve_citations(
        _PURE_NUMBERED_MARKDOWN, sources_collected=[], synthesis_telemetry=st
    )
    ct = st["citation_telemetry"]
    assert ct["citations_extracted"] == ct["numbered_refs_parsed"] == 2


def test_dual_count_never_feeds_the_gate() -> None:
    """citations_extracted=1 + numbered_refs_parsed=50 does NOT hard-fail — the
    zero-citation trigger keys ONLY on citations_extracted."""
    st = {
        "hit_max_steps": False,
        "self_terminated": True,
        "citation_telemetry": {"citations_extracted": 1, "numbered_refs_parsed": 50},
        "deliverable_coverage": {"shortfall_score": 0.0, "satisfied": ["x"], "missing": []},
        "liveness_telemetry": {"checked": 1, "live": 1, "dead": 0, "orphan": 0},
    }
    gate = evaluate_outcome_gate(st, report_word_count=5000)
    assert gate.outcome is MissionOutcome.COMPLETE


def test_mixed_reconciliation_replaces_the_exclusion_guard() -> None:
    """Count from the same extraction pass so forensic and live paths agree."""
    source = inspect.getsource(fnode._extract_and_resolve_citations)
    assert "telemetry=reconciliation" in source
    assert "_extract_numbered_fallback(" not in source


# ===========================================================================
# (c) Gate-boundary fixtures.
# ===========================================================================


def _zero_citation_telemetry() -> dict:
    return {
        "hit_max_steps": False,
        "self_terminated": True,
        "citation_telemetry": {"citations_extracted": 0},
        "deliverable_coverage": {"shortfall_score": 0.0, "satisfied": ["x"], "missing": []},
        "liveness_telemetry": {"checked": 0, "live": 0, "dead": 0, "orphan": 0},
    }


def test_sub_200_word_zero_citation_report_is_warning_not_clean() -> None:
    """A terse no-citation artifact stays usable but missing validation is loud."""
    assert MIN_REPORT_WORDS == 200
    under = evaluate_outcome_gate(_zero_citation_telemetry(), report_word_count=199)
    assert under.outcome is MissionOutcome.COMPLETE
    assert under.delivery_quality == DELIVERY_QUALITY_USABLE_WITH_WARNING

    at_floor = evaluate_outcome_gate(_zero_citation_telemetry(), report_word_count=200)
    assert at_floor.outcome is MissionOutcome.VALIDATION_FAILED
    assert at_floor.delivery_quality == DELIVERY_QUALITY_FAILED


def test_verified_references_zero_renders_on_live_zero_mission() -> None:
    """A degraded mission with liveness live=0 renders '**Verified references:** 0'."""
    synthesis_telemetry = {
        "citation_telemetry": {"distinct_urls_cited": 8, "citations_extracted": 8},
        "liveness_telemetry": {"checked": 8, "live": 0, "dead": 5, "orphan": 3},
    }
    meta = fnode._build_report_metadata(
        synthesis_result=_synthesis("# R"),
        synthesis_telemetry=synthesis_telemetry,
        citation_annotations=[],
        telemetry={},
    )
    assert meta.references == 0
    preamble = fnode._render_report_metadata_preamble(meta)
    assert "**Verified references:** 0" in preamble


def test_dead_rate_pre_quarantine_distinct_epoch_from_markers_quarantined() -> None:
    """Epoch pin (c3): liveness_telemetry (dead_rate numerator, PRE-quarantine)
    is produced and selected before quarantine, while markers_quarantined is
    written after quarantine — distinct finalize points."""
    src = inspect.getsource(fnode)
    i_liveness = src.index('pass_telemetry["liveness_telemetry"]')
    i_selected = src.index("synthesis_telemetry.update(selected_pass.telemetry)")
    i_quarantine = src.index("quarantine_dead_citations(")
    i_markers = src.index('synthesis_telemetry["markers_quarantined"]')
    assert i_liveness < i_selected < i_quarantine < i_markers, (
        "selected liveness_telemetry (PRE-quarantine dead count) must be "
        "canonical before quarantine; markers_quarantined follows it"
    )


# ===========================================================================
# (d) S82 invariant collected-test count pin.
# ===========================================================================


def test_s82_invariant_collected_count_is_79() -> None:
    """Pin the S82 invariant COLLECTED-test count at 79 via pytest --collect-only.

    79 = pytest-collected CASES (m01's 25 functions parametrize-expand to 29:
    29+11+11+13+9+6). 75 = the ``def test_`` FUNCTION count (25+11+11+13+9+6).
    These measure different things; neither is stale (decision #805). The
    ``N passed`` line reports the COLLECTED count, so that is what we pin.

    Re-pinned 70 -> 72 on 2026-08-13. The two added cases are m01's
    ``test_output_truncated_routes_to_validation_failed`` and
    ``test_untruncated_same_shape_is_not_failed_by_the_new_trigger``, landed by
    commit 35377e8 with the S88 runtime-budget detect-and-degrade fix
    (``finish_reason == "length"`` -> VALIDATION_FAILED). Genuine new coverage,
    so the pin moves rather than the tests.

    Re-pinned 72 -> 75 at S92 m07. The three cases prove that recovered
    generation errors require provider stop + citations, always carry a
    warning, and cannot complete on zero citations or an unknown finish.

    Re-pinned 75 -> 79 in the S92 m07 follow-up. The four cases prove that an
    error-retaining partial stays validation_failed, a completed selected
    bounded pass may recover with a warning, and neither an incoherent
    recovered flag nor an unselected replacement can false-COMPLETE.
    """
    files = [
        str(REPO_ROOT / "tests" / "unit" / f"test_s82_m0{i}_invariants.py")
        for i in range(1, 7)
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *files, "--collect-only", "-q"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    out = proc.stdout + proc.stderr
    per_file = [int(m) for m in re.findall(r"test_s82_m0\d_invariants\.py:\s*(\d+)", out)]
    if per_file:
        collected = sum(per_file)
    else:  # pragma: no cover — fallback for a standard pytest node-id reporter
        collected = len([ln for ln in out.splitlines() if "::" in ln and "test_s82_m0" in ln])
    assert collected == 79, (
        f"S82 invariant COLLECTED-test count drifted: expected 79, got {collected}. "
        "79 = collected cases (m01 25->29); 75 = def-test functions. Different "
        f"metrics, neither stale (decision #805).\n{out}"
    )
