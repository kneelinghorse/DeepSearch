"""Cross-sprint quality telemetry with reconciled S87 quality facts.

A pure, flat per-mission record built from the finalize ``synthesis_telemetry``
envelope plus the shipped report's word count. The worker emits it as a sibling
of ``telemetry['final_outcome']`` so every persisted mission carries a calibration
datapoint, and an offline JSONL series kept with the project records
accumulates them so the delivery policy can be calibrated from real data
rather than guessed. S87 m04 adds the exact critique, recovery, retrieval,
context-pressure, and retrieval-state facts needed to distinguish a healthy zero
from missing telemetry.

The builder is deliberately a leaf pure-function (no smolagents / runner import)
so it can be exercised in isolation; ``_STEP_BUDGET`` mirrors
``codeagent_runner.DEFAULT_MAX_STEPS`` (the locked step budget) and a guard test
pins the two equal so the constant never drifts.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence
from urllib.parse import urlparse

from deepsearch.agent.url_utils import arxiv_identity

# Mirrors codeagent_runner.DEFAULT_MAX_STEPS (the locked smolagents step budget).
# Duplicated as a plain int to keep this module a heavy-import-free leaf; the
# drift guard lives in tests/unit/test_s83_m05_invariants.py.
_STEP_BUDGET = 40
# v3 (S90): context_truncation split into evidence-loss vs by-design packing —
# adds critique_cited_dropped + evidence_loss_occurred. Additive only; every v2
# field is retained with unchanged meaning. Bumped so PD136 calibration can
# SEGMENT the series on this boundary rather than re-baselining it: the
# delivery_quality distribution shifts here because validation_context_truncated
# stopped firing on uncited-tail eviction.
# v4 (S93): coverage separates artifact format from content and records
# unassessed explicitly. Preserve historical rows; segment calibration by version.
# v5 (S96): assessment scope and operational events are separate from defects.
# v6 (S102): a version 2 claim assessment adds unverified_contradicted,
# inconsistent and unknown_verdicts, read from each claim's critic verdict.
QUALITY_RECORD_SCHEMA_VERSION = 6
# The assessment verdicts a critic may report (critique rule 10).
_MODEL_VERDICTS = frozenset({
    "supported", "contradicted", "inconsistent", "unassessed", "declared_context",
})


def _as_int(value: Any) -> int:
    """Coerce to int, defaulting to 0 on None / non-numeric (never raises)."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def _as_bool(value: Any) -> bool:
    return bool(value)


def _nullable_int(value: Any) -> int | None:
    """Coerce a present exact count while preserving missing as unknown."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _critique_record(st: Mapping[str, Any]) -> Dict[str, Any]:
    critique_raw = st.get("critique_telemetry")
    critique = _mapping(critique_raw)
    summary_raw = critique.get("annotation_summary")
    summary = _mapping(summary_raw)
    pass_skipped = bool(st.get("critique_skipped"))
    error_text = str(critique.get("error") or "").strip()
    claim_count = _nullable_int(critique.get("claim_count"))
    legacy_gaps = _nullable_int(critique.get("gap_notes_inserted"))

    applied: int | None = None
    skipped: int | None = None
    dropped: int | None = None
    errors: int | None = None
    if error_text or pass_skipped:
        status = "error"
        errors = 1
    elif summary:
        applied = _nullable_int(summary.get("applied"))
        skipped = _nullable_int(summary.get("skipped"))
        dropped = _nullable_int(summary.get("dropped"))
        reconciled = summary.get("reconciled") is True
        if None in (applied, skipped, dropped) or not reconciled:
            status = "error"
            errors = 1
        else:
            status = "ok"
            errors = 0
    elif critique:
        status = "legacy_unstructured"
        errors = 0
    else:
        status = "unknown"

    applied_ratio: float | None = None
    if applied is not None and claim_count is not None:
        applied_ratio = round(applied / max(1, claim_count), 6)

    return {
        "status": status,
        "applied": applied,
        "skipped": skipped,
        "dropped": dropped,
        "errors": errors,
        "pass_skipped": int(pass_skipped),
        "claim_count": claim_count,
        "applied_ratio": applied_ratio,
        "legacy_gap_notes_inserted": legacy_gaps,
    }


def _retrieval_failures(st: Mapping[str, Any]) -> Dict[str, Any]:
    outcomes = _mapping(st.get("tool_outcomes"))
    raw_by_tool = outcomes.get("by_tool")
    if not isinstance(raw_by_tool, Mapping):
        return {"available": False, "total": None, "by_tool": {}}

    failures: Dict[str, int] = {}
    for tool_name, raw_entry in raw_by_tool.items():
        entry = _mapping(raw_entry)
        count = max(0, _as_int(entry.get("error")))
        if count:
            failures[str(tool_name)] = count
    return {
        "available": True,
        "total": sum(failures.values()),
        "by_tool": failures,
    }


def _recovery_record(st: Mapping[str, Any]) -> Dict[str, Any]:
    recovery = _mapping(st.get("recovery"))
    status = str(recovery.get("status") or "").strip() or "unknown"
    has_pass_count = bool(recovery) or "recovery_passes" in st
    passes = (
        max(
            0,
            _as_int(recovery.get("passes")),
            _as_int(st.get("recovery_passes")),
        )
        if has_pass_count
        else None
    )
    decision = _mapping(recovery.get("decision"))
    triggered_raw = decision.get("triggered")
    triggered = triggered_raw if isinstance(triggered_raw, bool) else None
    return {
        "passes": passes,
        "outcome": status,
        "selected_artifact": str(
            recovery.get("selected_artifact") or "unknown"
        ),
        "triggered": triggered,
    }


def _context_truncation(st: Mapping[str, Any]) -> Dict[str, Any]:
    critique = _mapping(st.get("critique_telemetry"))
    grounding = _mapping(critique.get("grounding"))
    recovery = _mapping(st.get("recovery"))
    recovery_prompt = _mapping(recovery.get("prompt"))
    # EVIDENCE LOSS — the critique judged the report against material it could
    # not fully see. A dropped CITED source means a claim was assessed without
    # its own evidence; a truncated draft or liveness table is a direct blind
    # spot. These are the conditions worth raising a quality warning on.
    evidence_loss = {
        "critique_cited_dropped": _as_int(grounding.get("cited_dropped")),
        "critique_draft_chars_dropped": _as_int(
            grounding.get("draft_chars_dropped")
        ),
        "critique_liveness_chars_dropped": _as_int(
            grounding.get("liveness_chars_dropped")
        ),
        "recovery_draft_chars_dropped": _as_int(
            recovery_prompt.get("draft_chars_dropped")
        ),
        "recovery_live_sources_dropped": _as_int(
            recovery_prompt.get("live_sources_dropped")
        ),
        "recovery_live_sources_truncated": _as_int(
            recovery_prompt.get("live_sources_truncated")
        ),
    }
    # BY-DESIGN PACKING — every source body is capped at
    # SOURCE_GROUNDING_MAX_CHARS and uncited overflow is evicted from the tail
    # of GROUNDING_TOTAL_BUDGET_CHARS. On a source-rich mission this is the
    # normal case (observed: 201 available -> 74 included, cited_dropped=0), so
    # it is recorded for auditing but never warned on.
    by_design = {
        "critique_sources_dropped": _as_int(grounding.get("sources_dropped")),
        "critique_sources_truncated": _as_int(
            grounding.get("sources_truncated")
        ),
    }
    fields = {**evidence_loss, **by_design}
    available = bool(grounding) or bool(recovery_prompt)
    return {
        "available": available,
        # Retained with v2 meaning (ANY truncation) so the historical series
        # stays comparable across the v3 boundary.
        "occurred": any(value > 0 for value in fields.values()) if available else None,
        "evidence_loss_occurred": (
            any(value > 0 for value in evidence_loss.values()) if available else None
        ),
        **fields,
    }


def _retrieval_state(st: Mapping[str, Any]) -> Dict[str, Any]:
    state = _mapping(st.get("retrieval_state"))
    return {
        "available": bool(state),
        "discovered": _as_int(state.get("discovered")),
        "fetched_readable": _as_int(state.get("fetched_readable")),
        "grounding_sources": _as_int(state.get("grounding_sources")),
        "fetch_attempts": _as_int(state.get("fetch_attempts")),
        "probes": _as_int(state.get("probes")),
    }


def _has_readable_body(source: Mapping[str, Any]) -> bool:
    body = source.get("body")
    chars = len(body) if isinstance(body, str) else _as_int(source.get("body_chars"))
    return (source.get("content_quality") == "readable_body"
            and source.get("body_source") == "source_fetch" and chars > 0
            and source.get("fetched_gone") is not True)


def _redirect_aliases(source: Mapping[str, Any]) -> list[str]:
    aliases = source.get("redirected_from")
    return [a for a in aliases if isinstance(a, str) and a] if isinstance(aliases, list) else []


def _host_and_path(url: str) -> tuple[str, str] | None:
    try:
        parsed = urlparse(url)
        return (parsed.hostname or "").lower(), parsed.path
    except ValueError:
        return None


def _read_aliases(source: Mapping[str, Any]) -> list[str]:
    """Redirect aliases that read this source's page (S100 m04).

    An alias exists only when that exact URL's own successful fetch resolved to
    this source (decision 1013), so the stored body is what the alias serves. A
    redirect that changed host or landed on the site root stays unread: it may
    be a login page or a homepage.
    """
    final = _host_and_path(str(source.get("url")))
    if final is None or not final[0] or final[1] in {"", "/"}:
        return []
    return [alias for alias in _redirect_aliases(source)
            if (_host_and_path(alias) or ("", ""))[0] == final[0]]


def readable_source_urls(sources: Sequence[Mapping[str, Any]]) -> set[str]:
    """URLs whose fetched body was read, as ``cited_without_readable_body`` counts.

    The exact source URL, plus its same-site redirect aliases (S100 m04).
    """
    return {url for source in sources if _has_readable_body(source)
            for url in (str(source.get("url")), *_read_aliases(source))}


def build_evidence_dependencies(
    cited_urls: Sequence[str], sources: Sequence[Mapping[str, Any]],
    tool_outcomes: Mapping[str, Any],
) -> Dict[str, Any]:
    """Measure final cited-source availability, never claim accuracy.

    Inputs use the pipeline's canonical URL boundary. Exact origins remain
    distinct; no www stripping or guessed redirect relationship is permitted.
    ``body_chars`` is the content-free retained-replay equivalent of body length.
    Collection membership alone (``missing_cited_sources``) matches arXiv URL
    forms of one paper (S98 m02) and URLs whose own successful fetch redirected
    to a collected source (S98 m06). Readability and truncation also count a
    readable source's same-site redirect aliases (S100 m04); fetch sets stay
    exact.
    """
    cited = set(cited_urls)
    collected = {
        arxiv_identity(url) or url
        for source in sources
        for url in [str(source.get("url")), *_redirect_aliases(source)]
    }
    readable: set[str] = set()
    truncated: set[str] = set()
    for source in sources:
        if _has_readable_body(source):
            urls = {str(source.get("url")), *_read_aliases(source)}
            readable |= urls
            if source.get("body_truncated") is True:
                truncated |= urls
    # Generic tool records retain network_requested; the strict Ledger schema
    # intentionally drops it and cannot distinguish controller/cache refusals.
    records = tool_outcomes.get("records")
    complete = isinstance(records, list) and tool_outcomes.get("records_truncated") == 0
    failed_fetches = {
        str(r.get("url")) for r in records or []
        if isinstance(r, Mapping) and r.get("tool") == "source_fetch"
        and r.get("status") == "error"
        and r.get("network_requested") is not False
    } if isinstance(records, list) else set()
    unresolved = cited - readable
    return {
        "version": 1, "available": True, "cited_urls": len(cited),
        "readable_cited_urls": len(cited & readable),
        "missing_cited_sources": sum(
            (arxiv_identity(url) or url) not in collected for url in cited
        ),
        "cited_without_readable_body": len(unresolved),
        "truncated_cited_bodies": len(cited & truncated),
        "unresolved_cited_fetch_failures": len(unresolved & failed_fetches),
        "recovered_cited_fetch_failures": len(cited & readable & failed_fetches),
        "uncited_fetch_failures": len(failed_fetches - cited),
        "fetch_failure_attribution_complete": complete,
        "local_fetch_refusals": sum(
            1 for r in records or [] if isinstance(r, Mapping)
            and r.get("tool") == "source_fetch" and r.get("status") == "error"
            and r.get("network_requested") is False
        ) if isinstance(records, list) else None,
    }


def _exact_count(value: Any) -> int | None:
    """A non-negative integer count, else ``None``; booleans are not counts."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _verdict_provenance(assessment: Mapping[str, Any]) -> Dict[str, int]:
    """Version 2 claims whose critic verdict is a finding or cannot be read.

    S102 m02. ``unverified_contradicted`` counts unassessed records the critic
    called contradicted plus dropped contradicted items; ``inconsistent`` every
    record or dropped item it called inconsistent, verified or not.
    ``unknown_verdicts`` counts everything not shown to carry a valid verdict:
    the unassessed records without one, records whose own verdict cannot be
    read, and the dropped items without one. A malformed record or drop count
    is unknown, never safe. Never raises.
    """

    raw_records = assessment.get("records")
    records: list[Any] | None = raw_records if isinstance(raw_records, list) else None
    rows = [r for r in records or [] if isinstance(r, Mapping)
            and isinstance(r.get("verdict"), str) and r["verdict"] in _MODEL_VERDICTS]
    malformed = len(records) - len(rows) if records is not None else 0
    unassessed_rows = [r for r in rows if r["verdict"] == "unassessed"]
    unverified_contradicted = sum(r.get("model_verdict") == "contradicted" for r in unassessed_rows)
    # Without readable records, the builder's own count of verified ones remains.
    inconsistent = (sum("inconsistent" in (r["verdict"], r.get("model_verdict")) for r in rows)
                    if records is not None else _as_int(assessment.get("inconsistent")))
    known = sum(isinstance(r.get("model_verdict"), str) and r["model_verdict"] in _MODEL_VERDICTS
                for r in unassessed_rows)
    # A malformed record may be one of the unassessed claims the count leaves
    # unexplained, so the two estimates overlap: take the larger.
    unknown = max(_as_int(assessment.get("unassessed")) - known, malformed, 0)
    emitted = _exact_count(assessment.get("emitted"))
    reported = _exact_count(assessment.get("reported"))
    dropped = (emitted - reported if emitted is not None and reported is not None
               else _as_int(assessment.get("dropped")))
    known_dropped = 0
    by_verdict = assessment.get("dropped_by_model_verdict")
    for verdict, raw in (by_verdict.items() if isinstance(by_verdict, Mapping) else ()):
        count = _exact_count(raw)
        if count is None or verdict not in _MODEL_VERDICTS:
            continue
        known_dropped += count
        unverified_contradicted += count if verdict == "contradicted" else 0
        inconsistent += count if verdict == "inconsistent" else 0
    return {
        "unverified_contradicted": unverified_contradicted,
        "inconsistent": inconsistent,
        "unknown_verdicts": unknown + max(0, dropped - known_dropped),
    }


def _claim_assessment(assessment: Mapping[str, Any]) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "version": assessment.get("version"),
        "status": assessment.get("status", "unknown"),
        "assessed": _nullable_int(assessment.get("assessed")),
        "supported": _nullable_int(assessment.get("supported")),
        "contradicted": _nullable_int(assessment.get("contradicted")),
        "unassessed": _nullable_int(assessment.get("unassessed")),
        "declared_context": _nullable_int(assessment.get("declared_context")),
        "report_claim_count": None,
        "coverage_ratio": None,
        "accuracy_pass": None,
    }
    if assessment.get("version") == 2:
        record.update(_verdict_provenance(assessment))
    elif assessment.get("provenance_error") is True:
        record["provenance_error"] = True
    return record


def _citation_surface(st: Mapping[str, Any]) -> Dict[str, Any]:
    audit = _mapping(st.get("citation_consistency"))
    issue_fields = (
        "unmapped_protocol_citations", "unmapped_quarantined_citations",
        "reference_number_collisions", "unresolved_reader_markers",
        "annotation_coordinate_drift", "missing_unverified_flags", "missing_withdrawn_flags",
        "protocol_only", "quarantined_missing",
    )
    issues = [key for key in issue_fields if _as_int(audit.get(key)) > 0]
    if audit.get("error"):
        issues.append("audit_error")
    # A legacy false flag without the detailed counters is not evidence of a
    # harmless bibliography surplus. Keep its uncertainty actionable.
    if audit.get("consistent") is False and not any(key in audit for key in issue_fields):
        issues.append("legacy_inconsistency_unknown")
    return {
        "available": bool(audit), "actionable": bool(issues), "issues": issues,
        "accounting_only": bool(audit) and audit.get("consistent") is False and not issues,
        "references_only": _nullable_int(audit.get("references_only")),
        "unused_reference_numbers": list(audit.get("unused_reference_numbers") or []),
        "unsupported_reference_rows": _nullable_int(audit.get("unsupported_reference_rows")),
    }


def build_quality_record(
    synthesis_telemetry: Mapping[str, Any] | None,
    report_word_count: Any,
) -> Dict[str, Any]:
    """Build the flat quality-series record from the finalize telemetry envelope.

    Pure function — no I/O. Counter fields with a meaningful legacy zero still
    coerce defensively; critique and other audit fields preserve missing data as
    explicit ``unknown``/``None`` so absent validation cannot look healthy. The
    liveness sub-dict reads the REAL
    ``LivenessTelemetry.to_dict`` key ``hallucinated_url_count`` and maps it to
    the record's ``hallucinated`` field. ``terminating_step`` is ``step_count``
    verbatim — NO ``-1`` (``codeagent_runner.step_count`` is a 1-based
    ``len(action_steps)``).

    Args:
        synthesis_telemetry: the finalize ``synthesis_telemetry`` envelope.
        report_word_count: the shipped report's word count, persisted onto
            ``synthesis_telemetry['report_word_count']`` at finalize so the
            worker's ``final_state`` carries it (a builder-in-isolation fixture
            would otherwise mask a live ``word_count=0`` — the S42.2-class trap).

    Returns:
        A JSON-safe dict with fixed nested quality-fact groups.
    """

    st: Mapping[str, Any] = (
        synthesis_telemetry if isinstance(synthesis_telemetry, Mapping) else {}
    )
    citation = st.get("citation_telemetry")
    citation = citation if isinstance(citation, Mapping) else {}
    liveness = st.get("liveness_telemetry")
    liveness = liveness if isinstance(liveness, Mapping) else {}

    coverage = _mapping(st.get("deliverable_coverage"))
    assessment = _mapping(_mapping(st.get("critique_telemetry")).get("assessment"))
    return {
        "verification": {
            "version": 1,
            "scope": "sampled" if assessment.get("version") in (1, 2)
            and _as_int(assessment.get("assessed")) > 0 else "unassessed_or_unknown",
            "report_wide_accuracy": "unknown",
            "report_claim_count": None,
            "coverage_ratio": None,
            "clean_means": "no_detected_material_issue",
        },
        "evidence_dependencies": dict(_mapping(st.get("evidence_dependencies")))
        or {"available": False},
        "citation_surface": _citation_surface(st),
        "claim_assessment": _claim_assessment(assessment),
        "coverage": {
            "semantics_version": coverage.get("semantics_version"),
            "assessment_status": coverage.get("assessment_status", "unknown"),
            "content_status": coverage.get("content_status", "unknown"),
            "shortfall_score": coverage.get("shortfall_score"),
            "artifact_requirements": coverage.get("artifact_requirements", []),
            "content_requirements": coverage.get("content_requirements", []),
        },
        "schema_version": QUALITY_RECORD_SCHEMA_VERSION,
        "word_count": _as_int(report_word_count),
        "citations_extracted": _as_int(citation.get("citations_extracted")),
        "distinct_urls": _as_int(citation.get("distinct_urls_cited")),
        "delivery_quality": str(st.get("delivery_quality") or ""),
        "markers_quarantined": _as_int(st.get("markers_quarantined")),
        "liveness": {
            "live": _as_int(liveness.get("live")),
            "dead": _as_int(liveness.get("dead")),
            "orphan": _as_int(liveness.get("orphan")),
            "unknown": _as_int(liveness.get("unknown")),
            # Map the real to_dict key -> the record's short name. A mis-mapped
            # key (e.g. reading 'hallucinated') would silently read 0.
            "hallucinated": _as_int(liveness.get("hallucinated_url_count")),
            "available": bool(liveness),
        },
        "critique": _critique_record(st),
        "retrieval_failures": _retrieval_failures(st),
        "recovery": _recovery_record(st),
        "context_truncation": _context_truncation(st),
        "retrieval_state": _retrieval_state(st),
        "self_terminated": _as_bool(st.get("self_terminated")),
        "hit_max_steps": _as_bool(st.get("hit_max_steps")),
        "terminating_step": _as_int(st.get("step_count")),
        "budget": int(_STEP_BUDGET),
    }


__all__ = [
    "QUALITY_RECORD_SCHEMA_VERSION",
    "build_quality_record",
    "build_evidence_dependencies",
    "readable_source_urls",
]
