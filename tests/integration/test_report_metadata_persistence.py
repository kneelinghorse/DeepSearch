"""S72 PD22 — round-trip persistence test for ``protocol.report_metadata``.

HD4-pattern test (carry-forward from S68.5 ``test_writer_telemetry_persistence``,
S69 ``test_factscore_persistence``, and S71
``test_sources_collected_persistence``): build a populated
:class:`MissionProtocolComplete` whose ``report_metadata`` block carries the
PD22 4 at-a-glance fields plus a forensic block, run the protocol through
every documented serialisation hop, and assert the runtime
:class:`ReportMetadata` shape survives.

Persistence chain exercised:

    MissionProtocolComplete(report_metadata=ReportMetadata(...)) →
    protocol.model_dump_json()      (save_protocol on disk)
    protocol.to_dict()              (worker → ResultWriter)
    _strip_null_bytes(payload)      (Postgres NUL-byte sanitisation)
    json.dumps(payload, default=str, ensure_ascii=False)  (JSONB column)
    json.loads(encoded)             (round-trip)

If a future change accidentally drops ``report_metadata`` from
:class:`deepsearch.report.protocol.MissionProtocolComplete`, mistypes a header
field, or reverts the to_dict() override that serialises the nested
:class:`ReportMetadata`, these tests fail loudly at CI rather than at the
next paid TraceLab dispatch — closing the user-perceived workflow-integrity
floor (the legacy ``execution_metadata`` zeros that the new ``report_metadata``
block supersedes).

The tests do not require a database, network, LLM, or smolagents — the
round-trip is pure Python: Pydantic → dict → strip-NUL → JSON string → dict.
"""

from __future__ import annotations

import json

from deepsearch.report.protocol import (
    REPORT_METADATA_SCHEMA_VERSION,
    MissionProtocolComplete,
    ReportMetadata,
)
from deepsearch.report.synthesis import ResearchStatement, Synthesis
from deepsearch.worker.writer import _strip_null_bytes


def _populated_metadata() -> ReportMetadata:
    """Mirrors the post-S64 codeagent_finalize_node builder output.

    Each forensic key here lines up with what
    ``deepsearch.agent.codeagent_finalize_node._build_report_metadata``
    populates: schema_version, build_hash, agent terminator flags,
    distinct-domain count, token usage, tool-call summary, plus the nested
    liveness / citation / critique telemetry envelopes.
    """

    return ReportMetadata(
        sources_collected=47,
        references=23,
        agent_steps_used=18,
        agent_steps_max=25,
        runtime_seconds=142.4,
        forensic={
            "schema_version": REPORT_METADATA_SCHEMA_VERSION,
            "build_hash": "f2e01dca2b7b1234567890abcdef0000",
            "self_terminated": True,
            "hit_max_steps": False,
            "distinct_domains": 12,
            "token_usage": {"input": 1000, "output": 2000, "total": 3000},
            "tool_call_summary": {
                "total_calls": 18,
                "by_tool": {"web_search": 6, "source_fetch": 6, "url_liveness": 6},
                "scratchpad_keys": ["draft_v1", "notes"],
            },
            "liveness_telemetry": {
                "checked": 23,
                "live": 21,
                "dead": 1,
                "unknown": 1,
                "runtime_seconds": 1.4,
            },
            "citation_telemetry": {
                "citations_extracted": 23,
                "orphan_marker_count": 0,
                "distinct_urls_cited": 23,
                "runtime_seconds": 0.001,
            },
            "critique_telemetry": {
                "claim_count": 23,
                "gap_notes_inserted": 0,
                "runtime_seconds": 3.2,
            },
        },
    )


def _empty_protocol() -> MissionProtocolComplete:
    return MissionProtocolComplete(
        mission_id="OODS-FIGMA-HOST-01-S72-EMPTY",
        title="S72 PD22 default-None round-trip",
        research_statement=ResearchStatement(
            topic="topic", objective="objective", scope="scope"
        ),
        synthesis=Synthesis(),
    )


def _populated_protocol() -> MissionProtocolComplete:
    return MissionProtocolComplete(
        mission_id="OODS-FIGMA-HOST-01-S72-POP",
        title="S72 PD22 populated round-trip",
        research_statement=ResearchStatement(
            topic="topic", objective="objective", scope="scope"
        ),
        synthesis=Synthesis(),
        report_metadata=_populated_metadata(),
    )


# ---------------------------------------------------------------------------
# 1. The field exists on the model with the expected default + type, even
#    when the caller never sets it explicitly. Default is None — legacy
#    callers that never assign report_metadata stay round-trip-stable.
# ---------------------------------------------------------------------------


def test_report_metadata_default_is_none() -> None:
    protocol = _empty_protocol()
    assert protocol.report_metadata is None, (
        "report_metadata must default to None so legacy / pre-S72 protocols "
        "round-trip identically. Only codeagent_finalize_node populates it."
    )


def test_report_metadata_default_omitted_from_to_dict() -> None:
    """to_dict uses exclude_none, so an unpopulated block must not appear."""

    payload = _empty_protocol().to_dict()
    assert "report_metadata" not in payload, (
        "Unpopulated report_metadata must NOT appear in the to_dict() payload "
        "(exclude_none semantics). Otherwise downstream readers cannot "
        "distinguish 'pre-S72 mission' from 'post-S72 mission with empty "
        "metadata'."
    )


# ---------------------------------------------------------------------------
# 2. HD4 round-trip — Pydantic to_dict() preserves the at-a-glance + forensic
#    structure when populated.
# ---------------------------------------------------------------------------


def test_populated_report_metadata_survives_to_dict() -> None:
    payload = _populated_protocol().to_dict()

    assert "report_metadata" in payload, (
        "Populated report_metadata must appear on the to_dict() payload — "
        "the worker writes this verbatim to result_protocol JSONB."
    )
    block = payload["report_metadata"]
    assert isinstance(block, dict)
    assert block["sources_collected"] == 47
    assert block["references"] == 23
    assert block["agent_steps_used"] == 18
    assert block["agent_steps_max"] == 25
    assert block["runtime_seconds"] == 142.4
    assert isinstance(block["forensic"], dict)
    assert block["forensic"]["schema_version"] == REPORT_METADATA_SCHEMA_VERSION
    assert block["forensic"]["build_hash"].startswith("f2e01dca2b7b")
    assert block["forensic"]["distinct_domains"] == 12
    assert block["forensic"]["liveness_telemetry"]["live"] == 21
    assert block["forensic"]["citation_telemetry"]["distinct_urls_cited"] == 23


# ---------------------------------------------------------------------------
# 3. JSONB-safety — the populated payload survives the worker writer's full
#    path: _strip_null_bytes → json.dumps(default=str, ensure_ascii=False) →
#    json.loads. Mirrors ResultWriter._serialize_jsonb verbatim.
# ---------------------------------------------------------------------------


def test_report_metadata_round_trips_through_worker_writer_serialiser() -> None:
    payload = _populated_protocol().to_dict()

    cleaned = _strip_null_bytes(payload)
    encoded = json.dumps(cleaned, default=str, ensure_ascii=False)
    decoded = json.loads(encoded)

    assert "report_metadata" in decoded
    block = decoded["report_metadata"]
    # All 4 PD22 header fields survive verbatim.
    assert block["sources_collected"] == 47
    assert block["references"] == 23
    assert block["agent_steps_used"] == 18
    assert block["agent_steps_max"] == 25
    assert block["runtime_seconds"] == 142.4
    # Forensic dict survives nested.
    forensic = block["forensic"]
    assert forensic["schema_version"] == REPORT_METADATA_SCHEMA_VERSION
    assert forensic["self_terminated"] is True
    assert forensic["hit_max_steps"] is False
    assert forensic["token_usage"] == {"input": 1000, "output": 2000, "total": 3000}
    assert forensic["tool_call_summary"]["total_calls"] == 18
    assert forensic["liveness_telemetry"]["dead"] == 1


def test_report_metadata_strip_null_bytes_handles_embedded_nuls() -> None:
    """Postgres TEXT/JSONB rejects U+0000. ``_strip_null_bytes`` must strip
    them recursively from forensic-dict string fields without dropping the
    block.
    """

    protocol = MissionProtocolComplete(
        mission_id="OODS-FIGMA-HOST-01-S72-NUL",
        title="S72 PD22 NUL-byte handling",
        research_statement=ResearchStatement(
            topic="topic", objective="objective", scope="scope"
        ),
        synthesis=Synthesis(),
        report_metadata=ReportMetadata(
            sources_collected=1,
            references=1,
            agent_steps_used=1,
            agent_steps_max=25,
            runtime_seconds=1.0,
            forensic={
                "schema_version": REPORT_METADATA_SCHEMA_VERSION,
                "build_hash": "abc\x00def\x00",
            },
        ),
    )

    payload = protocol.to_dict()
    cleaned = _strip_null_bytes(payload)
    encoded = json.dumps(cleaned, default=str, ensure_ascii=False)
    decoded = json.loads(encoded)

    forensic = decoded["report_metadata"]["forensic"]
    assert "\x00" not in forensic["build_hash"]
    assert forensic["build_hash"] == "abcdef"


# ---------------------------------------------------------------------------
# 4. save_protocol on-disk path — model_dump_json() (used by save_protocol)
#    serialises the field for the worker.main read-back path.
# ---------------------------------------------------------------------------


def test_report_metadata_round_trips_through_model_dump_json() -> None:
    """save_protocol uses ``protocol.model_dump_json()``; the worker reads the
    file back via ``json.loads`` before passing to the writer. This guard
    locks the on-disk leg of the chain — including the default-None case
    (where Pydantic emits ``"report_metadata": null`` rather than omitting).
    """

    protocol = _populated_protocol()

    encoded = protocol.model_dump_json()
    decoded = json.loads(encoded)

    assert "report_metadata" in decoded
    block = decoded["report_metadata"]
    expected_header_keys = {
        "sources_collected",
        "references",
        "agent_steps_used",
        "agent_steps_max",
        "runtime_seconds",
        "forensic",
    }
    assert expected_header_keys.issubset(block.keys()), (
        f"ReportMetadata must expose the 6 PD22 keys verbatim "
        f"({sorted(expected_header_keys)}); got {sorted(block.keys())}"
    )
    assert block["forensic"]["citation_telemetry"]["distinct_urls_cited"] == 23


def test_empty_protocol_model_dump_json_carries_null_report_metadata() -> None:
    """Pydantic's model_dump_json includes ``"report_metadata": null`` for the
    unpopulated case (no exclude_none). The on-disk JSON therefore signals
    "field exists, no value" so the worker read-back path can branch
    deterministically.
    """

    encoded = _empty_protocol().model_dump_json()
    decoded = json.loads(encoded)
    assert "report_metadata" in decoded
    assert decoded["report_metadata"] is None


# ---------------------------------------------------------------------------
# 5. S74 m01 PD32 Path 2 — numbered-references fallback HD4 round-trip.
#    Confirms the S73-WIRING-02 `References: 0` failure mode is closed end-
#    to-end: when the agent emits a numbered References list and zero
#    `[cite:URL]` markers, the extractor falls back, the citation telemetry
#    counts the resolved URLs, _build_report_metadata reads a non-zero
#    references count, and the truthful preamble surface persists through
#    the full worker writer chain.
# ---------------------------------------------------------------------------


def test_numbered_fallback_round_trip_carries_truthful_references_count() -> None:
    """The numbered-fallback path produces a truthful references count that
    survives the worker writer chain.

    The S73-WIRING-02 failure mode: agent emits a numbered References list
    but skips `[cite:URL]` markers, ``citation_telemetry.distinct_urls_cited``
    reads 0, ``ReportMetadata.references`` reads 0, and the user-visible
    preamble lies relative to the body.

    This guard exercises the production extractor + builder against a
    canonical numbered-only markdown, then runs the protocol through the
    same JSONB serialisation chain as ``_strip_null_bytes`` / ``json.dumps``
    / ``json.loads``. The preamble stays truthful.
    """

    from deepsearch.agent.codeagent_finalize_node import (
        _build_report_metadata,
        _extract_and_resolve_citations,
        _render_report_metadata_preamble,
    )
    from deepsearch.agent.codeagent_runner import SynthesisResult

    final_markdown = (
        "## Per-Entity Analyses\n\n"
        "### Cloudflare Workers\n\n"
        "Workers caps CPU at 30s on the paid tier [1]. Cold starts are "
        "reported under 5ms because of the isolate model [2].\n\n"
        "### Vercel Functions\n\n"
        "Vercel's Node runtime has a 60s execution ceiling on Pro [3].\n\n"
        "## References\n\n"
        "[1] Workers limits — https://developers.cloudflare.com/workers/platform/limits/\n"
        "[2] No-Container Computing — https://blog.cloudflare.com/cloud-computing-without-containers/\n"
        "[3] Vercel Functions — https://vercel.com/docs/functions/runtimes\n"
    )

    synthesis_telemetry: dict = {}
    sources_collected = [
        {"url": "https://developers.cloudflare.com/workers/platform/limits/", "title": "Workers limits"},
        {"url": "https://blog.cloudflare.com/cloud-computing-without-containers/", "title": "No-Container"},
        {"url": "https://vercel.com/docs/functions/runtimes", "title": "Vercel"},
    ]

    stripped_markdown, annotations = _extract_and_resolve_citations(
        final_markdown,
        sources_collected=sources_collected,
        synthesis_telemetry=synthesis_telemetry,
    )

    # Numbered fallback returns the markdown unchanged — the human-readable
    # numbered references stay visible to the reader.
    assert stripped_markdown == final_markdown
    assert len(annotations) == 3, (
        "Numbered fallback must resolve all 3 inline `[N]` markers via the "
        "References section. If this fails, the fallback path regressed."
    )
    distinct_urls = {a.url for a in annotations}
    assert len(distinct_urls) == 3

    # citation_telemetry surface is truthful — mirrors what protocol consumers
    # (and the build_report_metadata helper) read.
    citation_telemetry = synthesis_telemetry["citation_telemetry"]
    assert citation_telemetry["distinct_urls_cited"] == 3, (
        "synthesis_telemetry['citation_telemetry']['distinct_urls_cited'] "
        "must reflect the fallback-resolved count. S73-WIRING-02 hit 0 here "
        "and the preamble lied; the fallback closes that bug."
    )

    # Build the run-integrity block via the production helper. References
    # count must be non-zero — this is the truthful preamble surface.
    synthesis_result = SynthesisResult(
        output_markdown=final_markdown,
        step_count=14,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=212.0,
        token_usage={"input": 12000, "output": 5000, "total": 17000},
        tool_call_summary={"total_calls": 14, "by_tool": {"web_search": 8, "source_fetch": 6}},
        authority_tier_distribution={"high": 7, "medium": 4, "low": 2},
        pool_size=13,
        step_trace=[],
        sources_collected=sources_collected,
    )
    metadata = _build_report_metadata(
        synthesis_result=synthesis_result,
        synthesis_telemetry=synthesis_telemetry,
        citation_annotations=annotations,
        telemetry={"runtime_identity": {"build_hash": "deadbeefcafe1234"}},
    )
    assert metadata.references == 3, (
        "ReportMetadata.references must read the non-zero distinct_urls_cited "
        "count. If this fails, the numbered fallback didn't propagate to the "
        "preamble surface and S73-WIRING-02 stays open."
    )
    preamble = _render_report_metadata_preamble(metadata)
    # S82 m05: the header label is now "Verified references" (live-only); on
    # this no-liveness fallback path the value stays the distinct-cited count (3).
    assert "**Verified references:** 3" in preamble, (
        "The user-visible preamble must read `**Verified references:** 3` for "
        "the numbered-fallback case — proving the truthful surface aligns with "
        "the body's numbered References list."
    )

    # HD4 round-trip — survive the full worker writer chain.
    protocol = MissionProtocolComplete(
        mission_id="OODS-FIGMA-HOST-01-S74-NUMBERED-FALLBACK",
        title="S74 m01 numbered-fallback round-trip",
        research_statement=ResearchStatement(
            topic="topic", objective="objective", scope="scope"
        ),
        synthesis=Synthesis(),
        report_metadata=metadata,
    )
    payload = protocol.to_dict()
    cleaned = _strip_null_bytes(payload)
    encoded = json.dumps(cleaned, default=str, ensure_ascii=False)
    decoded = json.loads(encoded)

    assert decoded["report_metadata"]["references"] == 3, (
        "The numbered-fallback references count must survive the worker "
        "JSONB serialisation chain — otherwise the deployed preamble will "
        "still lie even though the in-memory build is truthful."
    )


def test_numbered_fallback_with_orphan_marker_records_orphan_count() -> None:
    """An orphan numbered marker (no entry in references) is dropped silently.

    The production helper still tracks orphan_marker_count for the resolved
    annotations whose URL is not in sources_collected. This guard pins the
    fallback's interaction with the orphan-counting logic.
    """

    from deepsearch.agent.codeagent_finalize_node import _extract_and_resolve_citations

    final_markdown = (
        "Real claim [1]. Orphan numbered marker [9].\n\n"
        "## References\n\n"
        "[1] Real — https://real.example\n"
    )
    synthesis_telemetry: dict = {}
    # sources_collected is empty so the resolved [1] URL counts as orphan.
    _, annotations = _extract_and_resolve_citations(
        final_markdown,
        sources_collected=[],
        synthesis_telemetry=synthesis_telemetry,
    )

    # [9] silently dropped. [1] resolves and is counted.
    assert len(annotations) == 1
    # Persisted citation identities use the same AnyHttpUrl canonical form as
    # TraceLab; a host-only URL therefore carries the explicit root path.
    assert annotations[0].url == "https://real.example/"

    citation_telemetry = synthesis_telemetry["citation_telemetry"]
    assert citation_telemetry["citations_extracted"] == 1
    assert citation_telemetry["distinct_urls_cited"] == 1
    # The resolved URL has no source-collected entry, so it is orphan-marked.
    assert citation_telemetry["orphan_marker_count"] == 1
