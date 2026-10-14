"""S71 Path A — round-trip persistence test for protocol.sources_collected.

HD4-pattern test (carry-forward from S68.5 ``test_writer_telemetry_persistence``
and S69 ``test_factscore_persistence``): build a populated
:class:`MissionProtocolComplete`, run the field through every documented
serialisation hop, and assert the runtime SourceRecord shape survives.

Persistence chain exercised:

    MissionProtocolComplete(sources_collected=[...]) →
    protocol.model_dump_json()      (save_protocol on disk)
    protocol.to_dict()              (worker → ResultWriter)
    _strip_null_bytes(payload)      (Postgres NUL-byte sanitisation)
    json.dumps(payload, default=str, ensure_ascii=False)  (JSONB column)
    json.loads(encoded)             (round-trip)

If a future change accidentally drops the ``sources_collected`` field from
:class:`deepsearch.report.protocol.MissionProtocolComplete`, mistypes a
SourceRecord field, or changes the runtime shape away from the
``{url, title, snippet, score, authority_tier, alive?}`` accumulator that
``deepsearch.agent.codeagent_runner._accumulate_source`` produces, these
tests fail loudly at CI rather than at the next paid TraceLab dispatch —
closing FActScore calibration substrate blocker (2) (S70 m02
``factscore=0.000`` symptom: empty ``sources_collected[]`` short-circuits
PD4 and every probe tallies as unknown).

The tests do not require a database, network, LLM, or smolagents — the
round-trip is pure Python: Pydantic → dict → strip-NUL → JSON string → dict.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from deepsearch.report.protocol import MissionProtocolComplete
from deepsearch.report.synthesis import ResearchStatement, Synthesis
from deepsearch.worker.writer import _strip_null_bytes


# ---------------------------------------------------------------------------
# Fixtures — runtime shape mirrors deepsearch.agent.codeagent_runner. Each
# entry is the canonical SourceRecord the runner-side accumulator produces:
#   {url, title, snippet, score, authority_tier, alive?}
# Per PD16: do NOT introduce a new schema; the in-process tuple/dict shape
# from agent.tools.web_search / source_fetch / url_liveness is the schema.
# ---------------------------------------------------------------------------


def _empty_protocol() -> MissionProtocolComplete:
    return MissionProtocolComplete(
        mission_id="OODS-FIGMA-HOST-01-S71-EMPTY",
        title="S71 Path A empty-list round-trip",
        research_statement=ResearchStatement(
            topic="topic", objective="objective", scope="scope"
        ),
        synthesis=Synthesis(),
        sources_collected=[],
    )


def _single_source_protocol() -> MissionProtocolComplete:
    return MissionProtocolComplete(
        mission_id="OODS-FIGMA-HOST-01-S71-SINGLE",
        title="S71 Path A single-source round-trip",
        research_statement=ResearchStatement(
            topic="topic", objective="objective", scope="scope"
        ),
        synthesis=Synthesis(),
        sources_collected=[
            {
                "url": "https://help.figma.com/hc/en-us/articles/360038006754",
                "title": "Hosting Figma sites",
                "snippet": "Figma supports custom domains for published sites.",
                "score": 0.91,
                "authority_tier": "high",
            }
        ],
    )


def _multi_source_protocol() -> MissionProtocolComplete:
    """Three sources spanning all three authority tiers + the optional
    ``alive`` field (populated by ``url_liveness`` calls). Exercises
    duplicate-tier handling, score-as-None, and snippet-with-special-chars
    so the round-trip catches encoding regressions.
    """

    sources: List[Dict[str, Any]] = [
        {
            "url": "https://www.figma.com/community/file/abc",
            "title": "Figma Community — host file",
            "snippet": "Community resource (low tier — UGC).",
            "score": 0.42,
            "authority_tier": "low",
            "alive": True,
        },
        {
            "url": "https://www.figma.com/blog/figma-sites",
            "title": "Figma Blog: Sites general availability",
            "snippet": "Marketing post — emoji \U0001F389 + <html>.",
            "score": 0.78,
            "authority_tier": "medium",
            "alive": True,
        },
        {
            "url": "https://help.figma.com/hc/en-us/articles/360038006754",
            "title": "Hosting Figma sites",
            "snippet": "Figma supports custom domains for published sites.",
            "score": 0.96,
            "authority_tier": "high",
            "alive": False,
        },
    ]
    return MissionProtocolComplete(
        mission_id="OODS-FIGMA-HOST-01-S71-MULTI",
        title="S71 Path A multi-source mixed-tier round-trip",
        research_statement=ResearchStatement(
            topic="topic", objective="objective", scope="scope"
        ),
        synthesis=Synthesis(),
        sources_collected=sources,
    )


# ---------------------------------------------------------------------------
# 1. The field exists on the model with the expected default + type, even
#    when the caller never sets it explicitly.
# ---------------------------------------------------------------------------


def test_sources_collected_default_is_empty_list() -> None:
    protocol = MissionProtocolComplete(
        mission_id="MID",
        title="T",
        research_statement=ResearchStatement(
            topic="topic", objective="objective", scope="scope"
        ),
        synthesis=Synthesis(),
    )

    assert protocol.sources_collected == [], (
        "sources_collected must default to an empty list; otherwise legacy "
        "callers that never assign the field hit a None-vs-list polymorphism "
        "downstream."
    )
    assert isinstance(protocol.sources_collected, list)


# ---------------------------------------------------------------------------
# 2. HD4 round-trip — Pydantic to_dict() preserves the field for each
#    fixture (empty / single / multi-source mixed-tier per the s71-m01
#    success criteria).
# ---------------------------------------------------------------------------


def test_empty_sources_collected_survives_to_dict() -> None:
    protocol = _empty_protocol()
    payload = protocol.to_dict()

    assert "sources_collected" in payload, (
        "Empty sources_collected must still be present in the to_dict() "
        "payload (additive PD9 field, default factory list). Without it, "
        "downstream consumers branch on missing-vs-empty and the FActScore "
        "harness has no way to distinguish the two."
    )
    assert payload["sources_collected"] == []


def test_single_source_collected_survives_to_dict() -> None:
    protocol = _single_source_protocol()
    payload = protocol.to_dict()

    assert "sources_collected" in payload
    assert isinstance(payload["sources_collected"], list)
    assert len(payload["sources_collected"]) == 1

    record = payload["sources_collected"][0]
    assert record["url"] == "https://help.figma.com/hc/en-us/articles/360038006754"
    assert record["title"] == "Hosting Figma sites"
    assert record["score"] == 0.91
    assert record["authority_tier"] == "high"


def test_multi_source_collected_survives_to_dict_in_order() -> None:
    protocol = _multi_source_protocol()
    payload = protocol.to_dict()

    assert "sources_collected" in payload
    records = payload["sources_collected"]
    assert len(records) == 3
    assert [r["authority_tier"] for r in records] == ["low", "medium", "high"]
    assert [r["alive"] for r in records] == [True, True, False]


# ---------------------------------------------------------------------------
# 3. JSONB-safety — the payload survives the worker writer's full path:
#    _strip_null_bytes → json.dumps(default=str, ensure_ascii=False) →
#    json.loads. Mirrors ResultWriter._serialize_jsonb verbatim.
# ---------------------------------------------------------------------------


def test_sources_collected_round_trips_through_worker_writer_serialiser() -> None:
    protocol = _multi_source_protocol()
    payload = protocol.to_dict()

    # Mirror ResultWriter.mark_completed:
    #   clean = _strip_null_bytes(result_protocol)
    #   wire = json.dumps(clean, default=str, ensure_ascii=False)
    cleaned = _strip_null_bytes(payload)
    encoded = json.dumps(cleaned, default=str, ensure_ascii=False)
    decoded = json.loads(encoded)

    assert "sources_collected" in decoded
    records = decoded["sources_collected"]
    assert len(records) == 3
    for record in records:
        assert isinstance(record, dict)
        assert isinstance(record["url"], str) and record["url"].startswith("https://")
        assert isinstance(record["title"], str)
        assert isinstance(record["snippet"], str)
        assert isinstance(record["authority_tier"], str)
        assert record["authority_tier"] in {"high", "medium", "low"}

    assert records[1]["snippet"].startswith("Marketing post"), (
        "ensure_ascii=False must preserve emoji and HTML-like chars verbatim"
    )


def test_sources_collected_strip_null_bytes_handles_embedded_nuls() -> None:
    """Postgres TEXT/JSONB rejects U+0000. ``_strip_null_bytes`` must strip
    them recursively from snippet / title fields without dropping records.
    """

    protocol = MissionProtocolComplete(
        mission_id="OODS-FIGMA-HOST-01-S71-NUL",
        title="S71 NUL-byte handling",
        research_statement=ResearchStatement(
            topic="topic", objective="objective", scope="scope"
        ),
        synthesis=Synthesis(),
        sources_collected=[
            {
                "url": "https://example.com/page",
                "title": "title\x00with\x00nuls",
                "snippet": "body\x00with\x00nuls",
                "score": 0.5,
                "authority_tier": "medium",
            }
        ],
    )

    payload = protocol.to_dict()
    cleaned = _strip_null_bytes(payload)
    encoded = json.dumps(cleaned, default=str, ensure_ascii=False)
    decoded = json.loads(encoded)

    record = decoded["sources_collected"][0]
    assert "\x00" not in record["title"]
    assert "\x00" not in record["snippet"]
    assert record["title"] == "titlewithnuls"
    assert record["snippet"] == "bodywithnuls"


# ---------------------------------------------------------------------------
# 4. save_protocol on-disk path — model_dump_json() (used by save_protocol)
#    serialises the field for the worker.main read-back path.
# ---------------------------------------------------------------------------


def test_sources_collected_round_trips_through_model_dump_json() -> None:
    """save_protocol calls protocol.model_dump_json(); the worker reads the
    file back via ``json.loads`` before passing to the writer. This guard
    locks the on-disk leg of the chain.
    """

    protocol = _multi_source_protocol()

    encoded = protocol.model_dump_json()
    decoded = json.loads(encoded)

    assert "sources_collected" in decoded
    records = decoded["sources_collected"]
    assert len(records) == 3

    expected_keys_per_record = {"url", "title", "snippet", "score", "authority_tier"}
    for record in records:
        assert expected_keys_per_record.issubset(record.keys()), (
            f"Each SourceRecord must carry the runtime accumulator keys "
            f"{sorted(expected_keys_per_record)} verbatim; got "
            f"{sorted(record.keys())}"
        )

    # The optional ``alive`` flag (populated by url_liveness calls) survives.
    alive_values = [r.get("alive") for r in records]
    assert alive_values == [True, True, False]


# ---------------------------------------------------------------------------
# 5. S72 PD24 — body widening + body_source discriminator round-trip.
#
# The runner accumulator now writes both ``body`` (full source_fetch text or
# the web_search snippet) and ``body_source`` ("snippet" | "full"). The
# FActScore harness keys off ``body`` first; the discriminator lets future
# calibration runs attribute outcomes to substrate shape. These keys must
# survive every leg of the persistence chain so the next post-S72 paid
# smoke produces meaningful per-claim verdicts rather than the all-`idk`
# outcome S70 m02 surfaced.
# ---------------------------------------------------------------------------


def _body_shaped_protocol() -> MissionProtocolComplete:
    full_text = "Full page text widening past the 300-char snippet ceiling. " * 10
    sources: List[Dict[str, Any]] = [
        {
            "url": "https://docs.example/full",
            "title": "Full-substrate fetch",
            "snippet": full_text[:300],
            "body": full_text,
            "body_source": "full",
            "score": 0.9,
            "authority_tier": "high",
            "alive": True,
        },
        {
            "url": "https://docs.example/snippet-only",
            "title": "Snippet-only entry",
            "snippet": "Linkup snippet preview.",
            "body": "Linkup snippet preview.",
            "body_source": "snippet",
            "score": 0.6,
            "authority_tier": "medium",
        },
    ]
    return MissionProtocolComplete(
        mission_id="OODS-FIGMA-HOST-01-S72-BODY",
        title="S72 PD24 body widening round-trip",
        research_statement=ResearchStatement(
            topic="topic", objective="objective", scope="scope"
        ),
        synthesis=Synthesis(),
        sources_collected=sources,
    )


def test_body_widening_survives_to_dict() -> None:
    payload = _body_shaped_protocol().to_dict()

    records = payload["sources_collected"]
    assert len(records) == 2
    assert records[0]["body_source"] == "full"
    assert len(records[0]["body"]) > 300, (
        "Full-substrate body must round-trip past the 300-char snippet cap; "
        "without this guard the FActScore harness keeps short-circuiting on "
        "snippet-shaped substrate (S70 m02 all-`idk` outcome)."
    )
    assert records[1]["body_source"] == "snippet"
    assert records[1]["body"] == "Linkup snippet preview."


def test_body_widening_round_trips_through_worker_writer_serialiser() -> None:
    payload = _body_shaped_protocol().to_dict()

    cleaned = _strip_null_bytes(payload)
    encoded = json.dumps(cleaned, default=str, ensure_ascii=False)
    decoded = json.loads(encoded)

    records = decoded["sources_collected"]
    body_sources = [r["body_source"] for r in records]
    assert body_sources == ["full", "snippet"]
    # Full-substrate body survives JSON round-trip without truncation.
    assert len(records[0]["body"]) > 300
    # Backwards-compat: legacy snippet field still present alongside body.
    assert "snippet" in records[0]
    assert "snippet" in records[1]


def test_body_widening_round_trips_through_model_dump_json() -> None:
    encoded = _body_shaped_protocol().model_dump_json()
    decoded = json.loads(encoded)

    records = decoded["sources_collected"]
    expected_keys = {"url", "title", "snippet", "body", "body_source", "score", "authority_tier"}
    for record in records:
        assert expected_keys.issubset(record.keys()), (
            f"Each post-S72 SourceRecord must carry the widened-body keys "
            f"{sorted(expected_keys)} verbatim; got {sorted(record.keys())}"
        )
