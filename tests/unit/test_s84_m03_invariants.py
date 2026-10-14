"""S84 m03 invariants — writeback identity assertion (decision #823).

WHY THIS FILE EXISTS: CAT-R003 (execution e273cf44) persisted ANOTHER
mission's report — CAT-R004's header and body under CAT-R003's row, 0 of 47
citation URLs present in the persisted body, gates scored a 4,201-word draft
that never shipped, spans resolving into alien prose. The record could not
distinguish a writer race from a one-off; the deep dive converted that
uncertainty into this guard. Roadmap flip-trigger: wrong-document persistence
recurring AFTER this assertion == undiagnosed writer race == retire signal.

Pinned here:
1. The four synthetic mismatch fixtures mirroring the REAL CAT-R003 shapes
   (wrong body / zero URL overlap / stale draft / invalid spans) each land
   the mission BLOCKED with artifacts preserved and a machine-readable
   diagnostic in execution_metadata — never a silent ship.
2. Every check SKIPS when its signal is absent: legacy states (no wrapper
   H1, no citations, no result_identity envelope) persist byte-identically
   to pre-m03 behavior — the happy-path criterion.
3. The finalize node emits the result_identity envelope (scored-draft +
   report sha256, span_reference_len) nested in the already-declared
   synthesis_telemetry channel — no new AgentState channels.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.citation_liveness import LivenessTelemetry
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import CritiqueResult
from deepsearch.tracelab import TraceLabAuth
from deepsearch.worker.database import DatabaseConfig
from deepsearch.worker.lease import LeaseAttempt
from deepsearch.worker.main import DeepSearchWorker
from deepsearch.worker.poller import MissionRow
from deepsearch.worker.writer import (
    ExecutionMetadata,
    ResultIdentityError,
    assert_result_identity,
    identity_sha256,
    protocol_identity_sha256,
)

pytest.importorskip("aiohttp")


@pytest.fixture(autouse=True)
def _isolate_tracelab_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Identity tests must not inherit live service credentials from .env."""

    monkeypatch.setattr(
        "deepsearch.worker.main.TraceLabAuth",
        lambda: TraceLabAuth(
            base_url="",
            api_key="",
            username="",
            password="",
        ),
    )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 1. assert_result_identity — the graduated-check matrix (unit level).
# ---------------------------------------------------------------------------

_GOOD_BODY = (
    "# Research Report: test-mission-001 - Test Research Mission\n\n"
    "A claim backed by evidence.\n\n"
    "## References\n\n"
    "[1] Source — https://example.com/page\n"
)


def _protocol(urls: list[str], *, spans: list[tuple[int, int]] | None = None) -> dict:
    spans = spans or [(0, 5)] * len(urls)
    return {
        "citations": [
            {"type": "url_citation", "start_index": s, "end_index": e, "url": u}
            for u, (s, e) in zip(urls, spans)
        ]
    }


def _identity_envelope(
    report_text: str,
    *,
    span_ref_len: int | None = None,
    mission_id: str = "test-mission-001",
    title: str = "Test Research Mission",
) -> dict:
    return {
        "result_identity": {
            "algo": "sha256-utf8-nl",
            "mission_id": mission_id,
            "title": title,
            "report_sha256": identity_sha256(report_text),
            "report_len": len(report_text),
            "span_reference_len": (
                len(report_text) if span_ref_len is None else span_ref_len
            ),
        }
    }


def test_all_signals_absent_is_a_noop() -> None:
    """Legacy states carry no identity signals — the assertion must not fire
    (the pre-m03 byte-identical happy path)."""

    assert_result_identity(
        row_mission_id="test-mission-001",
        row_title="Test Research Mission",
        result_markdown="# Finalized report\n",
        result_protocol={"title": "Finalized protocol"},
        synthesis_telemetry=None,
        markdown_from_artifact=True,
    )


def test_happy_path_with_all_signals_present() -> None:
    assert_result_identity(
        row_mission_id="test-mission-001",
        row_title="Test Research Mission",
        result_markdown=_GOOD_BODY,
        result_protocol=_protocol(["https://example.com/page"]),
        synthesis_telemetry=_identity_envelope(_GOOD_BODY),
        markdown_from_artifact=True,
    )


def test_protocol_identity_accepts_canonical_json_equivalent() -> None:
    """Pretty-printed files and inline dicts share one structural identity."""

    protocol = {
        "mission_id": "test-mission-001",
        "citations": [],
        "created_at": "2026-08-29T12:00:00Z",
    }
    telemetry = _identity_envelope(_GOOD_BODY)
    telemetry["result_identity"]["protocol_sha256"] = protocol_identity_sha256(
        protocol
    )

    assert_result_identity(
        row_mission_id="test-mission-001",
        row_title="Test Research Mission",
        result_markdown=_GOOD_BODY,
        result_protocol=json.loads(json.dumps(protocol, indent=2)),
        synthesis_telemetry=telemetry,
        markdown_from_artifact=True,
    )


def test_protocol_mission_id_mismatch_is_content_free() -> None:
    protocol = {
        "mission_id": "another-mission",
        "citations": [],
        "source": "https://secret.example/private",
    }
    telemetry = _identity_envelope(_GOOD_BODY)
    telemetry["result_identity"]["protocol_sha256"] = protocol_identity_sha256(
        protocol
    )

    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=_GOOD_BODY,
            result_protocol=protocol,
            synthesis_telemetry=telemetry,
            markdown_from_artifact=True,
        )

    assert "protocol_mission_id_mismatch" in exc.value.diagnostic["violations"]
    rendered = json.dumps(exc.value.diagnostic, sort_keys=True)
    assert "another-mission" not in rendered
    assert "https://secret.example/private" not in rendered


def test_protocol_hash_mismatch_is_content_free() -> None:
    expected = {
        "mission_id": "test-mission-001",
        "citations": [],
        "summary": "expected protocol",
    }
    actual = {
        "mission_id": "test-mission-001",
        "citations": [],
        "summary": "https://secret.example/wrong-protocol",
    }
    telemetry = _identity_envelope(_GOOD_BODY)
    telemetry["result_identity"]["protocol_sha256"] = protocol_identity_sha256(
        expected
    )

    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=_GOOD_BODY,
            result_protocol=actual,
            synthesis_telemetry=telemetry,
            markdown_from_artifact=True,
        )

    assert "stale_protocol" in exc.value.diagnostic["violations"]
    rendered = json.dumps(exc.value.diagnostic, sort_keys=True)
    assert "expected protocol" not in rendered
    assert "https://secret.example/wrong-protocol" not in rendered


@pytest.mark.parametrize("surface", ["citation", "source"])
def test_protocol_identity_rejects_unsafe_urls_without_exposing_them(
    surface: str,
) -> None:
    """A self-consistent hash cannot admit a non-Ledger protocol URL."""

    secret_url = (
        "https://user:password@example.com/private"
        if surface == "citation"
        else "ftp://example.com/private"
    )
    protocol = {
        "mission_id": "test-mission-001",
        "citations": (
            [
                {
                    "type": "url_citation",
                    "start_index": 0,
                    "end_index": 5,
                    "url": secret_url,
                }
            ]
            if surface == "citation"
            else []
        ),
        "sources_collected": (
            [{"url": secret_url}] if surface == "source" else []
        ),
    }
    telemetry = _identity_envelope(_GOOD_BODY)
    telemetry["result_identity"]["protocol_sha256"] = protocol_identity_sha256(
        protocol
    )

    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=_GOOD_BODY,
            result_protocol=protocol,
            synthesis_telemetry=telemetry,
            markdown_from_artifact=True,
        )

    assert "unsafe_protocol_urls" in exc.value.diagnostic["violations"]
    assert exc.value.diagnostic["protocol_urls"] == {
        "invalid_citations": int(surface == "citation"),
        "invalid_sources": int(surface == "source"),
    }
    assert secret_url not in json.dumps(exc.value.diagnostic, sort_keys=True)


def test_protocol_hash_envelope_requires_mission_id() -> None:
    protocol = {"citations": [], "title": "Missing identity"}
    telemetry = _identity_envelope(_GOOD_BODY)
    telemetry["result_identity"]["protocol_sha256"] = protocol_identity_sha256(
        protocol
    )

    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=_GOOD_BODY,
            result_protocol=protocol,
            synthesis_telemetry=telemetry,
            markdown_from_artifact=True,
        )

    assert "protocol_mission_id_missing" in exc.value.diagnostic["violations"]


def test_wrong_body_mission_id_mismatch_raises() -> None:
    """CAT-R003 shape 1: another mission's header under this row."""

    wrong = _GOOD_BODY.replace(
        "test-mission-001 - Test Research Mission",
        "CAT-R004 - AI Search & Discovery in Ecommerce 2024-2026",
    )
    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=wrong,
            result_protocol=None,
            synthesis_telemetry=None,
            markdown_from_artifact=True,
        )
    assert "mission_id_mismatch" in exc.value.diagnostic["violations"]
    assert exc.value.diagnostic["header"]["parsed_mission_id"] == "CAT-R004"


def test_title_mismatch_fires_only_via_envelope() -> None:
    """ADVERSARIAL-REVIEW FIX: the H1 title derives from the generation-time
    context title, NOT row.title — comparing row.title against the H1
    false-fired on every context/row divergence. Title identity is now judged
    against the envelope the same finalize run recorded."""

    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=_GOOD_BODY,
            result_protocol=None,
            synthesis_telemetry=_identity_envelope(
                _GOOD_BODY, title="A Completely Different Title"
            ),
            markdown_from_artifact=True,
        )
    assert "title_mismatch" in exc.value.diagnostic["violations"]

    # The wrapper H1 title is normalize_mission_title(title, id) — an
    # envelope title carrying a duplicated id prefix must compare EQUAL.
    assert_result_identity(
        row_mission_id="test-mission-001",
        row_title="Test Research Mission",
        result_markdown=_GOOD_BODY,
        result_protocol=None,
        synthesis_telemetry=_identity_envelope(
            _GOOD_BODY, title="test-mission-001 - Test Research Mission"
        ),
        markdown_from_artifact=True,
    )


def test_row_title_divergence_without_envelope_does_not_fire() -> None:
    """Legacy states (no envelope): row.title can legitimately differ from
    the generation-time context title — the title must NOT be judged."""

    assert_result_identity(
        row_mission_id="test-mission-001",
        row_title="A Completely Different Row Title",
        result_markdown=_GOOD_BODY,
        result_protocol=None,
        synthesis_telemetry=None,
        markdown_from_artifact=True,
    )


def test_envelope_mission_id_mismatch_fires_without_h1() -> None:
    """Cross-mission contamination at the STATE level: the envelope was
    recorded by a different mission's finalize run than the claimed row."""

    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown="no header here\n",
            result_protocol=None,
            synthesis_telemetry=_identity_envelope(
                "no header here\n", mission_id="CAT-R004", title="Other"
            ),
            markdown_from_artifact=True,
        )
    assert "mission_id_mismatch" in exc.value.diagnostic["violations"]
    assert exc.value.diagnostic["header"]["envelope_mission_id"] == "CAT-R004"


def test_crlf_translation_cannot_hide_report_length_drift() -> None:
    """A normalized hash cannot excuse coordinates assigned to CRLF text."""

    original = _GOOD_BODY.replace("\n", "\r\n") + "quoted line\r\n"
    translated = original.replace("\r\n", "\n")  # what read_text hands back
    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=translated,
            result_protocol=None,
            synthesis_telemetry=_identity_envelope(original),
            markdown_from_artifact=True,
        )

    assert "stale_draft" not in exc.value.diagnostic["violations"]
    assert "report_length_mismatch" in exc.value.diagnostic["violations"]
    assert "span_reference_length_mismatch" in exc.value.diagnostic["violations"]


def test_noncanonical_artifact_is_rejected_even_when_identity_matches() -> None:
    report = _GOOD_BODY.replace("claim", "claim\x00", 1)

    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=report,
            result_protocol=None,
            synthesis_telemetry=_identity_envelope(report),
            markdown_from_artifact=True,
        )

    assert "noncanonical_report" in exc.value.diagnostic["violations"]


def test_marker_only_report_without_body_urls_skips_overlap() -> None:
    """ADVERSARIAL-REVIEW FIX: a correctly-cited report whose [cite:] markers
    were stripped and that has no References section contains zero body URLs
    — the overlap check must skip, not block."""

    body = (
        "# Research Report: test-mission-001 - Test Research Mission\n\n"
        "A well-cited claim whose markers were stripped at extraction.\n"
    )
    assert_result_identity(
        row_mission_id="test-mission-001",
        row_title="Test Research Mission",
        result_markdown=body,
        result_protocol=_protocol(["https://example.com/page"]),
        synthesis_telemetry=None,
        markdown_from_artifact=True,
    )


def test_hash_functions_mirror_each_other() -> None:
    """The finalize-side local mirror and the worker-side canonical hash MUST
    agree (import direction forbids sharing the function)."""

    sample = "line one\r\nline two\rline three\nplain"
    assert fnode._identity_sha256(sample) == identity_sha256(sample)


def test_zero_citation_url_overlap_raises() -> None:
    """CAT-R003 shape 2: 0 of N protocol citation URLs appear in the body."""

    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=_GOOD_BODY,
            result_protocol=_protocol(
                ["https://amazon.science/autoknow", "https://neo4j.com/blog/x"]
            ),
            synthesis_telemetry=None,
            markdown_from_artifact=True,
        )
    assert "zero_citation_url_overlap" in exc.value.diagnostic["violations"]
    assert exc.value.diagnostic["citation_overlap"]["found_in_body"] == 0

    # One overlapping URL is enough — partial overlap must NOT fire.
    assert_result_identity(
        row_mission_id="test-mission-001",
        row_title="Test Research Mission",
        result_markdown=_GOOD_BODY,
        result_protocol=_protocol(
            ["https://example.com/page", "https://amazon.science/autoknow"]
        ),
        synthesis_telemetry=None,
        markdown_from_artifact=True,
    )


def test_citation_overlap_uses_tracelab_canonical_url_identity() -> None:
    """Path/case/default-port variants must not false-block after finalize."""

    body = (
        "# Research Report: test-mission-001 - Test Research Mission\n\n"
        "A claim backed by evidence.\n\n"
        "## References\n\n"
        "[1] Source — https://Example.COM:443/a/../canonical\n"
    )

    assert_result_identity(
        row_mission_id="test-mission-001",
        row_title="Test Research Mission",
        result_markdown=body,
        result_protocol=_protocol(["https://example.com/canonical"]),
        synthesis_telemetry=None,
        markdown_from_artifact=True,
    )


@pytest.mark.parametrize(
    "url",
    (
        "https://example.com/wiki/Foo_(bar)",
        "https://example.com/emphasis!",
        "https://example.com/empty-query?",
        "https://example.com/wildcard*",
    ),
)
def test_citation_overlap_preserves_valid_terminal_url_punctuation(url: str) -> None:
    """Canonical overlap must retain legal suffixes before prose trimming."""

    body = (
        "# Research Report: test-mission-001 - Test Research Mission\n\n"
        f"## References\n\n[1] Exact source — {url}\n"
    )

    assert_result_identity(
        row_mission_id="test-mission-001",
        row_title="Test Research Mission",
        result_markdown=body,
        result_protocol=_protocol([url]),
        synthesis_telemetry=None,
        markdown_from_artifact=True,
    )


@pytest.mark.parametrize(
    ("rendered", "url"),
    (
        ("**https://example.com/page**", "https://example.com/page"),
        ("**https://example.com/page**,", "https://example.com/page"),
        ("`https://example.com/page`", "https://example.com/page"),
        ("~~https://example.com/page~~", "https://example.com/page"),
    ),
)
def test_citation_overlap_resolves_expected_markdown_wrapper_suffixes(
    rendered: str,
    url: str,
) -> None:
    """Wrapper delimiters are removable only against protocol identity."""

    body = (
        "# Research Report: test-mission-001 - Test Research Mission\n\n"
        f"## References\n\n[1] Wrapped source — {rendered}\n"
    )

    assert_result_identity(
        row_mission_id="test-mission-001",
        row_title="Test Research Mission",
        result_markdown=body,
        result_protocol=_protocol([url]),
        synthesis_telemetry=None,
        markdown_from_artifact=True,
    )


@pytest.mark.parametrize(
    ("rendered", "protocol_url"),
    (
        ("https://example.com/wildcard*", "https://example.com/wildcard"),
        ("https://example.com/emphasis!", "https://example.com/emphasis"),
        ("https://example.com/empty-query?", "https://example.com/empty-query"),
        ("https://example.com/page**", "https://example.com/page"),
        ("https://example.com/page~~", "https://example.com/page"),
    ),
)
def test_citation_overlap_does_not_infer_unmatched_suffix_wrappers(
    rendered: str,
    protocol_url: str,
) -> None:
    """Distinct legal URL suffixes cannot be stripped without source syntax."""

    body = (
        "# Research Report: test-mission-001 - Test Research Mission\n\n"
        f"## References\n\n[1] Plain source — {rendered}\n"
    )

    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=body,
            result_protocol=_protocol([protocol_url]),
            synthesis_telemetry=None,
            markdown_from_artifact=True,
        )

    assert "zero_citation_url_overlap" in exc.value.diagnostic["violations"]


def test_citation_overlap_trims_only_unmatched_markdown_closing_parenthesis() -> None:
    """A Markdown delimiter is removable without truncating the URL's own paren."""

    url = "https://example.com/wiki/Foo_(bar)"
    body = (
        "# Research Report: test-mission-001 - Test Research Mission\n\n"
        f"## References\n\n[1] [Exact source]({url})\n"
    )

    assert_result_identity(
        row_mission_id="test-mission-001",
        row_title="Test Research Mission",
        result_markdown=body,
        result_protocol=_protocol([url]),
        synthesis_telemetry=None,
        markdown_from_artifact=True,
    )


def test_stale_draft_hash_mismatch_raises() -> None:
    """CAT-R003 shape 3: the gates scored a document that is not the one
    being persisted."""

    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=_GOOD_BODY,
            result_protocol=None,
            synthesis_telemetry=_identity_envelope("a different document entirely"),
            markdown_from_artifact=True,
        )
    assert "stale_draft" in exc.value.diagnostic["violations"]
    assert exc.value.diagnostic["draft_hash"]["expected"] != (
        exc.value.diagnostic["draft_hash"]["actual"]
    )


def test_exact_report_and_span_reference_lengths_are_required() -> None:
    telemetry = _identity_envelope(_GOOD_BODY)
    telemetry["result_identity"]["report_len"] += 1
    telemetry["result_identity"]["span_reference_len"] += 2

    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=_GOOD_BODY,
            result_protocol=None,
            synthesis_telemetry=telemetry,
            markdown_from_artifact=True,
        )

    assert {
        "report_length_mismatch",
        "span_reference_length_mismatch",
    } <= set(exc.value.diagnostic["violations"])


def test_regenerated_fallback_with_identity_envelope_raises() -> None:
    """The finalize node hashed an artifact but the worker could not read it:
    shipping regenerated template bytes would be a silent identity loss."""

    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown="# regenerated from state\n",
            result_protocol=None,
            synthesis_telemetry=_identity_envelope(_GOOD_BODY),
            markdown_from_artifact=False,
        )
    assert "emitted_artifact_missing" in exc.value.diagnostic["violations"]


@pytest.mark.parametrize(
    "spans",
    [
        [(4102, 4208)],  # beyond span_reference_len — the real CAT-R003 shape
        [(5, 2)],  # end < start (only reachable via corrupted serialized JSON)
        [(-1, 4)],  # negative start
        [(5, 5)],  # empty slices cannot identify a cited claim
    ],
)
def test_invalid_spans_raise(spans) -> None:
    """CAT-R003 shape 4: spans that cannot resolve against the reference."""

    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=_GOOD_BODY,
            result_protocol=_protocol(["https://example.com/page"], spans=spans),
            synthesis_telemetry=_identity_envelope(_GOOD_BODY, span_ref_len=100),
            markdown_from_artifact=True,
        )
    assert "invalid_spans" in exc.value.diagnostic["violations"]


def test_spans_without_envelope_must_still_index_actual_markdown() -> None:
    """Persisted protocol coordinates always address persisted markdown."""

    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=_GOOD_BODY,
            result_protocol=_protocol(
                ["https://example.com/page"], spans=[(4102, 4208)]
            ),
            synthesis_telemetry=None,
            markdown_from_artifact=True,
        )

    assert "invalid_spans" in exc.value.diagnostic["violations"]


def test_citation_slice_must_be_nonempty_and_exactly_trimmed() -> None:
    claim_start = _GOOD_BODY.index("A claim")

    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=_GOOD_BODY,
            result_protocol=_protocol(
                ["https://example.com/page"],
                spans=[(claim_start - 1, claim_start + len("A claim"))],
            ),
            synthesis_telemetry=_identity_envelope(_GOOD_BODY),
            markdown_from_artifact=True,
        )

    assert "invalid_spans" in exc.value.diagnostic["violations"]


def test_citation_slice_cannot_exceed_tracelab_exact_claim_limit() -> None:
    report = "x" * 20_001

    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=report,
            result_protocol=_protocol(
                ["https://example.com/page"], spans=[(0, len(report))]
            ),
            synthesis_telemetry=_identity_envelope(report),
            markdown_from_artifact=True,
        )

    assert "invalid_spans" in exc.value.diagnostic["violations"]


def test_multiple_violations_all_collected() -> None:
    wrong = _GOOD_BODY.replace("test-mission-001", "CAT-R004")
    with pytest.raises(ResultIdentityError) as exc:
        assert_result_identity(
            row_mission_id="test-mission-001",
            row_title="Test Research Mission",
            result_markdown=wrong,
            result_protocol=_protocol(["https://absent.example/x"], spans=[(5, 2)]),
            synthesis_telemetry=_identity_envelope("different text"),
            markdown_from_artifact=True,
        )
    violations = exc.value.diagnostic["violations"]
    assert {
        "mission_id_mismatch",
        "zero_citation_url_overlap",
        "stale_draft",
        "invalid_spans",
    } <= set(violations)


# ---------------------------------------------------------------------------
# 2. Worker end-to-end: violations land BLOCKED with artifacts + diagnostic.
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_config():
    return DatabaseConfig(
        url="postgresql://test:test@localhost:5432/test",
        min_pool_size=1,
        max_pool_size=5,
    )


@pytest.fixture
def sample_mission_row():
    return MissionRow(
        id="test-mission-001",
        mission_id="test-mission-001",
        title="Test Research Mission",
        objective="Research objective 1",
        context={"background": "Test background", "focus": "Test focus"},
        status="queued",
        queued_at=datetime.now(timezone.utc),
        lease=LeaseAttempt(
            owner="worker-test",
            token="opaque-test-token",
            leased_at=datetime.now(timezone.utc),
            heartbeat_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc),
            attempt_count=1,
        ),
        success_criteria=["Research objective 2"],
        research_phases={"max_loops": 3},
        research_depth="baseline",
    )


def _final_state(report_path: str, protocol_path: str, **extra) -> dict:
    state = {
        "mission_id": "test-mission-001",
        "total_tokens": 1000,
        "report_path": report_path,
        "protocol_path": protocol_path,
        "routing_reason": "converged_with_coverage",
        "mission_outcome": "complete",
        "sources_found": [],
    }
    state.update(extra)
    return state


async def _run_worker(worker, row, final_state):
    with patch.object(worker, "_execute_research", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = final_state
        with patch("deepsearch.worker.main.build_execution_metadata") as mock_meta:
            mock_meta.return_value = ExecutionMetadata(
                total_tokens=0, duration_seconds=0.0
            )
            await worker._handle_mission(row)


def _write_artifacts(tmp_path, body: str, protocol: dict):
    report_path = tmp_path / "report.md"
    protocol_path = tmp_path / "protocol.json"
    report_path.write_text(body, encoding="utf-8")
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    return str(report_path), str(protocol_path)


def test_worker_blocks_wrong_body_with_diagnostic(
    mock_db_config, sample_mission_row, tmp_path
) -> None:
    """The exact CAT-R003 incident: another mission's report under this row
    lands BLOCKED — artifacts preserved, diagnostic in execution_metadata."""

    wrong_body = (
        "# Research Report: CAT-R004 - AI Search & Discovery in Ecommerce 2024-2026\n\n"
        "Rufus and agentic commerce content.\n"
    )
    rp, pp = _write_artifacts(tmp_path, wrong_body, {"title": "x"})
    worker = DeepSearchWorker(db_config=mock_db_config)
    mock_writer = AsyncMock()
    worker._writer = mock_writer

    asyncio.run(
        _run_worker(worker, sample_mission_row, _final_state(rp, pp))
    )

    mock_writer.mark_completed.assert_not_called()
    mock_writer.mark_failed.assert_called_once()
    kwargs = mock_writer.mark_failed.call_args.kwargs
    assert kwargs["outcome"].value == "blocked"
    assert kwargs["error_message"] == (
        '{"category":"persistence","exception_class":"ResultIdentityError"}'
    )
    assert kwargs["result_markdown"] == wrong_body  # suspect artifact preserved
    assert kwargs["execution_metadata"].identity_violation["violations"] == [
        "mission_id_mismatch"
    ]
    assert kwargs["execution_metadata"].lease_attempt["contract_version"] == (
        "tracelab-missions-lease-v2"
    )
    assert kwargs["execution_metadata"].lease_attempt["terminal_reset"] == {
        "deepsearch_lease_token": "cleared_atomically",
        "deepsearch_lease_expires_at": "cleared_atomically",
    }
    assert worker._stats["missions_failed"] == 1


def test_worker_blocks_stale_draft(mock_db_config, sample_mission_row, tmp_path) -> None:
    good_body = (
        "# Research Report: test-mission-001 - Test Research Mission\n\nContent.\n"
    )
    rp, pp = _write_artifacts(tmp_path, good_body, {"title": "x"})
    state = _final_state(
        rp,
        pp,
        synthesis_telemetry={
            "result_identity": {
                "algo": "sha256-utf8",
                "report_sha256": _sha("the draft the gates actually scored"),
                "report_len": 33,
                "span_reference_len": 100,
            }
        },
    )
    worker = DeepSearchWorker(db_config=mock_db_config)
    mock_writer = AsyncMock()
    worker._writer = mock_writer

    asyncio.run(_run_worker(worker, sample_mission_row, state))

    mock_writer.mark_failed.assert_called_once()
    kwargs = mock_writer.mark_failed.call_args.kwargs
    assert kwargs["outcome"].value == "blocked"
    assert "stale_draft" in kwargs["execution_metadata"].identity_violation["violations"]


def test_worker_blocks_zero_overlap_and_invalid_spans(
    mock_db_config, sample_mission_row, tmp_path
) -> None:
    # The body renders URLs — just none of the protocol's citation URLs
    # (the CAT-R003 shape: 0 of 47 overlapped while the body cited freely).
    good_body = (
        "# Research Report: test-mission-001 - Test Research Mission\n\n"
        "Content.\n\n## References\n\n[1] Other — https://unrelated.example/x\n"
    )
    protocol = {
        "citations": [
            {
                "type": "url_citation",
                "start_index": 4102,
                "end_index": 4208,
                "url": "https://amazon.science/autoknow",
            }
        ]
    }
    rp, pp = _write_artifacts(tmp_path, good_body, protocol)
    state = _final_state(
        rp,
        pp,
        synthesis_telemetry={
            "result_identity": {
                "algo": "sha256-utf8",
                "report_sha256": _sha(good_body),
                "report_len": len(good_body),
                "span_reference_len": 50,
            }
        },
    )
    worker = DeepSearchWorker(db_config=mock_db_config)
    mock_writer = AsyncMock()
    worker._writer = mock_writer

    asyncio.run(_run_worker(worker, sample_mission_row, state))

    kwargs = mock_writer.mark_failed.call_args.kwargs
    violations = kwargs["execution_metadata"].identity_violation["violations"]
    assert "zero_citation_url_overlap" in violations
    assert "invalid_spans" in violations


def test_worker_happy_path_with_identity_signals_completes(
    mock_db_config, sample_mission_row, tmp_path
) -> None:
    """Full identity signals, all consistent → mark_completed with the
    byte-equal artifact (the no-behavior-change criterion under the guard)."""

    good_body = (
        "# Research Report: test-mission-001 - Test Research Mission\n\n"
        "Claim. \n\n## References\n\n[1] S — https://example.com/page\n"
    )
    protocol = {
        "citations": [
            {
                "type": "url_citation",
                "start_index": 0,
                "end_index": 5,
                "url": "https://example.com/page",
            }
        ]
    }
    rp, pp = _write_artifacts(tmp_path, good_body, protocol)
    state = _final_state(
        rp,
        pp,
        synthesis_telemetry={
            "result_identity": {
                "algo": "sha256-utf8",
                "report_sha256": _sha(good_body),
                "report_len": len(good_body),
                "span_reference_len": len(good_body),
            }
        },
    )
    worker = DeepSearchWorker(db_config=mock_db_config)
    mock_writer = AsyncMock()
    worker._writer = mock_writer

    asyncio.run(_run_worker(worker, sample_mission_row, state))

    mock_writer.mark_failed.assert_not_called()
    mock_writer.mark_completed.assert_called_once()
    assert mock_writer.mark_completed.call_args.kwargs["result_markdown"] == good_body


# ---------------------------------------------------------------------------
# 3. Finalize-side envelope pins.
# ---------------------------------------------------------------------------


def _state() -> dict:
    return {
        "mission_id": "S84-M03-E2E",
        "mission_title": "Identity Mission",
        "mission_objectives": ["c1"],
        "mission_context": {
            "title": "Identity Mission",
            "objective": "verify identity envelope",
            "deliverables": ["Comparison Matrix"],
        },
        "sources_found": [],
        "telemetry": {},
    }


def test_finalize_emits_result_identity_envelope(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))
    captured: dict = {}

    def _capture_report(content, **kwargs):
        captured["report"] = content
        return str(tmp_path / "report.md")

    synthesis = SynthesisResult(
        output_markdown=(
            "# Report\n\nClaim [cite:https://live.example/c] with detail.\n\n"
            "## References\n\n[1] C — https://live.example/c\n"
        ),
        step_count=3,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=5.0,
        token_usage={"input": 10, "output": 10, "total": 20},
        tool_call_summary={"total_calls": 1},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0,
        step_trace=[{"step_index": 0, "type": "ActionStep"}],
        sources_collected=[
            {"url": "https://live.example/c", "title": "C", "snippet": "z"}
        ],
        error=None,
    )
    critique = CritiqueResult(
        output_markdown="",
        claim_count=0,
        gap_notes_inserted=0,
        runtime_seconds=0.0,
        token_usage={"input": 0, "output": 0, "total": 0},
        error=None,
    )

    async def _fake_liveness(citations, sources_collected, budget_seconds=45.0):
        for ann in citations:
            ann.live = True
        return list(citations), LivenessTelemetry(
            checked=len(citations), live=len(citations),
            distinct_urls_checked=len(citations),
        )

    async def run():
        with patch.object(
            fnode, "run_codeagent_synthesis", return_value=synthesis
        ), patch.object(fnode, "run_critique", return_value=critique), patch.object(
            fnode, "check_citations_liveness", new=_fake_liveness
        ), patch.object(
            fnode, "save_report", side_effect=_capture_report
        ):
            return await fnode.codeagent_finalize_node(_state())

    result = asyncio.run(run())

    identity = result["synthesis_telemetry"]["result_identity"]
    assert identity["algo"] == "sha256-utf8-nl"
    assert identity["mission_id"] == "S84-M03-E2E"
    assert identity["title"] == "Identity Mission"
    # The report hash is the hash of the EXACT string save_report received.
    assert identity["report_sha256"] == identity_sha256(captured["report"])
    assert identity["report_len"] == len(captured["report"])
    # The scored draft is pre-preamble — a DIFFERENT string than the report.
    assert identity["scored_draft_sha256"] != identity["report_sha256"]
    assert identity["scored_draft_len"] > 0
    assert identity["span_reference_len"] > 0
    # And the worker-side assertion accepts this exact pairing end-to-end.
    assert_result_identity(
        row_mission_id="S84-M03-E2E",
        row_title="Identity Mission",
        result_markdown=captured["report"],
        result_protocol=result["protocol_payload"],
        synthesis_telemetry=result["synthesis_telemetry"],
        markdown_from_artifact=True,
    )


def test_result_identity_nests_in_declared_channel() -> None:
    """No new AgentState channel: the envelope nests inside the declared
    synthesis_telemetry channel (S42.2 class)."""

    import inspect

    src = inspect.getsource(fnode)
    assert 'synthesis_telemetry["result_identity"]' in src
    assert 'result["result_identity"]' not in src
