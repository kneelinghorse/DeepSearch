"""Result writer for updating TraceLab mission status and results.

This module writes research results back to TraceLab's missions table,
including status transitions, markdown reports, and protocol JSON.
Replaces HTTP webhook callbacks with direct database writes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from deepsearch.agent.codeagent_runner import _canonical_ledger_url
from deepsearch.agent.outcome import FinalOutcomePayload, MissionOutcome
from deepsearch.mission.title_utils import normalize_mission_title
from deepsearch.report.citation import CITATION_CLAIM_MAX_CHARS
from deepsearch.report.text import require_canonical_report_text

from .database import DatabaseClient, DatabaseError
from .lease import LeaseAttempt

logger = logging.getLogger(__name__)

_TRACELAB_JOB_ID_MAX_LENGTH = 100


def _strip_null_bytes(value: Any) -> Any:
    """Recursively strip NUL (\\x00) bytes from strings.

    Postgres TEXT/JSONB cannot store U+0000 — scraped pages and some LLM
    outputs occasionally contain stray nulls that break the insert.
    """
    if isinstance(value, str):
        return value.replace("\x00", "") if "\x00" in value else value
    if isinstance(value, dict):
        return {k: _strip_null_bytes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_null_bytes(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_strip_null_bytes(v) for v in value)
    return value


def _derive_llm_backend(model_id: str) -> Optional[str]:
    """Map a smolagents model_id to the runtime_identity.llm_backend label.

    HD3 truthfulness: keeps the persisted backend label in sync with the
    actual call path rather than the legacy ``LLM_BACKEND`` env probe. Only
    the supported Luna identity and historical DeepSeek/Gemini prefixes are
    mapped; unknown identities leave the legacy value untouched.
    """

    if not model_id:
        return None
    lower = model_id.strip().lower()
    if lower == "gpt-5.6-luna":
        return "openai"
    if lower.startswith("deepseek"):
        return "deepseek"
    if lower.startswith("gemini"):
        return "gemini"
    return None


class WriterError(DatabaseError):
    """Raised when result writing fails."""
    pass


class StaleLeaseError(WriterError):
    """Raised when a terminal write lacks the active lease ownership proof."""

    pass


class ResultIdentityError(Exception):
    """S84 m03 — the artifact handed to writeback failed identity checks.

    Deliberately NOT a :class:`WriterError`/``DatabaseError`` subclass: an
    identity violation is evidence of the CAT-R003 wrong-document class
    (roadmap flip-trigger: recurrence after this guard → undiagnosed writer
    race → retire signal), not a database fault. The worker catches this
    explicitly and lands the mission BLOCKED with the diagnostic payload —
    never a silent ship.
    """

    def __init__(self, message: str, diagnostic: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.diagnostic: Dict[str, Any] = dict(diagnostic or {})


# Wrapper H1 emitted by build_report_header (templates.py) — the one place
# mission identity appears deterministically in result_markdown.
_REPORT_H1_RE = re.compile(r"^#\s+Research Report:\s+(?P<mid>.+?)\s+-\s+(?P<title>.*)$")
_REPORT_H1_PREFIX = "# Research Report:"
_REPORT_HTTP_URL_RE = re.compile(r"https?://[^\s<>\[\]{}\"']+", re.IGNORECASE)

_SHA_PREVIEW_CHARS = 12


def identity_sha256(text: str) -> str:
    """sha256 over NEWLINE-NORMALIZED utf-8.

    The worker reads artifacts back through universal-newline translation
    (``read_text`` maps ``\\r\\n``/``\\r`` to ``\\n``), so hashing raw bytes
    would false-fire ``stale_draft`` on any CRLF-bearing report (S84 m03
    adversarial-review fix). Both the finalize node and the assertion hash
    through this same normalization.
    """
    return hashlib.sha256(
        text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    ).hexdigest()


def protocol_identity_sha256(protocol: Mapping[str, Any]) -> str:
    """Return SHA-256 over a canonical, JSON-safe protocol object.

    File persistence pretty-prints the protocol while the process-fence
    fallback carries a dictionary. Hash the object semantics rather than one
    serialization's whitespace so both transports share one identity.
    """

    canonical = json.dumps(
        dict(protocol),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _expected_report_h1(mission_id: str, title: str) -> str:
    """Reconstruct the wrapper H1 exactly as build_report_header renders it."""
    safe_title = normalize_mission_title(title, mission_id) or mission_id
    return f"{_REPORT_H1_PREFIX} {mission_id} - {safe_title}"


def _canonical_report_urls(
    markdown: str,
    *,
    expected_urls: Optional[set[str]] = None,
) -> tuple[set[str], int]:
    """Extract canonical HTTP(S) identities without trusting raw spelling.

    Markdown emphasis/code delimiters are removed only when the resulting URL
    matches an expected protocol citation. Characters such as ``*`` and
    backticks can be legitimate URL suffixes, so unconditional stripping would
    weaken the persisted body/protocol identity check.
    """

    canonical: set[str] = set()
    matches = list(_REPORT_HTTP_URL_RE.finditer(markdown))
    for match in matches:
        raw_url = match.group(0)
        try:
            canonical.add(_canonical_ledger_url(raw_url))
        except (TypeError, ValueError):
            continue

        if not expected_urls:
            continue

        # A legal URL may itself end in ``*``, ``!``, ``?``, or a backtick.
        # Remove Markdown wrapper suffixes only when the matching opening
        # delimiter is present immediately before this exact regex span. A
        # short run of prose punctuation is allowed only *outside* that proven
        # closing wrapper; bare punctuation is never shortened by inference.
        punctuation_tail = len(raw_url) - len(raw_url.rstrip(".,;:!?"))
        for wrapper in ("***", "**", "~~", "`", "*", "_"):
            if match.start() < len(wrapper):
                continue
            if markdown[match.start() - len(wrapper) : match.start()] != wrapper:
                continue
            for tail_length in range(punctuation_tail + 1):
                wrapped = raw_url[:-tail_length] if tail_length else raw_url
                if not wrapped.endswith(wrapper):
                    continue
                unwrapped = wrapped[: -len(wrapper)]
                try:
                    unwrapped_identity = _canonical_ledger_url(unwrapped)
                except (TypeError, ValueError):
                    continue
                if unwrapped_identity in expected_urls:
                    canonical.add(unwrapped_identity)

        # Markdown links and parenthesized URLs have a provable opening
        # delimiter. Remove exactly one unmatched closing delimiter, while
        # preserving balanced path parentheses such as ``.../Foo_(bar)``.
        if match.start() > 0 and markdown[match.start() - 1] == "(":
            for tail_length in range(punctuation_tail + 1):
                wrapped = raw_url[:-tail_length] if tail_length else raw_url
                if not wrapped.endswith(")"):
                    continue
                if wrapped.count(")") <= wrapped.count("("):
                    continue
                unwrapped = wrapped[:-1]
                try:
                    unwrapped_identity = _canonical_ledger_url(unwrapped)
                except (TypeError, ValueError):
                    continue
                if unwrapped_identity in expected_urls:
                    canonical.add(unwrapped_identity)
    return canonical, len(matches)


def assert_result_identity(
    *,
    row_mission_id: str,
    row_title: str,
    result_markdown: Optional[str],
    result_protocol: Optional[Mapping[str, Any]],
    synthesis_telemetry: Optional[Mapping[str, Any]],
    markdown_from_artifact: bool,
) -> None:
    """S84 m03 writeback identity assertion (decision #823 / CAT-R003).

    Asserts the artifact about to be persisted belongs to THIS mission and
    is the document the outcome gates scored. Five graduated checks — each
    SKIPS when its signal is absent, so legacy/fixture states without
    identity signals persist byte-identically to pre-m03 behavior:

    1. Mission identity — when ``result_markdown`` opens with the wrapper H1
       (``# Research Report: <id> - <title>``), its id must equal the claimed
       row's mission_id and its title must equal the row title under the
       shared :func:`normalize_mission_title` (CAT-R003 shipped CAT-R004's
       header).
    2. Citation URL overlap — when the protocol carries citations, at least
       one citation URL must appear in the body (CAT-R003: 0 of 47).
    3. Draft identity — when the finalize node recorded a
       ``result_identity`` envelope (synthesis_telemetry), the persisted
       markdown must hash-match ``report_sha256``; if the emitted artifact
       was unreadable and the worker would ship REGENERATED bytes instead,
       that is itself a violation (the gate-scored document is gone).
    4. Protocol identity — a modern protocol must carry this mission's id and
       hash-match the canonical JSON object emitted by finalize.
    5. Span validity — citation spans must be structurally sane
       (0 <= start <= end) and, when the envelope carries
       ``span_reference_len``, in-bounds of the extraction reference text.

    Raises :class:`ResultIdentityError` carrying a machine-readable
    ``diagnostic`` dict naming every violated check. Never mutates inputs.
    """

    violations: list[str] = []
    diagnostic: Dict[str, Any] = {
        "markdown_from_artifact": markdown_from_artifact,
        "row_mission_id": row_mission_id,
    }

    if isinstance(result_markdown, str):
        try:
            require_canonical_report_text(result_markdown)
        except ValueError:
            diagnostic["report_text"] = {"canonical": False}
            violations.append("noncanonical_report")

    identity = None
    if isinstance(synthesis_telemetry, Mapping):
        candidate = synthesis_telemetry.get("result_identity")
        if isinstance(candidate, Mapping):
            identity = candidate
    envelope_id = ""
    envelope_title = ""
    if identity is not None:
        raw_id = identity.get("mission_id")
        envelope_id = raw_id.strip() if isinstance(raw_id, str) else ""
        envelope_title = str(identity.get("title") or "").strip()

    # -- 1. mission identity ------------------------------------------------
    # The envelope's mission_id/title were recorded by the SAME finalize run
    # that rendered the wrapper H1, so the expected header is reconstructed
    # exactly — no parsing ambiguity, no row-vs-context title divergence
    # (S84 m03 adversarial-review fix: row.title and the generation chain's
    # context title can legitimately differ, so row.title is never compared
    # against the H1). Cross-mission contamination is caught two ways:
    # envelope-id vs row-id (state-level) and H1 vs expected (artifact-level).
    expected_row_id = str(row_mission_id or "").strip()
    if envelope_id and expected_row_id and envelope_id != expected_row_id:
        diagnostic["header"] = {
            "envelope_mission_id": envelope_id,
            "expected_mission_id": expected_row_id,
        }
        violations.append("mission_id_mismatch")
    elif isinstance(result_markdown, str) and result_markdown:
        first_line = result_markdown.split("\n", 1)[0].strip()
        if first_line.startswith(_REPORT_H1_PREFIX):
            if envelope_id:
                expected_h1 = _expected_report_h1(envelope_id, envelope_title)
                if first_line != expected_h1:
                    diagnostic["header"] = {
                        "first_line": first_line,
                        "expected_h1": expected_h1,
                    }
                    if not first_line.startswith(
                        f"{_REPORT_H1_PREFIX} {envelope_id} - "
                    ):
                        violations.append("mission_id_mismatch")
                    else:
                        violations.append("title_mismatch")
            else:
                # Legacy state without an envelope: the id is still checkable
                # against the claimed row; the title is NOT (row.title vs
                # context-title divergence would false-fire).
                match = _REPORT_H1_RE.match(first_line)
                if match and expected_row_id:
                    parsed_id = match.group("mid").strip()
                    if parsed_id != expected_row_id:
                        diagnostic["header"] = {
                            "parsed_mission_id": parsed_id,
                            "expected_mission_id": expected_row_id,
                        }
                        violations.append("mission_id_mismatch")

    # -- 2. protocol mission identity --------------------------------------
    # Legacy protocols predate the result-identity envelope and may omit a
    # mission_id. A new finalize envelope carrying protocol_sha256 makes the
    # field mandatory; an explicitly present mismatch is always unsafe.
    expected_protocol_sha = (
        identity.get("protocol_sha256") if identity is not None else None
    )
    protocol_id_present = False
    protocol_id_valid = False
    if isinstance(result_protocol, Mapping) and "mission_id" in result_protocol:
        protocol_id_present = True
        raw_protocol_id = result_protocol.get("mission_id")
        protocol_id = (
            raw_protocol_id.strip() if isinstance(raw_protocol_id, str) else ""
        )
        protocol_id_valid = bool(protocol_id)
        if not protocol_id_valid:
            violations.append("protocol_mission_id_invalid")
        elif expected_row_id and protocol_id != expected_row_id:
            violations.append("protocol_mission_id_mismatch")
    if isinstance(expected_protocol_sha, str) and expected_protocol_sha:
        if not protocol_id_present:
            violations.append("protocol_mission_id_missing")
        diagnostic["protocol_identity"] = {
            "mission_id_present": protocol_id_present,
            "mission_id_valid": protocol_id_valid,
            "mission_id_matches": (
                protocol_id_valid
                and isinstance(result_protocol, Mapping)
                and str(result_protocol.get("mission_id") or "").strip()
                == expected_row_id
            ),
        }

    # -- 3. citation-URL x body overlap ------------------------------------
    citations = []
    if isinstance(result_protocol, Mapping):
        raw = result_protocol.get("citations")
        if isinstance(raw, list):
            citations = [c for c in raw if isinstance(c, Mapping)]
    citation_urls: set[str] = set()
    invalid_citation_urls = 0
    for citation in citations:
        raw_url = citation.get("url")
        if not isinstance(raw_url, str):
            invalid_citation_urls += 1
            continue
        try:
            canonical_url = _canonical_ledger_url(raw_url)
        except (TypeError, ValueError):
            invalid_citation_urls += 1
            continue
        if canonical_url != raw_url:
            invalid_citation_urls += 1
            continue
        citation_urls.add(canonical_url)
    invalid_source_urls = 0
    if isinstance(result_protocol, Mapping):
        raw_sources = result_protocol.get("sources_collected")
        if isinstance(raw_sources, list):
            for source in raw_sources:
                if not isinstance(source, Mapping):
                    invalid_source_urls += 1
                    continue
                raw_url = source.get("url")
                if not isinstance(raw_url, str):
                    invalid_source_urls += 1
                    continue
                try:
                    canonical_url = _canonical_ledger_url(raw_url)
                except (TypeError, ValueError):
                    invalid_source_urls += 1
                    continue
                if canonical_url != raw_url:
                    invalid_source_urls += 1
    if isinstance(expected_protocol_sha, str) and expected_protocol_sha and (
        invalid_citation_urls or invalid_source_urls
    ):
        diagnostic["protocol_urls"] = {
            "invalid_citations": invalid_citation_urls,
            "invalid_sources": invalid_source_urls,
        }
        violations.append("unsafe_protocol_urls")
    body_urls: set[str] = set()
    body_url_count = 0
    if isinstance(result_markdown, str):
        body_urls, body_url_count = _canonical_report_urls(
            result_markdown,
            expected_urls=citation_urls,
        )
    # Only judge overlap when the body actually renders URLs — a marker-only
    # report (no References section; [cite:] markers stripped at extraction)
    # legitimately contains none (S84 m03 adversarial-review fix).
    if citation_urls and body_url_count:
        overlap = len(citation_urls & body_urls)
        diagnostic["citation_overlap"] = {
            "citation_urls": len(citation_urls),
            "found_in_body": overlap,
        }
        if overlap == 0:
            violations.append("zero_citation_url_overlap")

    # -- 4. gate-scored draft == persisted draft ---------------------------
    if identity is not None:
        expected_sha = identity.get("report_sha256")
        expected_len = identity.get("report_len")
        if not markdown_from_artifact:
            # The finalize node ran (the envelope exists) but the worker
            # could not read its emitted artifact and would ship REGENERATED
            # bytes — on the post-S78 path that regeneration is a template
            # husk, and the gate-scored document is not what would persist.
            # Blocking here is the point, even when finalize's own report
            # persistence failed (no report_sha256 recorded).
            violations.append("emitted_artifact_missing")
        elif (
            isinstance(expected_sha, str)
            and expected_sha
            and isinstance(result_markdown, str)
        ):
            actual_sha = identity_sha256(result_markdown)
            diagnostic["draft_hash"] = {
                "expected": expected_sha[:_SHA_PREVIEW_CHARS],
                "actual": actual_sha[:_SHA_PREVIEW_CHARS],
                "report_len_expected": expected_len,
                "report_len_actual": len(result_markdown),
            }
            if actual_sha != expected_sha:
                violations.append("stale_draft")
        if (
            isinstance(expected_len, int)
            and isinstance(result_markdown, str)
            and expected_len != len(result_markdown)
        ):
            diagnostic["report_length"] = {
                "expected": expected_len,
                "actual": len(result_markdown),
            }
            violations.append("report_length_mismatch")

        if isinstance(expected_protocol_sha, str) and expected_protocol_sha:
            if not isinstance(result_protocol, Mapping):
                violations.append("protocol_artifact_missing")
            else:
                try:
                    actual_protocol_sha = protocol_identity_sha256(result_protocol)
                except (TypeError, ValueError):
                    diagnostic["protocol_hash"] = {"json_safe": False}
                    violations.append("protocol_not_json_safe")
                else:
                    diagnostic["protocol_hash"] = {
                        "json_safe": True,
                        "expected": expected_protocol_sha[:_SHA_PREVIEW_CHARS],
                        "actual": actual_protocol_sha[:_SHA_PREVIEW_CHARS],
                    }
                    if actual_protocol_sha != expected_protocol_sha:
                        violations.append("stale_protocol")

    # -- 5. span-validity spot-check ---------------------------------------
    span_reference_len = (
        identity.get("span_reference_len") if identity is not None else None
    )
    if (
        isinstance(span_reference_len, int)
        and isinstance(result_markdown, str)
        and span_reference_len != len(result_markdown)
    ):
        diagnostic["span_reference_length"] = {
            "expected": span_reference_len,
            "actual": len(result_markdown),
        }
        violations.append("span_reference_length_mismatch")

    invalid_spans = 0
    for c in citations:
        start = c.get("start_index")
        end = c.get("end_index")
        if not isinstance(start, int) or not isinstance(end, int):
            invalid_spans += 1
        elif start < 0 or end <= start:
            invalid_spans += 1
        elif isinstance(result_markdown, str) and end > len(result_markdown):
            invalid_spans += 1
        elif isinstance(result_markdown, str):
            cited_slice = result_markdown[start:end]
            if (
                len(cited_slice) > CITATION_CLAIM_MAX_CHARS
                or cited_slice != cited_slice.strip()
            ):
                invalid_spans += 1
        elif isinstance(span_reference_len, int) and end > span_reference_len:
            invalid_spans += 1
    if invalid_spans:
        diagnostic["spans"] = {
            "citations": len(citations),
            "invalid": invalid_spans,
            "span_reference_len": span_reference_len,
        }
        violations.append("invalid_spans")

    if violations:
        diagnostic["violations"] = violations
        raise ResultIdentityError("; ".join(violations), diagnostic=diagnostic)


@dataclass
class ExecutionMetadata:
    """Metadata about research execution for TraceLab tracking.

    S75 m01: ``loops_executed``, ``sources_count``, ``quality_gates_passed``
    were removed. Post-S64 entry-point flip the deployed path doesn't run
    legacy LangGraph stages or quality gates, so those fields emitted lying
    zeros / Falses on every codeagent run. T42.3 confirmed zero TraceLab
    readers (frontend ExecutionTimeline ported to ``protocol.report_metadata.*``).

    S84 m04: ``degraded_mode``/``degradation_policy`` were removed — their
    only writer was the empty_state default, so they persisted a constant
    not-degraded verdict for retired machinery. The live tier is
    ``final_outcome.delivery_quality``.

    Attributes:
        total_tokens: Total tokens consumed during research
        duration_seconds: Total execution time in seconds
        agent_version: Version of the DeepSearch agent
        convergence_reason: Why the research loop terminated
    """
    total_tokens: int
    duration_seconds: float
    agent_version: str = "deepsearch-1.0"
    convergence_reason: Optional[str] = None
    contract_id: Optional[str] = None
    contract_path: Optional[str] = None
    telemetry_path: Optional[str] = None
    runtime_identity: Dict[str, Any] = field(default_factory=dict)
    config_fingerprint: Optional[str] = None
    route_summary: Dict[str, Any] = field(default_factory=dict)
    finding_counts: Dict[str, Any] = field(default_factory=dict)
    source_counts: Dict[str, Any] = field(default_factory=dict)
    final_outcome: FinalOutcomePayload = field(default_factory=dict)
    synthesis_telemetry: Dict[str, Any] = field(default_factory=dict)
    # S84 m03 — populated ONLY when the writeback identity assertion fired;
    # carries the machine-readable diagnostic that lands with the BLOCKED row.
    identity_violation: Optional[Dict[str, Any]] = None
    # S84 m04 — the per-mission quality-series record (PD137 calibration
    # datapoint); previously computed but never persisted to the DB.
    quality_record: Optional[Dict[str, Any]] = None
    # S87 m05 — redacted lease-attempt audit. Never contains the opaque token.
    lease_attempt: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONB serialization."""
        data = {
            "total_tokens": self.total_tokens,
            "duration_seconds": round(self.duration_seconds, 2),
            "agent_version": self.agent_version,
        }
        if self.convergence_reason:
            data["convergence_reason"] = self.convergence_reason
        if self.contract_id:
            data["contract_id"] = self.contract_id
        if self.contract_path:
            data["contract_path"] = self.contract_path
        if self.telemetry_path:
            data["telemetry_path"] = self.telemetry_path
        if self.runtime_identity:
            data["runtime_identity"] = dict(self.runtime_identity)
        if self.config_fingerprint:
            data["config_fingerprint"] = self.config_fingerprint
        if self.route_summary:
            data["route_summary"] = dict(self.route_summary)
        if self.finding_counts:
            data["finding_counts"] = dict(self.finding_counts)
        if self.source_counts:
            data["source_counts"] = dict(self.source_counts)
        if self.final_outcome:
            data["final_outcome"] = dict(self.final_outcome)
        if self.synthesis_telemetry:
            data["synthesis_telemetry"] = dict(self.synthesis_telemetry)
        if self.identity_violation:
            data["identity_violation"] = dict(self.identity_violation)
        if self.quality_record:
            data["quality_record"] = dict(self.quality_record)
        if self.lease_attempt:
            data["lease_attempt"] = dict(self.lease_attempt)
        return data


class ResultWriter:
    """Writes research results to TraceLab's PostgreSQL missions table.

    Handles status transitions and result serialization for completed
    research missions. Uses parameterized queries for SQL injection safety.

    Usage:
        writer = ResultWriter(db_client)

        # After research completes successfully
        await writer.mark_completed(
            mission_id=mission_id,
            result_markdown=markdown_report,
            result_protocol=protocol.to_dict(),
            execution_metadata=metadata,
            lease=lease,
            job_id=job_id,
        )

        # Or if research fails
        await writer.mark_failed(
            mission_id,
            "Research failed: timeout",
            lease=lease,
            job_id=job_id,
        )
    """

    # Eligible Evidence Ledger terminal updates first lock the fenced mission
    # attempt, then enroll its immutable delivery snapshot. The terminal UPDATE
    # is reachable only when the insert succeeded or an exact enrollment
    # already exists. A reused job identity with a different result key,
    # attempt, status, or schema therefore cannot commit the mission transition.
    # The final branch makes an exact retry read-only; it deliberately does not
    # enroll historical terminal rows that predate this forward-only boundary.
    MARK_COMPLETED_SQL = """
        WITH eligible AS MATERIALIZED (
            SELECT id
            FROM missions
            WHERE id = $1
              AND status = 'in_progress'
              AND deepsearch_lease_owner = $6
              AND deepsearch_lease_token = $7
              AND deepsearch_attempt_count = $8
              AND deepsearch_lease_expires_at > NOW()
            FOR UPDATE
        ),
        enrolled AS (
            INSERT INTO deepsearch_evidence_outbox (
                mission_id,
                deepsearch_job_id,
                deepsearch_result_key,
                mission_attempt_count,
                terminal_status,
                schema_version
            )
            SELECT
                eligible.id,
                $10,
                $9,
                $8,
                'completed',
                1
            FROM eligible
            ON CONFLICT (mission_id, deepsearch_job_id) DO NOTHING
            RETURNING
                mission_id,
                deepsearch_job_id,
                deepsearch_result_key,
                mission_attempt_count,
                terminal_status,
                schema_version
        ),
        enrollment_match AS (
            SELECT eligible.id
            FROM eligible
            JOIN enrolled
              ON enrolled.mission_id = eligible.id
             AND enrolled.deepsearch_job_id = $10
             AND enrolled.deepsearch_result_key = $9
             AND enrolled.mission_attempt_count = $8
             AND enrolled.terminal_status = 'completed'
             AND enrolled.schema_version = 1
            UNION ALL
            SELECT eligible.id
            FROM eligible
            JOIN deepsearch_evidence_outbox AS existing
              ON existing.mission_id = eligible.id
             AND existing.deepsearch_job_id = $10
             AND existing.deepsearch_result_key = $9
             AND existing.mission_attempt_count = $8
             AND existing.terminal_status = 'completed'
             AND existing.schema_version = 1
        ),
        applied AS (
            UPDATE missions AS mission
            SET
                status = 'completed',
                completed_at = $2,
                result_markdown = $3,
                result_protocol = $4,
                execution_metadata = CASE
                    WHEN execution_metadata ? 'prior_execution_attempts' THEN
                        COALESCE($5::jsonb, '{}'::jsonb) || jsonb_build_object(
                            'prior_execution_attempts',
                            execution_metadata->'prior_execution_attempts'
                        )
                    ELSE $5::jsonb
                END,
                error_message = NULL,
                updated_at = $2,
                deepsearch_result_key = $9,
                deepsearch_job_id = $10,
                deepsearch_lease_token = NULL,
                deepsearch_lease_expires_at = NULL
            FROM enrollment_match
            WHERE mission.id = enrollment_match.id
            RETURNING mission.id
        )
        SELECT id FROM applied
        UNION ALL
        SELECT mission.id
        FROM missions AS mission
        JOIN deepsearch_evidence_outbox AS existing
          ON existing.mission_id = mission.id
         AND existing.deepsearch_job_id = $10
         AND existing.deepsearch_result_key = $9
         AND existing.mission_attempt_count = $8
         AND existing.terminal_status = 'completed'
         AND existing.schema_version = 1
        WHERE mission.id = $1
          AND mission.status = 'completed'
          AND mission.deepsearch_result_key = $9
          AND mission.deepsearch_job_id = $10
          AND mission.deepsearch_attempt_count = $8
        LIMIT 1
    """

    MARK_VALIDATION_FAILED_SQL = """
        WITH eligible AS MATERIALIZED (
            SELECT id
            FROM missions
            WHERE id = $1
              AND status = 'in_progress'
              AND deepsearch_lease_owner = $7
              AND deepsearch_lease_token = $8
              AND deepsearch_attempt_count = $9
              AND deepsearch_lease_expires_at > NOW()
            FOR UPDATE
        ),
        enrolled AS (
            INSERT INTO deepsearch_evidence_outbox (
                mission_id,
                deepsearch_job_id,
                deepsearch_result_key,
                mission_attempt_count,
                terminal_status,
                schema_version
            )
            SELECT
                eligible.id,
                $11,
                $10,
                $9,
                'validation_failed',
                1
            FROM eligible
            ON CONFLICT (mission_id, deepsearch_job_id) DO NOTHING
            RETURNING
                mission_id,
                deepsearch_job_id,
                deepsearch_result_key,
                mission_attempt_count,
                terminal_status,
                schema_version
        ),
        enrollment_match AS (
            SELECT eligible.id
            FROM eligible
            JOIN enrolled
              ON enrolled.mission_id = eligible.id
             AND enrolled.deepsearch_job_id = $11
             AND enrolled.deepsearch_result_key = $10
             AND enrolled.mission_attempt_count = $9
             AND enrolled.terminal_status = 'validation_failed'
             AND enrolled.schema_version = 1
            UNION ALL
            SELECT eligible.id
            FROM eligible
            JOIN deepsearch_evidence_outbox AS existing
              ON existing.mission_id = eligible.id
             AND existing.deepsearch_job_id = $11
             AND existing.deepsearch_result_key = $10
             AND existing.mission_attempt_count = $9
             AND existing.terminal_status = 'validation_failed'
             AND existing.schema_version = 1
        ),
        applied AS (
            UPDATE missions AS mission
            SET
                status = 'validation_failed',
                completed_at = $2,
                error_message = $3,
                result_markdown = $4,
                result_protocol = $5,
                execution_metadata = CASE
                    WHEN execution_metadata ? 'prior_execution_attempts' THEN
                        COALESCE($6::jsonb, '{}'::jsonb) || jsonb_build_object(
                            'prior_execution_attempts',
                            execution_metadata->'prior_execution_attempts'
                        )
                    ELSE $6::jsonb
                END,
                updated_at = $2,
                deepsearch_result_key = $10,
                deepsearch_job_id = $11,
                deepsearch_lease_token = NULL,
                deepsearch_lease_expires_at = NULL
            FROM enrollment_match
            WHERE mission.id = enrollment_match.id
            RETURNING mission.id
        )
        SELECT id FROM applied
        UNION ALL
        SELECT mission.id
        FROM missions AS mission
        JOIN deepsearch_evidence_outbox AS existing
          ON existing.mission_id = mission.id
         AND existing.deepsearch_job_id = $11
         AND existing.deepsearch_result_key = $10
         AND existing.mission_attempt_count = $9
         AND existing.terminal_status = 'validation_failed'
         AND existing.schema_version = 1
        WHERE mission.id = $1
          AND mission.status = 'validation_failed'
          AND mission.deepsearch_result_key = $10
          AND mission.deepsearch_job_id = $11
          AND mission.deepsearch_attempt_count = $9
        LIMIT 1
    """

    MARK_BLOCKED_SQL = """
        WITH applied AS (
            UPDATE missions
            SET
                status = 'blocked',
                completed_at = $2,
                error_message = $3,
                result_markdown = $4,
                result_protocol = $5,
                execution_metadata = CASE
                    WHEN execution_metadata ? 'prior_execution_attempts' THEN
                        COALESCE($6::jsonb, '{}'::jsonb) || jsonb_build_object(
                            'prior_execution_attempts',
                            execution_metadata->'prior_execution_attempts'
                        )
                    ELSE $6::jsonb
                END,
                updated_at = $2,
                deepsearch_result_key = $10,
                deepsearch_job_id = $11,
                deepsearch_lease_token = NULL,
                deepsearch_lease_expires_at = NULL
            WHERE id = $1
              AND status = 'in_progress'
              AND deepsearch_lease_owner = $7
              AND deepsearch_lease_token = $8
              AND deepsearch_attempt_count = $9
              AND deepsearch_lease_expires_at > NOW()
            RETURNING id
        )
        SELECT id FROM applied
        UNION ALL
        SELECT id FROM missions
        WHERE id = $1
          AND status = 'blocked'
          AND deepsearch_result_key = $10
          AND deepsearch_job_id = $11
        LIMIT 1
    """

    MARK_FAILED_SQL = MARK_BLOCKED_SQL

    # S90 (#959) — return a mission to 'queued' after a transient, zero-token
    # synthesis outage so the lease's remaining attempts become reachable.
    #
    # Deliberately NOT terminal: completed_at stays NULL and no result artifact
    # is written, because there is no result — only an outage. error_message
    # records why, and is overwritten by the next attempt.
    #
    # deepsearch_attempt_count is PRESERVED, not reset. That count is the only
    # thing bounding the requeue: the claim predicate refuses a row at or past
    # max_attempts, and the widened dead-letter sweep then lands it in 'blocked'.
    # Zeroing it here would make the loop unbounded.
    #
    # Lease-guarded and idempotent without using ``deepsearch_result_key``.
    # That key is reserved for terminal result persistence and completion
    # receipt correlation. A duplicate apply re-reads only the same queued,
    # unfenced attempt; a later claim increments the attempt count and no
    # longer matches this branch.
    MARK_REQUEUED_SQL = """
        WITH applied AS (
            UPDATE missions
            SET
                status = 'queued',
                execution_metadata = jsonb_set(
                    COALESCE(execution_metadata, '{}'::jsonb),
                    '{prior_execution_attempts}',
                    COALESCE(execution_metadata->'prior_execution_attempts', '[]'::jsonb)
                    || jsonb_build_array($6::jsonb)
                ),
                error_message = $2,
                updated_at = NOW(),
                deepsearch_result_key = NULL,
                deepsearch_job_id = NULL,
                deepsearch_lease_owner = NULL,
                deepsearch_lease_token = NULL,
                deepsearch_lease_expires_at = NULL
            WHERE id = $1
              AND status = 'in_progress'
              AND deepsearch_lease_owner = $3
              AND deepsearch_lease_token = $4
              AND deepsearch_attempt_count = $5
              AND deepsearch_lease_expires_at > NOW()
            RETURNING id
        )
        SELECT id FROM applied
        UNION ALL
        SELECT id FROM missions
        WHERE id = $1
          AND status = 'queued'
          AND deepsearch_attempt_count = $5
          AND error_message = $2
          AND deepsearch_lease_owner IS NULL
          AND deepsearch_lease_token IS NULL
          AND deepsearch_lease_expires_at IS NULL
          AND deepsearch_result_key IS NULL
          AND deepsearch_job_id IS NULL
        LIMIT 1
    """

    def __init__(self, db_client: DatabaseClient) -> None:
        """Initialize the result writer.

        Args:
            db_client: Connected DatabaseClient instance
        """
        self.db_client = db_client

    async def mark_completed(
        self,
        mission_id: str,
        result_markdown: str,
        result_protocol: Dict[str, Any],
        execution_metadata: ExecutionMetadata,
        lease: LeaseAttempt,
        job_id: str,
        completed_at: Optional[datetime] = None,
    ) -> bool:
        """Mark a mission as completed with results.

        Sets status to 'completed' and stores the research results
        including markdown report, protocol JSON, and execution metadata.

        Args:
            mission_id: TraceLab mission UUID
            result_markdown: Markdown-formatted research report
            result_protocol: MissionProtocolComplete dictionary (JSONB)
            execution_metadata: ExecutionMetadata with run statistics
            lease: Active lease attempt returned by the atomic claim
            job_id: Fenced DeepSearch execution identifier used by the receipt
            completed_at: Optional completion timestamp (defaults to now)

        Returns:
            True when the write applies or an identical retry already applied

        Raises:
            StaleLeaseError: If ownership expired or conflicts with the result
            WriterError: If database update or serialization fails
        """
        if completed_at is None:
            # Use timezone-naive UTC datetime for PostgreSQL/asyncpg compatibility
            completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        job_id = str(job_id or "").strip()
        if not job_id:
            raise WriterError("job_id is required for terminal persistence")
        if len(job_id) > _TRACELAB_JOB_ID_MAX_LENGTH:
            raise WriterError("job_id exceeds TraceLab's 100-character limit")

        logger.info(
            "Marking mission %s as completed (markdown: %d chars)",
            mission_id,
            len(result_markdown),
        )

        try:
            # Citation spans and identity were assigned against this exact
            # representation. Never move them at the database boundary.
            clean_markdown = require_canonical_report_text(result_markdown)
            clean_protocol = _strip_null_bytes(result_protocol)
            clean_metadata = _strip_null_bytes(execution_metadata.to_dict())

            # Serialize protocol and metadata for JSONB columns
            protocol_json = self._serialize_jsonb(clean_protocol, "result_protocol")
            metadata_json = self._serialize_jsonb(clean_metadata, "execution_metadata")

            result = await self.db_client.fetchval(
                self.MARK_COMPLETED_SQL,
                mission_id,
                completed_at,
                clean_markdown,
                protocol_json,
                metadata_json,
                lease.owner,
                lease.token,
                lease.attempt_count,
                lease.persistence_key(mission_id),
                job_id,
            )

            if result is None:
                raise StaleLeaseError(
                    f"Mission {mission_id} completion rejected: stale lease or conflicting result"
                )

            logger.info(
                "Mission %s marked as completed (tokens: %d)",
                mission_id,
                execution_metadata.total_tokens,
            )
            return True

        except WriterError:
            raise
        except Exception as e:
            logger.error("Failed to mark mission %s as completed: %s", mission_id, e)
            raise WriterError(f"Failed to mark completed: {e}") from e

    async def mark_failed(
        self,
        mission_id: str,
        error_message: str,
        *,
        outcome: MissionOutcome | str = MissionOutcome.BLOCKED,
        lease: LeaseAttempt,
        job_id: str,
        result_markdown: str | None = None,
        result_protocol: Dict[str, Any] | None = None,
        execution_metadata: ExecutionMetadata | None = None,
        completed_at: Optional[datetime] = None,
    ) -> bool:
        """Mark a mission as failed with an error message.

        Sets status to 'validation_failed' or 'blocked' and records the error.

        Args:
            mission_id: TraceLab mission UUID
            error_message: Description of the failure
            outcome: Validation-failed or blocked terminal state
            lease: Active lease attempt returned by the atomic claim
            job_id: Fenced DeepSearch execution identifier used by the receipt
            result_markdown: Optional report artifact preserved on failure
            result_protocol: Optional protocol artifact preserved on failure
            execution_metadata: Optional execution diagnostics
            completed_at: Optional completion timestamp (defaults to now)

        Returns:
            True when the write applies or an identical retry already applied

        Raises:
            StaleLeaseError: If ownership expired or conflicts with the result
            WriterError: If database update fails
        """
        if completed_at is None:
            # Use timezone-naive UTC datetime for PostgreSQL/asyncpg compatibility
            completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        job_id = str(job_id or "").strip()
        if not job_id:
            raise WriterError("job_id is required for terminal persistence")
        if len(job_id) > _TRACELAB_JOB_ID_MAX_LENGTH:
            raise WriterError("job_id exceeds TraceLab's 100-character limit")

        if isinstance(outcome, MissionOutcome):
            normalized_outcome = outcome
        else:
            try:
                normalized_outcome = MissionOutcome(str(outcome or "").strip().lower())
            except ValueError as exc:
                raise WriterError(f"Unsupported terminal outcome: {outcome}") from exc

        if normalized_outcome is MissionOutcome.COMPLETE:
            raise WriterError("mark_failed cannot persist the COMPLETE outcome")

        # Truncate very long error messages
        if len(error_message) > 4000:
            error_message = error_message[:3997] + "..."

        logger.error(
            "Marking mission %s as %s: %s",
            mission_id,
            normalized_outcome.value,
            error_message[:200]
        )

        try:
            if result_markdown is None:
                clean_markdown = None
            elif normalized_outcome is MissionOutcome.VALIDATION_FAILED:
                # validation_failed enrolls the immutable Evidence Ledger
                # outbox, so its report must obey the same lossless contract
                # as completed. BLOCKED cannot enroll and retains its legacy
                # diagnostic sanitation path below.
                clean_markdown = require_canonical_report_text(result_markdown)
            else:
                clean_markdown = _strip_null_bytes(result_markdown)
            clean_protocol = (
                _strip_null_bytes(result_protocol)
                if result_protocol is not None
                else None
            )
            clean_metadata = (
                _strip_null_bytes(execution_metadata.to_dict())
                if execution_metadata is not None
                else None
            )
            sql = (
                self.MARK_VALIDATION_FAILED_SQL
                if normalized_outcome is MissionOutcome.VALIDATION_FAILED
                else self.MARK_BLOCKED_SQL
            )
            result = await self.db_client.fetchval(
                sql,
                mission_id,
                completed_at,
                _strip_null_bytes(error_message),
                clean_markdown,
                self._serialize_optional_jsonb(clean_protocol, "result_protocol"),
                self._serialize_optional_jsonb(clean_metadata, "execution_metadata"),
                lease.owner,
                lease.token,
                lease.attempt_count,
                lease.persistence_key(mission_id),
                job_id,
            )

            if result is None:
                raise StaleLeaseError(
                    f"Mission {mission_id} terminal write rejected: "
                    "stale lease or conflicting result"
                )

            logger.info("Mission %s marked as %s", mission_id, normalized_outcome.value)
            return True

        except WriterError:
            raise
        except Exception as e:
            logger.error("Failed to mark mission %s as failed: %s", mission_id, e)
            raise WriterError(f"Failed to mark failed: {e}") from e

    async def mark_requeued(
        self,
        mission_id: str,
        error_message: str,
        *,
        lease: LeaseAttempt,
        execution_metadata: ExecutionMetadata,
    ) -> bool:
        """Return a mission to 'queued' after a transient, zero-token outage.

        S90 (#959). ``blocked`` is terminal — nothing in the worker ever moved a
        mission back to 'queued' — so a provider outage longer than the
        in-process retry window destroyed the run and left the lease's remaining
        attempts unreachable. This is the path that makes them reachable.

        The caller is responsible for establishing that the failure was
        genuinely transient and that attempts remain; this method only performs
        the lease-guarded write.

        Args:
            mission_id: TraceLab mission UUID
            error_message: Why the attempt was abandoned
            lease: Active lease attempt returned by the atomic claim

        Returns:
            True when the write applies or an identical retry already applied

        Raises:
            StaleLeaseError: If ownership expired or conflicts
            WriterError: If the database update fails
        """
        if len(error_message) > 4000:
            error_message = error_message[:3997] + "..."

        logger.warning(
            "Requeuing mission %s after transient failure (attempt %d): %s",
            mission_id,
            lease.attempt_count,
            error_message[:200],
        )

        try:
            result = await self.db_client.fetchval(
                self.MARK_REQUEUED_SQL,
                mission_id,
                _strip_null_bytes(error_message),
                lease.owner,
                lease.token,
                lease.attempt_count,
                self._serialize_jsonb(
                    _strip_null_bytes(execution_metadata.to_dict()), "execution_metadata"
                ),
            )

            if result is None:
                raise StaleLeaseError(
                    f"Mission {mission_id} requeue rejected: "
                    "stale lease or conflicting transition"
                )

            logger.info("Mission %s returned to queued", mission_id)
            return True

        except (WriterError, StaleLeaseError):
            raise
        except Exception as e:
            logger.error("Failed to requeue mission %s: %s", mission_id, e)
            raise WriterError(f"Failed to requeue: {e}") from e

    @staticmethod
    def _serialize_jsonb(data: Dict[str, Any], field_name: str) -> str:
        """Serialize a dictionary to JSON string for JSONB column.

        Args:
            data: Dictionary to serialize
            field_name: Name of field (for error messages)

        Returns:
            JSON string

        Raises:
            WriterError: If serialization fails
        """
        try:
            return json.dumps(data, default=str, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            raise WriterError(f"Failed to serialize {field_name}: {e}") from e

    @classmethod
    def _serialize_optional_jsonb(
        cls,
        data: Dict[str, Any] | None,
        field_name: str,
    ) -> str | None:
        if data is None:
            return None
        return cls._serialize_jsonb(data, field_name)


def build_execution_metadata(
    state: Dict[str, Any],
    duration_seconds: float,
) -> ExecutionMetadata:
    """Build ExecutionMetadata from completed AgentState.

    Convenience function to extract execution statistics from
    the final AgentState after research completes.

    S75 m01: ``loops_executed`` / ``sources_count`` / ``quality_gates_passed``
    derivation removed. Those fields emitted lying values on the post-S64
    codeagent path; T42.3 cleared the readers and the dataclass was trimmed.

    Args:
        state: Final AgentState dictionary
        duration_seconds: Total execution time

    Returns:
        ExecutionMetadata populated from state
    """
    # Extract token count from telemetry (preferred) with legacy fallbacks.
    tokens = 0
    telemetry = state.get("telemetry") if isinstance(state.get("telemetry"), dict) else {}
    token_usage_raw = telemetry.get("token_usage")
    token_usage = token_usage_raw if isinstance(token_usage_raw, dict) else {}
    for key in ("total_tokens", "total"):
        raw = token_usage.get(key)
        if raw:
            try:
                tokens = int(raw)
                break
            except (TypeError, ValueError):
                continue
    if not tokens:
        raw_total = state.get("total_tokens")
        if raw_total:
            try:
                tokens = int(raw_total)
            except (TypeError, ValueError):
                tokens = 0

    # Get convergence reason if available
    convergence = state.get("termination_reason")

    # S68.5 HD2 — read the agent-side telemetry envelope. Mirrors the
    # ``synthesis_telemetry`` dict that ``codeagent_finalize_node`` writes onto
    # AgentState (engine, model_id, framework, step_count, runtime_seconds,
    # token_usage, tool_call_summary, synthesis_error + nested critique_/
    # citation_/liveness_telemetry envelopes). Verbatim copy; no flattening.
    synthesis_telemetry_raw = state.get("synthesis_telemetry")
    synthesis_telemetry = (
        dict(synthesis_telemetry_raw)
        if isinstance(synthesis_telemetry_raw, Mapping)
        else {}
    )

    from deepsearch.llm.usage import compiler_usage_events

    compiler_events = compiler_usage_events(state.get("mission_contract") or {})
    if compiler_events:
        synthesis_telemetry["compiler_provider_attempts"] = compiler_events

    runtime_identity_raw = telemetry.get("runtime_identity")
    runtime_identity = (
        dict(runtime_identity_raw) if isinstance(runtime_identity_raw, Mapping) else {}
    )

    # S68.5 HD3 — runtime_identity.model truthfulness. When the CodeAgent
    # self-reports its model via synthesis_telemetry["model_id"], override the
    # legacy LLM_BACKEND env probe so the persisted runtime_identity matches
    # the actual smolagents OpenAIServerModel binding. Backward-compatible:
    # legacy missions without synthesis_telemetry (or without model_id) keep
    # whatever the legacy probe wrote.
    synth_model_id = synthesis_telemetry.get("model_id") if synthesis_telemetry else None
    if isinstance(synth_model_id, str) and synth_model_id.strip():
        runtime_identity["model"] = synth_model_id.strip()
        derived_backend = _derive_llm_backend(synth_model_id.strip())
        if derived_backend:
            runtime_identity["llm_backend"] = derived_backend

    config_fingerprint = str(telemetry.get("config_fingerprint") or "").strip() or None
    routing_trace = telemetry.get("routing_trace")
    route_summary = (
        dict(routing_trace[-1])
        if (
            isinstance(routing_trace, list)
            and routing_trace
            and isinstance(routing_trace[-1], Mapping)
        )
        else {}
    )
    finding_counts_raw = telemetry.get("finding_counts")
    finding_counts = (
        dict(finding_counts_raw) if isinstance(finding_counts_raw, Mapping) else {}
    )
    source_counts_raw = telemetry.get("source_counts")
    source_counts = (
        dict(source_counts_raw) if isinstance(source_counts_raw, Mapping) else {}
    )
    final_outcome_raw = telemetry.get("final_outcome")
    final_outcome = (
        dict(final_outcome_raw) if isinstance(final_outcome_raw, Mapping) else {}
    )
    telemetry_path = str(telemetry.get("telemetry_artifact_path") or "").strip() or None

    # S84 m04 truth repair — total_tokens. The legacy chain above
    # (telemetry['token_usage'] / state['total_tokens']) has NO writers on
    # the post-S64 path, so the field persisted a constant 0 on every
    # production mission. All-attempt accounting is authoritative when the
    # modern finalize path provides it; otherwise sum the selected live stages.
    # The explicit legacy chain remains the fallback for pre-S64 states.
    attempt_accounting_used = False
    if synthesis_telemetry:
        recovery_raw = synthesis_telemetry.get("recovery")
        recovery = recovery_raw if isinstance(recovery_raw, Mapping) else {}
        attempt_raw = recovery.get("attempt_accounting")
        attempt = attempt_raw if isinstance(attempt_raw, Mapping) else {}
        attempt_usage_raw = attempt.get("token_usage")
        attempt_usage = (
            attempt_usage_raw if isinstance(attempt_usage_raw, Mapping) else {}
        )
        if "total" in attempt_usage:
            try:
                tokens = int(attempt_usage.get("total") or 0)
                attempt_accounting_used = True
            except (TypeError, ValueError):
                attempt_accounting_used = False

    if not attempt_accounting_used and not tokens and synthesis_telemetry:
        synth_usage_raw = synthesis_telemetry.get("token_usage")
        synth_usage = synth_usage_raw if isinstance(synth_usage_raw, Mapping) else {}
        critique_raw = synthesis_telemetry.get("critique_telemetry")
        critique_env = critique_raw if isinstance(critique_raw, Mapping) else {}
        critique_usage_raw = critique_env.get("token_usage")
        critique_usage = (
            critique_usage_raw if isinstance(critique_usage_raw, Mapping) else {}
        )
        try:
            tokens = int(synth_usage.get("total") or 0) + int(
                critique_usage.get("total") or 0
            )
        except (TypeError, ValueError):
            tokens = 0

    # S84 m04 — persist the quality-series record (its inputs were persisted
    # but the record itself never reached the DB; the JSONL artifact is
    # written before the record exists).
    quality_record_raw = (
        telemetry.get("quality_record") if isinstance(telemetry, dict) else None
    )
    if not isinstance(quality_record_raw, Mapping):
        quality_record_raw = synthesis_telemetry.get("quality_record")
    quality_record = (
        dict(quality_record_raw) if isinstance(quality_record_raw, Mapping) else None
    )

    return ExecutionMetadata(
        total_tokens=tokens,
        duration_seconds=duration_seconds,
        convergence_reason=convergence,
        contract_id=(
            str(state.get("mission_contract_id")).strip()
            if state.get("mission_contract_id")
            else None
        ),
        contract_path=(
            str(state.get("mission_contract_path")).strip()
            if state.get("mission_contract_path")
            else None
        ),
        telemetry_path=telemetry_path,
        runtime_identity=runtime_identity,
        config_fingerprint=config_fingerprint,
        route_summary=route_summary,
        finding_counts=finding_counts,
        source_counts=source_counts,
        final_outcome=final_outcome,
        synthesis_telemetry=synthesis_telemetry,
        quality_record=quality_record,
    )


__all__ = [
    "ResultWriter",
    "WriterError",
    "StaleLeaseError",
    "ExecutionMetadata",
    "build_execution_metadata",
]
