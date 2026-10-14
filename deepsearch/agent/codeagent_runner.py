"""S64 CodeAgent synthesis runner — live-retrieval tool surface.

Owns the smolagents ``CodeAgent`` with ``max_steps=40`` and live retrieval.
S93 resolves provider/model/request parameters from shared settings for the
compiler, synthesis, critique and retained-memory finalization. The default
output budget remains 32768; local construction may fall back to at most
16384. See docs/model_configuration.md for the exact provider contract.

- ``web_search``     — single-lane Linkup search.
- ``scholar_search`` — academic backend (OpenAlex + Semantic Scholar). [S79 m01.a]
- ``source_fetch``   — URL → page text with status taxonomy.
- ``url_liveness``   — HEAD/Range probe (also reused by the S66+ critique agent).
- ``scratchpad_*``   — closure-captured key/value store, fresh per run.

The CodeAgent now retrieves cold via these tools rather than reading from an
upstream admitted pool. This is the S64 seam change — the S62 fixed-pool tools
(``source_lookup``, ``list_sources_by_authority_tier``, ``search_sources``) and
their dependency on an injected ``admitted_pool`` are gone. See
``cmos/reports/sprint_64_approach_and_plan.md`` §"Implementation seam" for why
upstream admission is bypassed for the synthesis path.

What carries forward from S62 verbatim: the ``OpenAIServerModel`` bind, the
step-trace serializer, the authority-tier URL classifier (used by S65 retrieval,
not invoked here). The agent loop ceiling has moved (S62=20, S63–S78=25,
S79=40) per S79 m01.d — ``max_steps`` is reviewable per cmos/foundational-docs
guidance, not part of the smolagents bind contract.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse

from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

from deepsearch.domain_policy import domain_matches, normalize_domains
from deepsearch.error_safety import safe_exception_details, safe_exception_text
from deepsearch.mission.scope import AuthoredScope, ScopeAdmission

from .completion_control import CompletionController
from .execution_control import (
    CancellationEvent,
    ExecutionCancelledError,
    current_execution_cancel_event,
    raise_if_execution_cancelled,
    wait_for_execution_cancel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults retained for callers and historical telemetry. The effective request
# configuration lives in deepsearch.llm.runtime, and the step ceiling stays 40.
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "deepseek-flash"
DEFAULT_API_BASE = "https://api.deepseek.com/v1"
DEFAULT_MAX_STEPS = 40
DEFAULT_MAX_TOKENS = 32768
_FALLBACK_MAX_TOKENS = 16384
DEFAULT_TEMPERATURE = 0.0

# Default request timeout; environment overrides apply to all active stages.
DEFAULT_REQUEST_TIMEOUT_S = 300.0
# The lease execution fence owns retries above the client layer. Pinning both
# transports removes transitive-library retry drift from the startup snapshot.
OPENAI_CLIENT_MAX_RETRIES = 0
SMOLAGENTS_RATE_LIMIT_RETRY = False
# smolagents 1.24 ``LogLevel.OFF`` is -1 (ERROR=0, INFO=1). Its AgentLogger
# writes through a Rich console rather than Python logging, so the production
# safety boundary must be the agent's own verbosity level. Generated code,
# tool arguments, and execution output can all contain fetched page bodies or
# signed URLs and must never reach worker stdout/stderr.
SMOLAGENTS_VERBOSITY_OFF = -1

# LEDGER-2 — every source-fetch/liveness attempt is a projection input.  This
# audit is independent of the generic 250-record diagnostic sample below: a
# late failed fetch must never disappear because unrelated search/scratchpad
# calls consumed the diagnostic budget.  TraceLab's schema-version-1
# projection accepts at most 1,000 total entries and fails on a non-zero
# truncation counter.  This is the ledger-record hard ceiling; finalize also
# guards the combined sources + citations + critique + tool-outcome projection
# upper bound because those inputs consume the same TraceLab entry budget.
LEDGER_RECORD_MAX_ENTRIES = 1_000
LEDGER_RECORD_URL_MAX_CHARS = 4_096
# RFC 9110 status-code syntax is three digits. Some bot walls return
# unregistered 6xx-9xx codes; retaining the observed code is safer than
# discarding an otherwise complete paid-run audit.
LEDGER_RECORD_STATUS_CODE_MAX = 999
_LEDGER_RELEVANT_TOOLS = frozenset({"source_fetch", "url_liveness"})
_LEDGER_ERROR_CATEGORIES = frozenset(
    {
        "body_too_large",
        "blocked",
        "cloud_metadata_target",
        "connect",
        "dead",
        "dns",
        "error",
        "excluded_domain",
        "extraction_failed",
        "extraction_failed_or_spa_shell",
        "http_blocked",
        "http_404",
        "http_dead",
        "http_error",
        "invalid",
        "invalid_port",
        "invalid_scheme_or_host",
        "loopback_target",
        "malformed_fetch_response",
        "non_public_address",
        "other",
        "redirect_loop",
        "redirect_missing_location",
        "timeout",
        "too_many_redirects",
        "unexpected",
        "unknown",
        "unsupported_content_type",
        "url_credentials_not_allowed",
    }
)
_HTTP_URL_ADAPTER = TypeAdapter(AnyHttpUrl)

# S89 — transient-generation retry. Production mission
# pedr-retrieval-evaluation-R001 (2026-08-13) died at step 1 on
# AgentGenerationError('Error in generating model output:\nRequest timed
# out.'): three openai-client connect attempts spanning 16.4s, 0 input
# tokens, 0 output tokens, 0 tool calls. Nothing in the worker ever returns
# a mission to 'queued', so `blocked` is terminal and the lease's remaining
# attempts are unreachable — a sub-minute network blip cost the entire
# mission and required a human to re-dispatch. These retries widen the
# outage a run can absorb from ~16s to ~2min (30s + 60s of backoff plus the
# client's own connect attempts). Gated hard by
# _is_retryable_generation_failure so a run that already burned tokens is
# never silently re-run.
SYNTHESIS_RETRY_ATTEMPTS = 3
SYNTHESIS_RETRY_BASE_DELAY_S = 30.0

# Transport/server conditions that a later attempt could plausibly survive.
# Deliberately excludes deterministic failures (auth, malformed request,
# context-length) — retrying those just burns the lease.
_GENERATION_FAILURE_REASONS = frozenset(
    {
        "authentication",
        "balance",
        "connection",
        "context_length",
        "invalid_request",
        "other",
        "overloaded",
        "rate_limit",
        "server",
        "timeout",
    }
)
_TRANSIENT_GENERATION_REASONS = frozenset(
    {"connection", "overloaded", "rate_limit", "server", "timeout"}
)

# S82 m03 (PD135) — stranded-draft recovery convention. When the agent
# exhausts max_steps WITHOUT calling final_answer (the cross-store failure
# class: report assembled in scratchpad but result.output empty / marker-less),
# the runner recovers the best [cite:URL]-bearing scratchpad draft so a
# substantive cited artifact ships instead of an empty stub. The outcome gate
# grades that artifact from its independent quality evidence: a complete cited
# fallback may ship with a warning, while zero-citation or truncated output
# remains validation_failed.
# contract_to_prompt pins this key so the agent keeps its latest COMPLETE draft
# here; recovery prefers it, then falls back to the longest cited value.
SCRATCHPAD_RECOVERY_KEY = "report_draft"

# smolagents 1.24 handles max-step exhaustion with a separate model call whose
# system message replaces (rather than extends) the CodeAgent instructions.
# Without a task-specific override, that last paid call does not see the
# citation/report contract and may emit Python-shaped ``final_answer(...)``
# instead of the raw Markdown expected by finalize.
_MAX_STEPS_FINAL_ANSWER_INSTRUCTIONS = """
The action loop cannot continue. Produce the best complete report possible
from the evidence already present in memory. The task instructions above remain
authoritative for this fallback response.

Return only the raw Markdown report. Do not return Python, XML tags, Markdown
code fences, or a final_answer(...) call. Preserve exact [cite:URL] markers on
supported claims and include the required References section. Do not invent
evidence or URLs; identify an evidence gap when memory cannot support a claim.
""".strip()

_MAX_STEPS_FINAL_ANSWER_POST_MESSAGE = """
Using only the memory above, answer the original task now:
{{task}}

Return only the raw Markdown report, not code or a final_answer(...) wrapper.
""".strip()

# Model-facing retrieval payload budgets. Full readable fetch text is retained
# in the run-local evidence state, while the tool response gets a deterministic
# preview so a large page cannot consume the agent context in one call.
SEARCH_RESULT_SNIPPET_MAX_CHARS = 2_000
SEARCH_TOOL_PAYLOAD_MAX_CHARS = 40_000
SOURCE_FETCH_TOOL_TEXT_MAX_CHARS = 12_000
SOURCE_FETCH_TOOL_PAYLOAD_MAX_CHARS = 16_000

# Model-supplied tool arguments are untrusted. These caps mirror the concrete
# transport defaults and keep every retrieval path machine-inventoryable; the
# per-mission process remains the hard post-lease-loss bound.
SOURCE_FETCH_MAX_TIMEOUT_S = 15.0
SOURCE_FETCH_MAX_REDIRECTS = 5
URL_LIVENESS_MAX_TIMEOUT_S = 5.0
URL_LIVENESS_MAX_REQUESTS = 3
SCHOLAR_BACKEND_TIMEOUT_S = 20.0
SCHOLAR_MAX_BACKENDS = 2


def _bounded_tool_timeout(value: object, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return maximum
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return maximum
    if not math.isfinite(numeric) or numeric <= 0:
        return maximum
    return min(numeric, maximum)


# ---------------------------------------------------------------------------
# Authority-tier derivation (verbatim from S62 spike — carries forward to S65
# retrieval. Not invoked in the S64 synthesis path).
# ---------------------------------------------------------------------------

_HIGH_HOST_PATTERNS: tuple[tuple[str, str], ...] = (
    ("docs.aws.amazon.com", "*"),
    ("aws.amazon.com", "/lambda/"),
    ("aws.amazon.com", "/faqs"),
    ("aws.amazon.com", "/documentation"),
    ("aws.amazon.com", "/serverless"),
    ("cloud.google.com", "/run/docs"),
    ("cloud.google.com", "/functions/docs"),
    ("cloud.google.com", "/cloud-run"),
    ("cloud.google.com", "/docs/"),
    ("vercel.com", "/docs"),
    ("vercel.com", "/templates"),
    ("fly.io", "/docs"),
    ("docs.fly.io", "*"),
    ("railway.app", "/docs"),
    ("docs.railway.app", "*"),
    ("railway.com", "/docs"),
    ("docs.railway.com", "*"),
    ("kubernetes.io", "/docs"),
    ("developers.cloudflare.com", "*"),
    ("developer.mozilla.org", "*"),
    ("playwright.dev", "/docs"),
    ("pptr.dev", "*"),
    ("nodejs.org", "/docs"),
    ("nodejs.org", "/api"),
    # S79 m01.b (PD94) — academic publishers + preprint servers.
    ("arxiv.org", "*"),
    ("acm.org", "*"),
    ("dl.acm.org", "*"),
    ("ieee.org", "*"),
    ("ieeexplore.ieee.org", "*"),
    ("springer.com", "*"),
    ("link.springer.com", "*"),
    ("nature.com", "*"),
    ("sciencedirect.com", "*"),
    ("openalex.org", "*"),
    ("api.openalex.org", "*"),
    ("semanticscholar.org", "*"),
    ("api.semanticscholar.org", "*"),
    ("openreview.net", "*"),
    ("plos.org", "*"),
    ("biorxiv.org", "*"),
    # S79 m01.b (PD94) — IETF + W3C + identity / security standards bodies.
    ("ietf.org", "*"),
    ("tools.ietf.org", "*"),
    ("datatracker.ietf.org", "*"),
    ("rfc-editor.org", "*"),
    ("w3.org", "*"),
    ("oauth.net", "*"),
    ("openid.net", "*"),
    ("nist.gov", "*"),
    ("iso.org", "*"),
)


# S79 m01.b (PD94) — generic .edu TLD shape.
# Higher-education institutional hosts (e.g., cs.stanford.edu, mit.edu) classify
# as ``high`` even when they aren't on the explicit allowlist. The shape is
# evaluated AFTER ``_HIGH_HOST_PATTERNS`` so vendor-specific path matchers
# (``aws.amazon.com/lambda/``) keep their precedence; ``_LOW_HOSTS`` short-
# circuits still run first to keep e.g. forum/aggregator hosts low — there
# are no .edu hosts on ``_LOW_HOSTS`` today, so the precedence order is
# observably equivalent.
_HIGH_HOST_TLD_SUFFIXES: tuple[str, ...] = (".edu",)

_MEDIUM_HOST_PATTERNS: tuple[tuple[str, str], ...] = (
    ("aws.amazon.com", "/blogs"),
    ("aws.amazon.com", "/blog"),
    ("cloud.google.com", "/blog"),
    ("cloud.google.com", "/pricing"),
    ("vercel.com", "/blog"),
    ("vercel.com", "/pricing"),
    ("vercel.com", "/changelog"),
    ("fly.io", "/blog"),
    ("fly.io", "/pricing"),
    ("railway.app", "/blog"),
    ("railway.app", "/pricing"),
    ("railway.com", "/blog"),
    ("railway.com", "/pricing"),
    ("github.com", "*"),
    ("hub.docker.com", "*"),
)

_LOW_HOSTS: frozenset[str] = frozenset(
    {
        "stackoverflow.com",
        "serverfault.com",
        "superuser.com",
        "stackexchange.com",
        "reddit.com",
        "old.reddit.com",
        "news.ycombinator.com",
        "medium.com",
        "dev.to",
        "hashnode.com",
        "hashnode.dev",
        "substack.com",
        "oneuptime.com",
        "checkthat.ai",
        "quora.com",
    }
)


# S78 m01.b (PD80) — generic first-party documentation host shapes.
# Vendor docs typically live under `developers.<X>` / `docs.<X>` / `help.<X>`
# / `developer.<X>` / `support.<X>`. The S62 explicit allowlist
# (_HIGH_HOST_PATTERNS) covers a fixed set of cloud-infrastructure vendors
# but rejects everything else as `low`, including `developers.figma.com`
# and `help.figma.com`. The OODS-PLUGIN-REVIEW-01 forensic deep-dive (S77 m04)
# surfaced this gap. The generalization runs after the explicit allowlist
# (so vendor-specific path matchers like `aws.amazon.com/lambda/` still win)
# and before the medium / fallback paths.
_HIGH_HOST_SUBDOMAIN_PREFIXES: tuple[str, ...] = (
    "developers.",
    "developer.",
    "docs.",
    "help.",
    "support.",
)


def _host_of(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def _path_of(url: str) -> str:
    try:
        return (urlparse(url).path or "").lower()
    except Exception:
        return ""


def _matches_pattern(host: str, path: str, host_pat: str, path_pat: str) -> bool:
    if host != host_pat and not host.endswith("." + host_pat):
        return False
    if path_pat == "*":
        return True
    return path.startswith(path_pat)


def derive_authority_tier(
    url: str,
    *,
    high_authority_domains: Sequence[str] | None = None,
    default_tier: str = "low",
) -> str:
    """Classify a URL as ``"high"``, ``"medium"`` or ``"low"`` per the S62 heuristic.

    S78 m01.b (PD80) extends the classifier with a generic first-party
    documentation host pattern: any host whose first label is
    ``developers.`` / ``developer.`` / ``docs.`` / ``help.`` / ``support.``
    classifies as ``"high"``. The explicit ``_HIGH_HOST_PATTERNS`` allowlist
    runs first so vendor-specific path matchers (e.g. ``aws.amazon.com/lambda/``)
    keep their precedence; only when no explicit pattern matches does the
    generic shape kick in. Existing ``_LOW_HOSTS`` exact matches are still
    short-circuited first to avoid a misclassification on community hosts
    that happen to use a docs-like subdomain.

    S79 m01.b (PD94) adds two more high-tier vectors: 25 explicit
    academic + standards hosts (arXiv/ACM/IEEE/IETF/W3C/NIST/OAuth/OpenID/etc.)
    appended to ``_HIGH_HOST_PATTERNS``, and a generic ``.edu`` TLD suffix
    table (``_HIGH_HOST_TLD_SUFFIXES``) evaluated AFTER the explicit allowlist
    so vendor-specific path matchers retain precedence. Closes the
    universe-of-retrieval gap surfaced by post-S78 missions
    (PLUGIN-BACKEND-AUTH-01 / STRUCTURED-OUTPUT-VIEWER-01) where ``tools.ietf.org``
    and arXiv URLs were structurally biased away from the reranker because
    they fell through to ``low``.
    """

    host = _host_of(url)
    path = _path_of(url)
    if not host:
        return "low"
    if domain_matches(host, high_authority_domains):
        return "high"
    for low_host in _LOW_HOSTS:
        if host == low_host or host.endswith("." + low_host):
            return "low"
    for h, p in _HIGH_HOST_PATTERNS:
        if _matches_pattern(host, path, h, p):
            return "high"
    for suffix in _HIGH_HOST_TLD_SUFFIXES:
        if host.endswith(suffix) and "." in host[: -len(suffix)]:
            return "high"
    for prefix in _HIGH_HOST_SUBDOMAIN_PREFIXES:
        if host.startswith(prefix) and "." in host[len(prefix):]:
            return "high"
    for h, p in _MEDIUM_HOST_PATTERNS:
        if _matches_pattern(host, path, h, p):
            return "medium"
    if host in {"fly.io", "vercel.com", "railway.app", "railway.com"}:
        return "medium"
    return default_tier if default_tier in {"high", "medium", "low"} else "low"


# ---------------------------------------------------------------------------
# Pool normalisation — kept for callers that still build a normalised view of
# upstream sources (e.g., ``codeagent_finalize_node`` until s64-m03 removes the
# upstream-pool harvest). The runner itself no longer consumes a pool.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _PoolEntry:
    """1-based-indexed normalised view of one source."""

    index: int
    url: str
    title_hint: str
    snippet: str
    slots: tuple[str, ...]
    domains: tuple[str, ...]
    authority_tier: str

    def to_lookup_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title_hint": self.title_hint,
            "url": self.url,
            "snippet": self.snippet,
            "slots": list(self.slots),
            "domains": list(self.domains),
            "authority_tier": self.authority_tier,
        }


def _coerce_str_seq(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Mapping):
        out: list[str] = []
        for v in value.values():
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
        return tuple(out)
    if isinstance(value, Iterable):
        out_list: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out_list.append(item.strip())
        return tuple(out_list)
    return ()


def _extract_title(source: Mapping[str, Any]) -> str:
    for key in ("title_hint", "title", "name"):
        v = source.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    url = str(source.get("url") or "").strip()
    if url:
        path = _path_of(url).rstrip("/").rsplit("/", 1)[-1]
        if path:
            return path.replace("-", " ").replace("_", " ")
    return ""


def _extract_snippet(source: Mapping[str, Any]) -> str:
    for key in ("snippet", "summary", "description", "content"):
        v = source.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_slots(source: Mapping[str, Any]) -> tuple[str, ...]:
    tags = source.get("tags")
    if isinstance(tags, Mapping):
        slots = _coerce_str_seq(tags.get("slots"))
        if slots:
            return slots
    return _coerce_str_seq(source.get("target_slots"))


def _extract_domains(source: Mapping[str, Any]) -> tuple[str, ...]:
    tags = source.get("tags")
    if isinstance(tags, Mapping):
        domains = _coerce_str_seq(tags.get("domains"))
        if domains:
            return domains
    url = str(source.get("url") or "").strip()
    host = _host_of(url)
    return (host,) if host else ()


def normalise_pool(
    sources: Sequence[Mapping[str, Any]] | None,
) -> list[_PoolEntry]:
    """Produce a 1-indexed normalised pool from arbitrary admitted source dicts."""

    if not sources:
        return []
    out: list[_PoolEntry] = []
    next_index = 1
    for src in sources:
        if not isinstance(src, Mapping):
            continue
        url = str(src.get("url") or "").strip()
        if not url:
            continue
        existing_index = src.get("index")
        if isinstance(existing_index, int) and existing_index >= 1:
            entry_index = existing_index
        else:
            entry_index = next_index
        out.append(
            _PoolEntry(
                index=entry_index,
                url=url,
                title_hint=_extract_title(src),
                snippet=_extract_snippet(src),
                slots=_extract_slots(src),
                domains=_extract_domains(src),
                authority_tier=derive_authority_tier(url),
            )
        )
        next_index = max(next_index, entry_index) + 1
    return out


# ---------------------------------------------------------------------------
# Tool factory — wraps the four S64 live tools with smolagents @tool and a
# per-call telemetry log. The five callables exposed to the agent are
# web_search, source_fetch, url_liveness, scratchpad_write, scratchpad_read.
# ---------------------------------------------------------------------------


def _canonical_ledger_url(value: Any) -> str:
    """Return TraceLab-compatible canonical HTTP(S) text or raise.

    TraceLab runs the same Pydantic ``AnyHttpUrl`` boundary before source
    identity, hashing, and upsert.  Canonicalizing here makes a long URL one
    stable value across collected-source, citation, and tool-outcome surfaces.
    Userinfo is rejected explicitly because ``AnyHttpUrl`` permits it while
    the Evidence Ledger contract must never persist embedded credentials.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("ledger URL must be non-empty text")
    normalized = value.strip()
    try:
        parsed = urlparse(normalized)
        # Accessing .port raises for malformed ports even when hostname parses.
        parsed.port
    except ValueError as exc:
        raise ValueError("ledger URL has an invalid port") from exc
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("ledger URL userinfo is forbidden")
    try:
        canonical = str(_HTTP_URL_ADAPTER.validate_python(normalized))
    except ValidationError as exc:
        raise ValueError("ledger URL must be canonical HTTP(S)") from exc
    if len(canonical) > LEDGER_RECORD_URL_MAX_CHARS:
        raise ValueError("ledger URL exceeds the 4096-character contract")
    return canonical


def _sanitize_ledger_error_category(value: Any, *, status: str) -> str | None:
    """Map source-tool errors onto a finite secret-safe taxonomy."""

    if status == "ok":
        return None
    if not isinstance(value, str):
        return "unknown"
    category = value.strip().lower()
    if category.startswith("non_public_address:"):
        category = "non_public_address"
    return category if category in _LEDGER_ERROR_CATEGORIES else "unexpected"


def _strict_ledger_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project one call-log row onto the exact LEDGER-2 outcome schema."""

    tool = record.get("tool")
    if tool not in _LEDGER_RELEVANT_TOOLS:
        raise ValueError("call is not a ledger-relevant tool outcome")
    status = record.get("status")
    if status not in {"ok", "error"}:
        raise ValueError("ledger outcome status must be ok or error")
    status_code = record.get("status_code")
    if status_code is not None and (
        not isinstance(status_code, int)
        or isinstance(status_code, bool)
        or not 0 <= status_code <= LEDGER_RECORD_STATUS_CODE_MAX
    ):
        raise ValueError("ledger outcome status_code is outside 0..999")

    projected: dict[str, Any] = {
        "tool": tool,
        "url": _canonical_ledger_url(record.get("url")),
        "status": status,
        "status_code": status_code,
        "error_category": _sanitize_ledger_error_category(
            record.get("error_category"),
            status=status,
        ),
    }
    if tool == "url_liveness":
        alive = record.get("alive")
        if not isinstance(alive, bool):
            raise ValueError("url_liveness ledger outcome requires boolean alive")
        projected["alive"] = alive
    return projected


def _validate_strict_ledger_record_envelope(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an already-produced ledger row without normalizing drift.

    Raw tool-call rows intentionally carry diagnostic-only fields that
    :func:`_strict_ledger_record` projects away. Once the strict envelope is
    produced, however, recovery merging must reject any extra, missing, or
    non-canonical field. Otherwise a malformed recovery row could reach
    TraceLab even though the first-pass producer is strict.
    """

    projected = _strict_ledger_record(record)
    if set(record) != set(projected) or any(
        record[key] != projected[key] for key in projected
    ):
        raise ValueError("ledger record is not the exact strict envelope")
    return projected


def _ledger_records_from_call_log(
    call_log: Sequence[Mapping[str, Any]],
    *,
    max_records: int = LEDGER_RECORD_MAX_ENTRIES,
) -> tuple[list[dict[str, Any]], int]:
    """Build the ordered lossless audit before generic diagnostic sampling.

    Invalid or over-budget relevant rows are not coerced into apparently valid
    evidence.  They increment the loud truncation counter that makes TraceLab
    reject the handoff atomically; valid rows retain call order and duplicates.
    """

    records: list[dict[str, Any]] = []
    truncated = 0
    for raw in call_log:
        if raw.get("tool") not in _LEDGER_RELEVANT_TOOLS:
            continue
        if raw.get("stored_access") is True:
            # Stored reads are provenance-bearing tool calls, not HTTP attempts.
            continue
        try:
            projected = _strict_ledger_record(raw)
        except (TypeError, ValueError):
            truncated += 1
            continue
        if len(records) >= max_records:
            truncated += 1
            continue
        records.append(projected)
    return records, truncated


def _generic_tool_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the legacy diagnostic row without persisting URL userinfo."""

    projected: dict[str, Any] = {}
    for key, value in record.items():
        if key in {"url", "final_url"} and isinstance(value, str):
            try:
                parsed = urlparse(value.strip())
                parsed.port
            except ValueError:
                projected[key] = "[redacted-invalid-url]"
                continue
            if parsed.username is not None or parsed.password is not None:
                projected[key] = "[redacted-invalid-url]"
                continue
        projected[key] = value[:200] if isinstance(value, str) else value
    return projected


@dataclass(slots=True)
class CodeAgentToolBundle:
    """Tools + telemetry surface for one CodeAgent run.

    ``pool`` is retained for callers that read ``pool_size``. With the S64 live
    tools no upstream pool is injected, so it is empty. Authority telemetry is
    intentionally derived from ``sources_collected`` — the production
    collection — rather than that empty legacy surface.

    Discovery, successful readable fetches, fetch attempts, and liveness probes
    have separate stores. ``sources_collected`` is only the deduplicated
    grounding union; failed fetches and diagnostic probes never enter it.
    """

    tools: list[Any]
    pool: list[_PoolEntry] = field(default_factory=list)
    scratchpad: dict[str, str] = field(default_factory=dict)
    call_log: list[dict[str, Any]] = field(default_factory=list)
    sources_collected: list[dict[str, Any]] = field(default_factory=list)
    sources_discovered: list[dict[str, Any]] = field(default_factory=list)
    sources_fetched: list[dict[str, Any]] = field(default_factory=list)
    fetch_attempts: list[dict[str, Any]] = field(default_factory=list)
    probes: list[dict[str, Any]] = field(default_factory=list)
    scope_admission: ScopeAdmission | None = None

    def tool_call_summary(self) -> dict[str, Any]:
        counts: Counter[str] = Counter(c["tool"] for c in self.call_log)
        return {
            "total_calls": len(self.call_log),
            "by_tool": dict(counts),
            "scratchpad_keys": sorted(self.scratchpad.keys()),
        }

    def tool_outcomes(self, max_records: int = 250) -> dict[str, Any]:
        """S84 m04 — per-call OUTCOME telemetry (truth pass).

        The pre-m04 call_log recorded inputs only, so a full Linkup outage
        (every web_search a sentinel error) persisted identically to a thin
        topic (successful searches, few results). The tool wrappers now log
        AFTER the impl with outcome fields (status, error_category,
        result_count, sentinel/circuit_open, status_code); this rolls them
        up per tool and carries a CAPPED per-call record list (inputs
        truncated to 200 chars). ``records_truncated`` makes the cap loud —
        never a silent truncation. The pinned ``tool_call_summary`` shape is
        untouched; this is a new sibling surface.
        """
        by_tool: dict[str, dict[str, Any]] = {}
        for rec in self.call_log:
            name = str(rec.get("tool") or "unknown")
            entry = by_tool.setdefault(
                name,
                {
                    "calls": 0,
                    "ok": 0,
                    "error": 0,
                    "results_total": 0,
                    "circuit_open": 0,
                    "by_category": {},
                },
            )
            entry["calls"] += 1
            status = rec.get("status")
            if status == "error":
                entry["error"] += 1
                category = str(rec.get("error_category") or "unknown")
                entry["by_category"][category] = (
                    entry["by_category"].get(category, 0) + 1
                )
                if rec.get("circuit_open"):
                    entry["circuit_open"] += 1
            elif status == "ok":
                entry["ok"] += 1
            result_count = rec.get("result_count")
            if isinstance(result_count, int):
                entry["results_total"] += result_count
        ledger_records, ledger_records_truncated = _ledger_records_from_call_log(
            self.call_log
        )
        records = [
            _generic_tool_record(rec) for rec in self.call_log[:max_records]
        ]
        return {
            "by_tool": by_tool,
            "records": records,
            "records_truncated": max(0, len(self.call_log) - max_records),
            "ledger_records": ledger_records,
            "ledger_records_truncated": ledger_records_truncated,
        }

    def authority_tier_summary(self) -> dict[str, int]:
        counts: Counter[str] = Counter(
            str(entry.get("authority_tier") or "low")
            for entry in self.sources_collected
        )
        return {
            "high": int(counts.get("high", 0)),
            "medium": int(counts.get("medium", 0)),
            "low": int(counts.get("low", 0)),
        }

    def retrieval_state_summary(self) -> dict[str, int]:
        """Return truthful, non-overlapping retrieval-state counts."""

        return {
            "discovered": len(self.sources_discovered),
            "fetched_readable": len(self.sources_fetched),
            "fetch_attempts": len(self.fetch_attempts),
            "probes": len(self.probes),
            "grounding_sources": len(self.sources_collected),
        }

    @property
    def pool_size(self) -> int:
        return len(self.pool)


def build_codeagent_tools(
    deliverable_spec: Sequence[str] | None = None,
    *,
    excluded_domains: Sequence[str] | None = None,
    high_authority_domains: Sequence[str] | None = None,
    scope_admission: ScopeAdmission | None = None,
    cancel_event: CancellationEvent | None = None,
    completion_controller: CompletionController | None = None,
) -> CodeAgentToolBundle:
    """Build the live retrieval tool surface.

    Tools (insertion order is the surface order presented to the agent):
    ``web_search``, ``source_fetch``, ``url_liveness``, ``scratchpad_write``,
    ``scratchpad_read``, ``scholar_search`` (S79 m01.a), ``completeness_check``
    (S79 m02.a — appended at the end so the position-based test contracts
    under ``tests/unit/test_codeagent_runner.py`` keep their ``bundle.tools[0..2]``
    invariants).

    Args:
        deliverable_spec: Optional list of deliverable bullet entries from
            the mission contract (S79 m02.a). When supplied, the
            ``completeness_check`` tool's closure-captures it and the agent
            can call ``completeness_check(draft_markdown)`` to surface the
            satisfied/missing deliverables in its current draft. When
            ``None``, ``completeness_check`` returns the defensive empty
            shape (no false-positive shortfall).
        excluded_domains: Mission-local domains rejected by every live
            retrieval boundary, including subdomains and redirect targets.
        high_authority_domains: Mission-local primary domains promoted before
            authority multiplication. The tuple is closure-scoped to one run.

    Each call is recorded in a closure-captured ``call_log`` so the runner can
    surface a tool-call summary in its telemetry. The scratchpad state is held
    in its own closure-captured dict shared between scratchpad_write and
    scratchpad_read. Tool implementations live in ``deepsearch.agent.tools.*``;
    this factory only wires them through the smolagents ``@tool`` decorator.
    """

    from smolagents import tool  # imported here so the module imports without smolagents

    from .tools.completeness_check import completeness_check as _completeness_check_impl
    from .tools.scholar_search import scholar_search as _scholar_search_impl
    from .tools.scratchpad import build_scratchpad
    from .tools.source_fetch import source_fetch as _source_fetch_impl
    from .tools.source_passages import select_source_passage, validate_passage_request
    from .tools.url_liveness import url_liveness as _url_liveness_impl
    from .tools.web_search import web_search as _web_search_impl

    # Snapshot the spec into the closure so subsequent state mutations don't
    # change what the agent's completeness_check call sees.
    bound_deliverable_spec: tuple[str, ...] = (
        tuple(s for s in deliverable_spec if isinstance(s, str))
        if deliverable_spec
        else ()
    )
    admission = scope_admission or ScopeAdmission(AuthoredScope())
    bound_excluded_domains = normalize_domains(excluded_domains)
    bound_high_authority_domains = normalize_domains(high_authority_domains)
    bound_cancel_event = (
        cancel_event
        if cancel_event is not None
        else current_execution_cancel_event()
    )

    scratchpad_bundle = build_scratchpad()
    call_log: list[dict[str, Any]] = []
    sources_collected: list[dict[str, Any]] = []
    sources_discovered: list[dict[str, Any]] = []
    sources_fetched: list[dict[str, Any]] = []
    fetch_attempts: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    _sources_index: dict[str, int] = {}
    _discovered_index: dict[str, int] = {}
    _fetched_index: dict[str, int] = {}
    # Exact request/final-redirect identities only; scoped to this attempt.
    # A successful explicit refresh replaces the version, even if shorter.
    fetched_cache: dict[str, dict[str, Any]] = {}

    def _log(tool_name: str, **fields: Any) -> None:
        call_log.append({"tool": tool_name, **fields})

    def _retrieval_refusal(tool_name: str, gap: str, url: str = "") -> str | None:
        if completion_controller is None:
            return None
        reason = completion_controller.allow_retrieval(gap, scratchpad_bundle.state)
        if reason:
            _log(tool_name, url=url, status="error", error_category=reason,
                 status_code=0, network_requested=False, stored_access=True, result_count=0)
        return reason

    def _int_value(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _accumulate_source(
        entry: dict[str, Any],
        *,
        target: list[dict[str, Any]] | None = None,
        index: dict[str, int] | None = None,
        replace_body: bool = False,
    ) -> None:
        """Append or merge evidence by URL without degrading readable content."""

        target = sources_collected if target is None else target
        index = _sources_index if index is None else index
        url = str(entry.get("url") or "").strip()
        if not url or domain_matches(url, bound_excluded_domains):
            return
        if entry.get("authority_tier") not in {"high", "medium", "low"}:
            entry["authority_tier"] = derive_authority_tier(
                url,
                high_authority_domains=bound_high_authority_domains,
            )
        if url in index:
            existing = target[index[url]]
            existing_quality = 2 if existing.get("content_quality") == "readable_body" else 1
            incoming_quality = 2 if entry.get("content_quality") == "readable_body" else 1
            existing_content_chars = max(
                len(str(existing.get("body") or "")),
                len(str(existing.get("snippet") or "")),
            )
            incoming_content_chars = max(
                len(str(entry.get("body") or "")),
                len(str(entry.get("snippet") or "")),
            )
            for key, value in entry.items():
                if value is None or value == "":
                    continue
                if not replace_body and key in {
                    "body", "snippet", "body_source", "content_quality"
                }:
                    if incoming_quality < existing_quality or (
                        incoming_quality == existing_quality
                        and incoming_content_chars < existing_content_chars
                    ):
                        continue
                if (
                    key == "authority_tier"
                    and existing.get("score") is not None
                    and entry.get("score") is None
                ):
                    # Fetch/probe classification is not a scoring event. Keep
                    # the tier that produced the stored rerank score so
                    # telemetry cannot drift after a scholar result is fetched.
                    continue
                existing[key] = value
        else:
            index[url] = len(target)
            target.append(dict(entry))

    def _bounded_search_payload(
        results: Any,
    ) -> tuple[list[dict[str, Any]], int, int, int]:
        """Bound search results returned to the model, preserving list shape."""

        if not isinstance(results, list):
            results = [
                {
                    "error": "malformed_payload",
                    "message": f"expected list, got {type(results).__name__}",
                    "title": "",
                    "url": "",
                    "snippet": "",
                }
            ]
        if any(not isinstance(item, dict) for item in results):
            results = [
                {
                    "error": "malformed_payload",
                    "message": "result list contains a non-mapping item",
                    "title": "",
                    "url": "",
                    "snippet": "",
                }
            ]
        bounded: list[dict[str, Any]] = []
        payload_chars = 0
        fields_truncated = 0
        records_dropped = 0
        for position, item in enumerate(results):
            copy = dict(item)
            for key, cap in (
                ("title", 500),
                ("snippet", SEARCH_RESULT_SNIPPET_MAX_CHARS),
                ("message", 1_000),
            ):
                value = copy.get(key)
                if isinstance(value, str) and len(value) > cap:
                    copy[key] = value[:cap]
                    fields_truncated += 1
            candidate = [*bounded, copy]
            candidate_chars = len(
                json.dumps(candidate, ensure_ascii=False, sort_keys=True, default=str)
            )
            if candidate_chars > SEARCH_TOOL_PAYLOAD_MAX_CHARS:
                records_dropped = len(results) - position
                break
            bounded.append(copy)
            payload_chars = candidate_chars
        return bounded, payload_chars, fields_truncated, records_dropped

    def _scope_search_refusal(tool_name: str, query: str) -> list[dict[str, Any]] | None:
        reason = admission.search_refusal(tool_name, query)
        if reason:
            admission.reject("", tool_name, reason)
            _log(tool_name, status="error", error_category=reason,
                 network_requested=False, result_count=0)
            return [{"error": reason, "title": "", "url": "", "snippet": "",
                     "message": admission.scope.prompt()}]
        return None

    def _admit_search_results(results: Any, surface: str) -> Any:
        if not isinstance(results, list):
            return results
        admitted: list[Any] = []
        for item in results:
            if not isinstance(item, dict) or item.get("error"):
                admitted.append(item)
                continue
            url = str(item.get("url") or "")
            reason = admission.refusal(url)
            if reason:
                admission.reject(url, surface, reason)
            elif not domain_matches(url, bound_excluded_domains):
                admitted.append(item)
        return admitted

    @tool
    def web_search(
        query: str, max_results: int = 20, gap: str = "",
    ) -> list[dict[str, Any]]:
        """Search the web via Linkup (single-lane). Returns up to max_results
        items shaped ``{title, url, snippet, score, authority_tier}``. On
        rate-limit / 5xx / malformed payload, returns a single-item sentinel
        list with an ``error`` key the agent can branch on.

        Args:
            query: Natural-language or keyword search query.
            max_results: Soft cap on returned items (1..50).
            gap: Specific missing evidence/content target for bounded assembly retrieval.
        """
        raise_if_execution_cancelled(bound_cancel_event)
        scope_refusal = _scope_search_refusal("web_search", query)
        if scope_refusal is not None:
            return scope_refusal
        if admission.scope.allowed_domains:
            sites = " OR ".join("site:" + d for d in admission.scope.allowed_domains)
            query = f"({query}) ({sites})"
        refused = _retrieval_refusal("web_search", gap)
        if refused:
            return [{"error": refused, "title": "", "url": "", "snippet": "",
                     "message": (completion_controller.notice()
                                 if completion_controller is not None else refused)}]
        if bound_excluded_domains or bound_high_authority_domains:
            results = _web_search_impl(
                query,
                max_results,
                excluded_domains=bound_excluded_domains,
                high_authority_domains=bound_high_authority_domains,
            )
        else:
            results = _web_search_impl(query, max_results)
        raise_if_execution_cancelled(bound_cancel_event)
        excluded_result_count = 0
        if isinstance(results, list) and bound_excluded_domains:
            filtered: list[dict[str, Any]] = []
            for item in results:
                if isinstance(item, dict) and domain_matches(
                    item.get("url"), bound_excluded_domains
                ):
                    excluded_result_count += 1
                    continue
                filtered.append(item)
            results = filtered
        results = _admit_search_results(results, "web_search")
        results, payload_chars, payload_fields_truncated, payload_records_dropped = (
            _bounded_search_payload(results)
        )
        results = [item for item in results if item.get("error") or admission.admit(
            str(item.get("url") or ""), "web_search"
        )]
        # S84 m04 — log AFTER the impl so the record carries the OUTCOME:
        # a sentinel-error search must persist distinguishably from a
        # successful-but-thin one.
        sentinel = (
            isinstance(results, list)
            and len(results) == 1
            and isinstance(results[0], dict)
            and bool(results[0].get("error"))
        )
        if sentinel:
            category = str(results[0].get("error") or "unknown")
            _log(
                "web_search",
                query=query,
                max_results=max_results,
                status="error",
                error_category=category,
                sentinel=True,
                circuit_open=category == "circuit_open",
                result_count=0,
                excluded_result_count=excluded_result_count,
                payload_chars=payload_chars,
                payload_budget_chars=SEARCH_TOOL_PAYLOAD_MAX_CHARS,
                estimated_payload_tokens=(payload_chars + 3) // 4,
                payload_fields_truncated=payload_fields_truncated,
                payload_records_dropped=payload_records_dropped,
            )
        else:
            _log(
                "web_search",
                query=query,
                max_results=max_results,
                status="ok",
                error_category=None,
                sentinel=False,
                circuit_open=False,
                result_count=len(results) if isinstance(results, list) else 0,
                excluded_result_count=excluded_result_count,
                payload_chars=payload_chars,
                payload_budget_chars=SEARCH_TOOL_PAYLOAD_MAX_CHARS,
                estimated_payload_tokens=(payload_chars + 3) // 4,
                payload_fields_truncated=payload_fields_truncated,
                payload_records_dropped=payload_records_dropped,
            )
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                if item.get("error"):
                    continue
                url = str(item.get("url") or "").strip()
                if not url:
                    continue
                snippet_text = item.get("snippet") or ""
                _accumulate_source(
                    {
                        "url": url,
                        "title": item.get("title") or "",
                        "snippet": snippet_text,
                        "body": "",
                        "body_source": "search_snippet",
                        "content_quality": "snippet",
                        "score": item.get("score"),
                        "authority_tier": item.get("authority_tier") or "",
                    },
                    target=sources_discovered,
                    index=_discovered_index,
                )
                _accumulate_source(
                    {
                        "url": url,
                        "title": item.get("title") or "",
                        "snippet": snippet_text,
                        "body": "",
                        "body_source": "search_snippet",
                        "content_quality": "snippet",
                        "score": item.get("score"),
                        "authority_tier": item.get("authority_tier") or "",
                    }
                )
        return results

    @tool
    def source_fetch(
        url: str,
        timeout_s: float = SOURCE_FETCH_MAX_TIMEOUT_S,
        start: int = 0,
        length: int = SOURCE_FETCH_TOOL_TEXT_MAX_CHARS,
        query: str = "",
        refresh: bool = False,
        gap: str = "",
    ) -> dict[str, Any]:
        """Fetch once, then find/read bounded passages from the stored source.

        Returns ``{text, status, status_code, fetched_at, final_url}`` where
        ``status`` is one of ``ok | blocked | dead | timeout | error``.

        Args:
            url: Absolute URL to fetch.
            timeout_s: Per-request timeout in seconds (default 15s).
            start: Zero-based Unicode character offset into the stored body.
            length: Maximum passage characters (1..12000).
            query: Literal case-insensitive find text (up to 200 characters).
                Returns at most five match offsets and context around the first.
                Find/range reads require a prior successful fetch and never fetch.
            refresh: Fetch a new version explicitly; default reuses the collected
                snapshot and its original fetched_at. Incompatible with find/range.
            gap: Specific missing evidence/content target for bounded assembly retrieval.
        """
        raise_if_execution_cancelled(bound_cancel_event)
        timeout_s = _bounded_tool_timeout(timeout_s, SOURCE_FETCH_MAX_TIMEOUT_S)
        local_error = ""
        try:
            validate_passage_request(start, length, query)
            if not isinstance(refresh, bool) or (refresh and (start or query)):
                raise ValueError("invalid_refresh")
        except ValueError as exc:
            local_error = str(exc)
        if local_error:
            _log("source_fetch", url=url, status="error", error_category=local_error,
                 status_code=0, network_requested=False, stored_access=True, result_count=0)
            return {"text": "", "status": "error", "error": local_error,
                    "status_code": 0, "final_url": url, "network_requested": False}
        cached = fetched_cache.get(url) if not refresh else None
        reason = (
            "scope_source_not_allowed" if not admission.scope.allows(url)
            else admission.refusal(str(cached.get("final_url") or url) if cached else url)
        )
        if reason:
            admission.reject(url, "source_fetch", reason)
            _log("source_fetch", url=url, status="error", error_category=reason,
                 status_code=0, network_requested=False, stored_access=True, result_count=0)
            return {"text": "", "status": "blocked", "error": reason,
                    "status_code": 0, "final_url": url, "network_requested": False}
        if cached is None and not (start or query):
            refused = _retrieval_refusal("source_fetch", gap, url)
            if refused:
                return {"text": "", "status": "error", "status_code": 0,
                        "error": refused, "final_url": url, "network_requested": False}
        network_requested = False
        stored_access = cached is not None or bool(start or query)
        out: dict[str, Any]
        if domain_matches(url, bound_excluded_domains):
            out = {
                "text": "",
                "status": "blocked",
                "status_code": 0,
                "fetched_at": "",
                "final_url": url,
                "error": "excluded_domain",
            }
        elif cached is not None:
            out = cached
        elif start or query:
            out = {"text": "", "status": "error", "status_code": 0,
                   "final_url": url, "error": "missing_stored_body"}
        elif bound_excluded_domains or scope_admission is not None:
            network_requested = True
            out = _source_fetch_impl(
                url,
                timeout_s,
                excluded_domains=bound_excluded_domains,
                scope_admission=admission,
            )
        else:
            network_requested = True
            out = _source_fetch_impl(url, timeout_s)
        raise_if_execution_cancelled(bound_cancel_event)
        if isinstance(out, dict) and domain_matches(
            out.get("final_url"), bound_excluded_domains
        ):
            out = {
                "text": "",
                "status": "blocked",
                "status_code": _int_value(out.get("status_code")),
                "fetched_at": out.get("fetched_at") or "",
                "final_url": str(out.get("final_url") or url),
                "error": "excluded_domain",
            }
        if not isinstance(out, dict):
            _log(
                "source_fetch",
                url=url,
                status="error",
                error_category="malformed_fetch_response",
                network_requested=network_requested,
                status_code=0,
                result_count=0,
                payload_chars=0,
                payload_budget_chars=SOURCE_FETCH_TOOL_PAYLOAD_MAX_CHARS,
                estimated_payload_tokens=0,
                payload_truncated_chars=0,
            )
            return {
                "text": "",
                "status": "error",
                "status_code": 0,
                "fetched_at": "",
                "final_url": str(url or ""),
                "error": "malformed_fetch_response",
            }

        if out.get("status") == "ok" and out.get("text"):
            if not admission.admit(str(out.get("final_url") or url), "source_fetch"):
                out = {"text": "", "status": "blocked", "status_code": 0,
                       "final_url": str(out.get("final_url") or url),
                       "error": "scope_source_not_allowed"}
        if out.get("status") != "ok" or not out.get("text"):
            admission.unavailable[url] = str(out.get("error") or out.get("status") or "empty")
        else:
            admission.unavailable.pop(url, None)
        fetch_status = str(out.get("status") or "error")
        fetched_text = out.get("text")
        fetched_chars = len(fetched_text) if isinstance(fetched_text, str) else 0
        public_out = {
            key: out[key]
            for key in (
                "text",
                "status",
                "status_code",
                "fetched_at",
                "final_url",
                "error",
                "content_type",
                "bytes_read",
                "text_truncated",
            )
            if key in out
        }
        public_out.update(
            {
                "text": (
                    fetched_text[:SOURCE_FETCH_TOOL_TEXT_MAX_CHARS]
                    if isinstance(fetched_text, str)
                    else ""
                ),
                "status": fetch_status,
                "status_code": _int_value(out.get("status_code")),
                "fetched_at": str(out.get("fetched_at") or "")[:100],
                "final_url": str(out.get("final_url") or url or ""),
                "text_chars_total": fetched_chars,
            }
        )
        body_sha256 = hashlib.sha256(str(fetched_text or "").encode()).hexdigest()
        source_url = str(out.get("final_url") or url)
        public_out.update({
            "source_id": hashlib.sha256(source_url.encode()).hexdigest(),
            "body_sha256": body_sha256,
            "body_complete": fetch_status == "ok" and not bool(out.get("text_truncated")),
            "cache_hit": cached is not None,
            "network_requested": network_requested,
        })
        if fetch_status == "ok":
            try:
                public_out.update(
                    select_source_passage(str(fetched_text or ""), start, length, query)
                )
            except ValueError as exc:
                public_out.update({"text": "", "status": "error", "error": str(exc)})

        if "error" in public_out:
            public_out["error"] = str(public_out["error"] or "")[:500]
        if "content_type" in public_out:
            public_out["content_type"] = str(public_out["content_type"] or "")[:200]

        public_out["text_chars_returned"] = len(public_out["text"])
        public_out["text_truncated_for_tool"] = (
            public_out.get("range_start", 0) > 0
            or fetched_chars > public_out["text_chars_returned"]
        )
        payload_chars = len(
            json.dumps(public_out, ensure_ascii=False, sort_keys=True, default=str)
        )
        while payload_chars > SOURCE_FETCH_TOOL_PAYLOAD_MAX_CHARS and public_out["text"]:
            excess = payload_chars - SOURCE_FETCH_TOOL_PAYLOAD_MAX_CHARS
            public_out["text"] = public_out["text"][: max(0, len(public_out["text"]) - excess)]
            public_out["text_chars_returned"] = len(public_out["text"])
            public_out["text_truncated_for_tool"] = True
            if "range_start" in public_out:
                public_out["range_end"] = public_out["range_start"] + len(public_out["text"])
                public_out["remaining_chars"] = fetched_chars - public_out["range_end"]
                public_out["next_start"] = (
                    public_out["range_end"] if public_out["remaining_chars"] else None
                )
            payload_chars = len(
                json.dumps(public_out, ensure_ascii=False, sort_keys=True, default=str)
            )
        if "range_start" in public_out:
            public_out["range_end"] = public_out["range_start"] + len(public_out["text"])
            public_out["remaining_chars"] = fetched_chars - public_out["range_end"]
            public_out["next_start"] = (
                public_out["range_end"] if public_out["remaining_chars"] else None
            )
        public_chars = int(public_out["text_chars_returned"])
        _log(
            "source_fetch",
            url=url,
            status="ok" if public_out["status"] == "ok" else "error",
            error_category=public_out.get("error"),
            network_requested=network_requested,
            stored_access=stored_access,
            cache_hit=cached is not None,
            source_id=public_out["source_id"],
            body_sha256=body_sha256,
            range_start=public_out.get("range_start"),
            range_end=public_out.get("range_end"),
            query=query,
            matches=public_out.get("matches", []),
            body_complete=public_out["body_complete"],
            # Preserve the typed boundary value for the strict Ledger audit;
            # malformed bool/string codes must fail loudly, not coerce to int.
            status_code=out.get("status_code"),
            result_count=fetched_chars,
            payload_chars=payload_chars,
            payload_budget_chars=SOURCE_FETCH_TOOL_PAYLOAD_MAX_CHARS,
            estimated_payload_tokens=(payload_chars + 3) // 4,
            payload_truncated_chars=max(0, fetched_chars - public_chars),
        )
        final_url = str(out.get("final_url") or url or "").strip()
        if not stored_access:
            fetch_attempts.append(
                {
                    "url": str(url or ""),
                    "final_url": final_url,
                    "status": fetch_status,
                    "status_code": _int_value(out.get("status_code")),
                    "error": out.get("error"),
                    "readable_chars": fetched_chars if fetch_status == "ok" else 0,
                }
            )
        if network_requested and final_url and fetch_status == "ok" and fetched_chars > 0:
            # Only actual validated redirects establish aliases. Never infer
            # equivalence by removing www or changing scheme/origin.
            for alias, previous in list(fetched_cache.items()):
                if str(previous.get("final_url")) == final_url:
                    fetched_cache[alias] = out
            fetched_cache[url] = out
            fetched_cache[final_url] = out
            body_text = str(fetched_text)
            entry = {
                "url": final_url,
                "title": "",
                "snippet": body_text[:300],
                "body": body_text,
                "body_source": "source_fetch",
                "content_quality": "readable_body",
                "fetched_at": out.get("fetched_at") or "",
                "content_type": out.get("content_type") or "",
                "body_chars": fetched_chars,
                "body_truncated": bool(out.get("text_truncated")),
                "score": None,
                "authority_tier": derive_authority_tier(
                    final_url,
                    high_authority_domains=bound_high_authority_domains,
                ),
            }
            _accumulate_source(
                entry,
                target=sources_fetched,
                index=_fetched_index,
                replace_body=True,
            )
            _accumulate_source(entry, replace_body=True)
        return public_out

    @tool
    def url_liveness(
        url: str,
        timeout_s: float = URL_LIVENESS_MAX_TIMEOUT_S,
        gap: str = "",
    ) -> dict[str, Any]:
        """Probe a URL with HEAD; GET fallbacks on 405 (Range) and other 4xx
        (browser User-Agent, so a reported 4xx is GET-confirmed).

        Returns ``{alive, status_code, final_url, error}`` where ``error`` is
        ``None`` or a transport category (``timeout | connect | invalid |
        other``). Useful before paying the cost of a full ``source_fetch``
        and for citation verification.

        Args:
            url: Absolute URL to probe.
            timeout_s: Per-request timeout in seconds (default 5s).
            gap: Specific missing evidence/content target for bounded assembly retrieval.
        """
        raise_if_execution_cancelled(bound_cancel_event)
        refused = _retrieval_refusal("url_liveness", gap, url)
        if refused:
            return {"alive": None, "status_code": 0, "final_url": url, "error": refused}
        timeout_s = _bounded_tool_timeout(timeout_s, URL_LIVENESS_MAX_TIMEOUT_S)
        out: dict[str, Any]
        scope_reason = admission.refusal(url)
        if scope_reason:
            admission.reject(url, "url_liveness", scope_reason)
            out = {"alive": None, "status_code": 0, "final_url": url, "error": scope_reason}
        elif domain_matches(url, bound_excluded_domains):
            out = {
                "alive": False,
                "status_code": 0,
                "final_url": url,
                "error": "excluded_domain",
            }
        elif bound_excluded_domains or scope_admission is not None:
            out = _url_liveness_impl(
                url,
                timeout_s,
                excluded_domains=bound_excluded_domains,
                scope_admission=admission,
            )
        else:
            out = _url_liveness_impl(url, timeout_s)
        raise_if_execution_cancelled(bound_cancel_event)
        if isinstance(out, dict) and domain_matches(
            out.get("final_url"), bound_excluded_domains
        ):
            out = {
                "alive": False,
                "status_code": _int_value(out.get("status_code")),
                "final_url": str(out.get("final_url") or url),
                "error": "excluded_domain",
            }
        # S84 m04 — outcome-bearing log (alive verdict + transport category).
        if isinstance(out, dict):
            probe_error = out.get("error")
            _log(
                "url_liveness",
                url=url,
                status="ok" if probe_error is None else "error",
                error_category=probe_error if isinstance(probe_error, str) else None,
                status_code=out.get("status_code"),
                alive=out.get("alive"),
            )
        else:
            _log(
                "url_liveness",
                url=url,
                status="error",
                error_category="other",
                status_code=0,
                alive=False,
            )
        if isinstance(out, dict):
            final_url = str(out.get("final_url") or url or "").strip()
            probes.append(
                {
                    "url": str(url or ""),
                    "final_url": final_url,
                    "alive": bool(out.get("alive")),
                    "status_code": _int_value(out.get("status_code")),
                    "error": out.get("error"),
                }
            )
        return out

    @tool
    def scratchpad_write(key: str, value: str) -> str:
        """Persist a value under a key in the agent's scratchpad. Overwrites prior value.

        Args:
            key: Scratchpad key (e.g. "aws_lambda_notes", "matrix_draft_v1").
            value: String value to store.
        """
        raise_if_execution_cancelled(bound_cancel_event)
        result = scratchpad_bundle.write(key, value)
        if completion_controller is not None and key == "report_draft":
            completion_controller.observe_draft(value)
        _log(
            "scratchpad_write",
            status="ok",
            key=key,
            value_len=len(value) if isinstance(value, str) else 0,
        )
        return result

    @tool
    def scratchpad_read(key: str) -> str:
        """Read a previously-written scratchpad value. Returns empty string if absent.

        Args:
            key: Scratchpad key to read.
        """
        raise_if_execution_cancelled(bound_cancel_event)
        result = scratchpad_bundle.read(key)
        _log("scratchpad_read", key=key, hit=bool(result), status="ok")
        return result

    @tool
    def scholar_search(
        query: str, max_results: int = 20, gap: str = "",
    ) -> list[dict[str, Any]]:
        """Search academic literature (OpenAlex primary, Semantic Scholar fallback).

        Use ``scholar_search`` for primary literature, RFCs, and standards
        (arXiv, ACM, IEEE, IETF, W3C, NIST, OAuth, OpenID, etc.). Use
        ``web_search`` for product docs, blogs, vendor reference, and forums.

        Returns up to ``max_results`` items shaped
        ``{title, url, snippet, score, authority_tier}``. ``authority_tier``
        defaults to ``high`` for academic results unless the URL is a known
        low-tier host. On rate-limit / 5xx / malformed payload from BOTH
        backends, returns a single-item sentinel list with an ``error`` key
        the agent can branch on.

        Args:
            query: Natural-language or keyword search query.
            max_results: Soft cap on returned items (1..50).
            gap: Specific missing evidence/content target for bounded assembly retrieval.
        """
        raise_if_execution_cancelled(bound_cancel_event)
        scope_refusal = _scope_search_refusal("scholar_search", query)
        if scope_refusal is not None:
            return scope_refusal
        refused = _retrieval_refusal("scholar_search", gap)
        if refused:
            return [{"error": refused, "title": "", "url": "", "snippet": "",
                     "message": (completion_controller.notice()
                                 if completion_controller is not None else refused)}]
        if bound_excluded_domains or bound_high_authority_domains:
            results = _scholar_search_impl(
                query,
                max_results,
                excluded_domains=bound_excluded_domains,
                high_authority_domains=bound_high_authority_domains,
            )
        else:
            results = _scholar_search_impl(query, max_results)
        raise_if_execution_cancelled(bound_cancel_event)
        scholar_excluded_count = 0
        if isinstance(results, list) and bound_excluded_domains:
            scholar_filtered: list[dict[str, Any]] = []
            for item in results:
                if isinstance(item, dict) and domain_matches(
                    item.get("url"), bound_excluded_domains
                ):
                    scholar_excluded_count += 1
                    continue
                scholar_filtered.append(item)
            results = scholar_filtered
        results = _admit_search_results(results, "scholar_search")
        results, payload_chars, payload_fields_truncated, payload_records_dropped = (
            _bounded_search_payload(results)
        )
        results = [item for item in results if item.get("error") or admission.admit(
            str(item.get("url") or ""), "scholar_search"
        )]
        # S84 m04 — outcome-bearing log (same sentinel taxonomy as web_search).
        scholar_sentinel = (
            isinstance(results, list)
            and len(results) == 1
            and isinstance(results[0], dict)
            and bool(results[0].get("error"))
        )
        if scholar_sentinel:
            scholar_category = str(results[0].get("error") or "unknown")
            _log(
                "scholar_search",
                query=query,
                max_results=max_results,
                status="error",
                error_category=scholar_category,
                sentinel=True,
                circuit_open=scholar_category == "circuit_open",
                result_count=0,
                excluded_result_count=scholar_excluded_count,
                payload_chars=payload_chars,
                payload_budget_chars=SEARCH_TOOL_PAYLOAD_MAX_CHARS,
                estimated_payload_tokens=(payload_chars + 3) // 4,
                payload_fields_truncated=payload_fields_truncated,
                payload_records_dropped=payload_records_dropped,
            )
        else:
            _log(
                "scholar_search",
                query=query,
                max_results=max_results,
                status="ok",
                error_category=None,
                sentinel=False,
                circuit_open=False,
                result_count=len(results) if isinstance(results, list) else 0,
                excluded_result_count=scholar_excluded_count,
                payload_chars=payload_chars,
                payload_budget_chars=SEARCH_TOOL_PAYLOAD_MAX_CHARS,
                estimated_payload_tokens=(payload_chars + 3) // 4,
                payload_fields_truncated=payload_fields_truncated,
                payload_records_dropped=payload_records_dropped,
            )
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                if item.get("error"):
                    continue
                url = str(item.get("url") or "").strip()
                if not url:
                    continue
                snippet_text = item.get("snippet") or ""
                _accumulate_source(
                    {
                        "url": url,
                        "title": item.get("title") or "",
                        "snippet": snippet_text,
                        "body": "",
                        "body_source": "scholar_snippet",
                        "content_quality": "snippet",
                        "score": item.get("score"),
                        "authority_tier": item.get("authority_tier") or "",
                    },
                    target=sources_discovered,
                    index=_discovered_index,
                )
                _accumulate_source(
                    {
                        "url": url,
                        "title": item.get("title") or "",
                        "snippet": snippet_text,
                        "body": "",
                        "body_source": "scholar_snippet",
                        "content_quality": "snippet",
                        "score": item.get("score"),
                        "authority_tier": item.get("authority_tier") or "",
                    }
                )
        return results

    @tool
    def completeness_check(draft_markdown: str) -> dict[str, Any]:
        """Surface which named deliverables in the mission spec are present
        vs missing in the current draft. Telemetry-only — the agent decides
        whether to keep retrieving or finalize. (S79 m02.a, PD97.)

        Returns artifact/content assessments and satisfied/missing requirements.
        For word-bounded missions, report_word_budget estimates the full rendered
        report including its wrapper. Revise over/under-length drafts and recheck
        before final_answer; the final persisted artifact remains authoritative.
        A null ``shortfall_score`` means unassessed; format labels do not require
        matching headings. Citation presence does not establish claim support.

        Args:
            draft_markdown: The current draft as a single Markdown string.
        """
        raise_if_execution_cancelled(bound_cancel_event)
        _log(
            "completeness_check",
            draft_len=len(draft_markdown) if isinstance(draft_markdown, str) else 0,
            status="ok",
        )
        assessment = _completeness_check_impl(draft_markdown, list(bound_deliverable_spec))
        if completion_controller is not None:
            word_budget = completion_controller.check_report_word_budget(draft_markdown)
            if word_budget:
                assessment["report_word_budget"] = word_budget
        return assessment

    return CodeAgentToolBundle(
        tools=[
            web_search,
            source_fetch,
            url_liveness,
            scratchpad_write,
            scratchpad_read,
            scholar_search,
            completeness_check,
        ],
        pool=[],
        scratchpad=scratchpad_bundle.state,
        call_log=call_log,
        sources_collected=sources_collected,
        sources_discovered=sources_discovered,
        sources_fetched=sources_fetched,
        fetch_attempts=fetch_attempts,
        probes=probes,
        scope_admission=admission,
    )


# ---------------------------------------------------------------------------
# Model bind helper.
# ---------------------------------------------------------------------------


class CodeAgentRunnerError(RuntimeError):
    """Raised when the CodeAgent runner cannot complete synthesis."""


class SourceCollectionValidationError(CodeAgentRunnerError, ValueError):
    """The collected-source projection cannot be proven complete and canonical."""


_SOURCE_COLLECTION_DROP_REASON_CODES = (
    "not_object",
    "invalid_url",
    "invalid_title",
    "invalid_snippet",
    "invalid_alive",
    "invalid_record",
    "canonical_metadata_conflict",
)


def _canonicalize_sources_collected(
    sources: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the strict TraceLab-compatible collected-source envelope.

    TraceLab canonicalizes HTTP URLs before it merges ``sources_collected``.
    Canonical-equivalent records may enrich a missing title, snippet, or
    liveness verdict, but conflicting non-empty values make the entire
    projection invalid.  Apply the same rule before persistence so a raw URL
    spelling cannot become a second source identity or make a terminal outbox
    delivery deterministically dead-letter.  Invalid URLs and conflicts fail
    with structural diagnostics only; URL text and source metadata never enter
    the exception.
    """

    canonical_sources: list[dict[str, Any]] = []
    index_by_url: dict[str, int] = {}

    def _optional_text(
        source: Mapping[str, Any],
        field: str,
        *,
        source_index: int,
    ) -> str | None:
        value = source.get(field)
        if value is None:
            return None
        if not isinstance(value, str):
            raise SourceCollectionValidationError(
                f"sources_collected[{source_index}].{field} must be text or null"
            )
        normalized = value.strip()
        return normalized or None

    def _optional_alive(
        source: Mapping[str, Any],
        *,
        source_index: int,
    ) -> bool | None:
        value = source.get("alive")
        if value is None:
            return None
        if not isinstance(value, bool):
            raise SourceCollectionValidationError(
                f"sources_collected[{source_index}].alive must be boolean or null"
            )
        return value

    for source_index, raw_source in enumerate(sources):
        if not isinstance(raw_source, Mapping):
            raise SourceCollectionValidationError(
                f"sources_collected[{source_index}] must be an object"
            )
        try:
            canonical_url = _canonical_ledger_url(raw_source.get("url"))
        except (TypeError, ValueError) as exc:
            raise SourceCollectionValidationError(
                f"sources_collected[{source_index}].url is invalid"
            ) from exc

        incoming = dict(raw_source)
        incoming["url"] = canonical_url
        incoming_title = _optional_text(
            raw_source,
            "title",
            source_index=source_index,
        )
        incoming_snippet = _optional_text(
            raw_source,
            "snippet",
            source_index=source_index,
        )
        incoming_alive = _optional_alive(
            raw_source,
            source_index=source_index,
        )
        normalized_projection = {
            "title": incoming_title,
            "snippet": incoming_snippet,
            "alive": incoming_alive,
        }
        for field_name, value in normalized_projection.items():
            if value is None:
                incoming.pop(field_name, None)
            else:
                incoming[field_name] = value

        existing_index = index_by_url.get(canonical_url)
        if existing_index is None:
            index_by_url[canonical_url] = len(canonical_sources)
            canonical_sources.append(incoming)
            continue

        existing = canonical_sources[existing_index]
        for field_name, incoming_value in normalized_projection.items():
            existing_value = existing.get(field_name)
            if existing_value is None:
                if incoming_value is not None:
                    existing[field_name] = incoming_value
                continue
            if incoming_value is None or incoming_value == existing_value:
                continue
            raise SourceCollectionValidationError(
                "sources_collected canonical duplicate has conflicting "
                f"{field_name} metadata"
            )

        # Projection-visible values are reconciled above.  Preserve the richer
        # internal grounding fields with the existing quality ratchet so a
        # compatible canonical duplicate does not discard a fetched body.
        existing_quality = (
            2 if existing.get("content_quality") == "readable_body" else 1
        )
        incoming_quality = (
            2 if incoming.get("content_quality") == "readable_body" else 1
        )
        existing_chars = max(
            len(str(existing.get("body") or "")),
            len(str(existing.get("snippet") or "")),
        )
        incoming_chars = max(
            len(str(incoming.get("body") or "")),
            len(str(incoming.get("snippet") or "")),
        )
        content_upgrade = (incoming_quality, incoming_chars) >= (
            existing_quality,
            existing_chars,
        )
        for field_name, value in incoming.items():
            if field_name in {"url", "title", "snippet", "alive"}:
                continue
            if value is None or value == "":
                continue
            if field_name in {"body", "body_source", "content_quality"}:
                if not content_upgrade:
                    continue
            if (
                field_name == "authority_tier"
                and existing.get("score") is not None
            ):
                if incoming.get("score") is None:
                    continue
            existing[field_name] = value

    return canonical_sources


def _source_collection_drop_reason(
    exc: SourceCollectionValidationError,
) -> str:
    """Map a strict per-record failure to one bounded structural reason code."""

    message = str(exc)
    markers = (
        ("must be an object", "not_object"),
        (".url is invalid", "invalid_url"),
        (".title must be text or null", "invalid_title"),
        (".snippet must be text or null", "invalid_snippet"),
        (".alive must be boolean or null", "invalid_alive"),
    )
    for marker, reason in markers:
        if marker in message:
            return reason
    return "invalid_record"


def _source_projection_richness_key(source: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return an order-independent preference key for one canonical record."""

    def _text(field_name: str) -> str:
        value = source.get(field_name)
        return value if isinstance(value, str) else ""

    title = _text("title")
    snippet = _text("snippet")
    body = _text("body")
    projection_fields = sum(
        source.get(field_name) is not None
        for field_name in ("title", "snippet", "alive")
    )
    populated_fields = sum(
        value is not None and value != "" for value in source.values()
    )
    alive = source.get("alive")
    alive_token = "true" if alive is True else "false" if alive is False else ""
    return (
        int(source.get("content_quality") == "readable_body"),
        len(body),
        projection_fields,
        len(snippet) + len(title),
        populated_fields,
        int(source.get("score") is not None),
        title,
        snippet,
        alive_token,
        body,
        _text("body_source"),
        _text("content_quality"),
        _text("authority_tier"),
        str(source.get("score") or ""),
    )


def _project_sources_collected(
    sources: Sequence[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project valid sources without letting one unsafe row erase its siblings.

    Every input record first passes through the strict Ledger canonicalizer in
    isolation. Invalid records are dropped individually. Canonical duplicates
    are then processed richest-first: compatible records merge through that
    same strict helper, while a conflicting record loses to the deterministic
    richer record. Telemetry contains only fixed reason codes and integer
    counts—never source URLs, metadata, record indexes, or exception text.
    """

    reason_counts: Counter[str] = Counter()
    groups: dict[str, list[dict[str, Any]]] = {}
    for raw_source in sources:
        try:
            canonical_source = _canonicalize_sources_collected([raw_source])[0]
        except SourceCollectionValidationError as exc:
            reason_counts[_source_collection_drop_reason(exc)] += 1
            continue
        canonical_url = str(canonical_source["url"])
        groups.setdefault(canonical_url, []).append(canonical_source)

    canonical_sources: list[dict[str, Any]] = []
    merged_records = 0
    for records in groups.values():
        ranked = sorted(records, key=_source_projection_richness_key, reverse=True)
        selected = ranked[0]
        for candidate in ranked[1:]:
            try:
                # Put the richer selection second so the strict helper's
                # equal-quality ratchet cannot overwrite it with the candidate.
                selected = _canonicalize_sources_collected(
                    [candidate, selected]
                )[0]
            except SourceCollectionValidationError:
                reason_counts["canonical_metadata_conflict"] += 1
            else:
                merged_records += 1
        canonical_sources.append(selected)

    bounded_reason_counts = {
        reason: int(reason_counts.get(reason, 0))
        for reason in _SOURCE_COLLECTION_DROP_REASON_CODES
        if reason_counts.get(reason, 0)
    }
    dropped_records = sum(bounded_reason_counts.values())
    telemetry = {
        "input_records": len(sources),
        "projected_records": len(canonical_sources),
        "dropped_records": dropped_records,
        "merged_records": merged_records,
        "reason_counts": bounded_reason_counts,
    }
    return canonical_sources, telemetry


def build_default_model(
    *,
    settings: Any | None = None,
    api_key: str | None = None,
    request_timeout_s: float | None = None,
) -> Any:
    """Build the resolved model shared by synthesis, critique and recovery.

    Construction makes no inference request. Stage callers cannot override
    provider parameters: smolagents model defaults win over per-call kwargs.
    """
    from dataclasses import replace

    from smolagents import OpenAIServerModel
    from smolagents.models import REMOVE_PARAMETER  # type: ignore[import-untyped]

    from deepsearch.config import DeepSearchSettings
    from deepsearch.llm.runtime import ModelConfigurationError, resolve_effective_model

    configured = settings if settings is not None else DeepSearchSettings.load()
    if api_key is not None:
        key_field = "openai_api_key" if configured.llm_backend == "openai" else "deepseek_api_key"
        configured = replace(configured, **{key_field: api_key})
    if request_timeout_s is not None:
        configured = replace(configured, llm_request_timeout_seconds=request_timeout_s)
    try:
        effective = resolve_effective_model(configured)
    except ModelConfigurationError as exc:
        # All ModelConfigurationError messages are fixed, content-free strings.
        raise CodeAgentRunnerError(str(exc)) from exc

    def _construct(cap: int) -> Any:
        selected = replace(effective, max_output_tokens=cap)
        parameters = selected.request_parameters()
        parameters["temperature"] = REMOVE_PARAMETER
        if selected.provider == "openai":
            parameters["max_tokens"] = REMOVE_PARAMETER
        model = OpenAIServerModel(
            model_id=selected.model,
            api_base=selected.base_url,
            api_key=selected.api_key,
            **parameters,
            retry=SMOLAGENTS_RATE_LIMIT_RETRY,
            client_kwargs={
                "timeout": selected.request_timeout,
                "max_retries": OPENAI_CLIENT_MAX_RETRIES,
            },
        )
        model.effective_config = selected.public_metadata()
        from deepsearch.llm.usage import observe_model_requests

        observe_model_requests(model, model.effective_config)
        return model

    try:
        return _construct(effective.max_output_tokens)
    except ValueError:
        # Construction is local only; never retry a provider request here.
        fallback_cap = min(effective.max_output_tokens, _FALLBACK_MAX_TOKENS)
        if fallback_cap < effective.max_output_tokens:
            logger.warning("Model construction rejected configured cap; using bounded fallback")
            return _construct(fallback_cap)
        raise


# ---------------------------------------------------------------------------
# Step trace serializer (verbatim from S62 spike).
# ---------------------------------------------------------------------------


def _step_to_dict(step: Any, idx: int) -> dict[str, Any]:
    out: dict[str, Any] = {"step_index": idx, "type": type(step).__name__}
    for attr in (
        "step_number",
        "task",
        "is_final_answer",
        "code_action",
        "observations",
        "model_output",
        "action_output",
        "error",
    ):
        if hasattr(step, attr):
            v = getattr(step, attr)
            if v is None:
                continue
            if hasattr(v, "model_dump"):
                try:
                    v = v.model_dump(mode="json")
                except Exception:
                    v = str(v)
            elif not isinstance(v, (str, int, float, bool, list, dict)):
                v = str(v)
            out[attr] = v
    tu = getattr(step, "token_usage", None)
    if tu is not None:
        out["token_usage"] = {
            "input_tokens": getattr(tu, "input_tokens", None),
            "output_tokens": getattr(tu, "output_tokens", None),
            "total_tokens": getattr(tu, "total_tokens", None),
        }
    timing = getattr(step, "timing", None)
    if timing is not None:
        out["timing"] = {
            "start_time": getattr(timing, "start_time", None),
            "end_time": getattr(timing, "end_time", None),
            "duration": getattr(timing, "duration", None),
        }
    tool_calls = getattr(step, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [
            {"name": getattr(tc, "name", None), "arguments": getattr(tc, "arguments", None)}
            for tc in tool_calls
        ]
    return out


def _step_finish_reason(step: Any) -> str | None:
    """Return the model ``finish_reason`` for one ActionStep, or ``None``.

    S88 — smolagents 1.24.0 (non-streaming, the runner's path) stores the raw
    OpenAI ``ChatCompletion`` on ``ChatMessage.raw`` and keeps that message on
    ``ActionStep.model_output_message``, so ``raw.choices[0].finish_reason`` is
    the provider's authoritative signal that a generation was cut at
    ``max_tokens`` (``"length"``). The getattr chain fails safe to ``None`` for
    the streaming path (raw is None), the max-steps fallback step (no
    model_output_message — that is the separate ``hit_max_steps`` case), and
    fake models in unit tests.
    """

    message = getattr(step, "model_output_message", None)
    return _message_finish_reason(message)


def _message_finish_reason(message: Any) -> str | None:
    """Return the provider finish reason retained on a chat message."""

    raw = getattr(message, "raw", None)
    choices = getattr(raw, "choices", None)
    if not choices:
        return None
    return getattr(choices[0], "finish_reason", None)


def _message_text(message: Any) -> str:
    """Return explicit text from one smolagents chat message, or ``""``.

    Never stringify an unknown response object: its repr may carry provider or
    request data.
    """

    if isinstance(message, str):
        return message
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _normalise_token_usage_envelope(
    usage: Any,
) -> tuple[dict[str, int], bool]:
    """Return observed provider usage and whether its total is exact."""

    zero = {"input": 0, "output": 0, "total": 0}
    if usage is None:
        return zero, False

    def _read(names: tuple[str, ...]) -> tuple[int, bool]:
        for name in names:
            try:
                if isinstance(usage, Mapping):
                    if name not in usage:
                        continue
                    raw = usage.get(name)
                else:
                    raw = getattr(usage, name, None)
            except Exception:  # noqa: BLE001 — malformed telemetry is incomplete
                return 0, False
            if raw is None:
                continue
            if isinstance(raw, bool):
                return 0, False
            try:
                value = int(raw)
            except (TypeError, ValueError, OverflowError):
                return 0, False
            return (value, True) if value >= 0 else (0, False)
        return 0, False

    input_tokens, has_input = _read(("input_tokens", "input"))
    output_tokens, has_output = _read(("output_tokens", "output"))
    total_tokens, has_total = _read(("total_tokens", "total"))
    component_total = input_tokens + output_tokens
    envelope_consistent = True
    if not has_total:
        total_tokens = component_total
    elif total_tokens < component_total:
        # Contradictory provider fields cannot prove an exact total. Preserve
        # the larger observed lower bound rather than the impossible smaller one.
        total_tokens = component_total
        envelope_consistent = False
    return (
        {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
        },
        bool(envelope_consistent and (has_total or (has_input and has_output))),
    )


# ---------------------------------------------------------------------------
# Runner — synchronous, called from an async node via ``asyncio.to_thread``.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SynthesisResult:
    """Structured output of one CodeAgent synthesis run."""

    output_markdown: str
    step_count: int
    self_terminated: bool
    hit_max_steps: bool
    runtime_seconds: float
    token_usage: dict[str, int]
    tool_call_summary: dict[str, Any]
    authority_tier_distribution: dict[str, int]
    pool_size: int
    step_trace: list[dict[str, Any]]
    # False means the recorded provider usage is an observed lower bound.
    token_usage_complete: bool = False
    error: str | None = None
    # S90 (#959) — True when ``error`` is set AND the failure that ended the run
    # satisfied every condition of _is_retryable_generation_failure (transient
    # transport/server condition, zero tokens consumed). The worker uses this to
    # return the mission to 'queued' instead of landing it terminally in
    # 'blocked', making the lease's remaining attempts reachable. Always False
    # on success.
    error_transient: bool = False
    sources_collected: list[dict[str, Any]] = field(default_factory=list)
    sources_discovered: list[dict[str, Any]] = field(default_factory=list)
    sources_fetched: list[dict[str, Any]] = field(default_factory=list)
    fetch_attempts: list[dict[str, Any]] = field(default_factory=list)
    probes: list[dict[str, Any]] = field(default_factory=list)
    scope_telemetry: dict[str, Any] = field(default_factory=dict)
    retrieval_state: dict[str, int] = field(default_factory=dict)
    recovery_passes: int = 0
    # S82 m03 (PD135) — True when output_markdown was salvaged from the
    # scratchpad because the agent hit max_steps without a usable final_answer.
    recovered_from_scratchpad: bool = False
    # S88 — True when the final-answer generation was cut at max_tokens
    # (finish_reason == "length"): the shipped report is truncated. Feeds the
    # outcome gate so a budget-truncated report is flagged/recovered, never
    # silently shipped complete. Defaults False (legacy fixtures / fake models).
    output_truncated: bool = False
    # S84 m04 — per-call tool OUTCOME telemetry (by-tool rollup + capped
    # per-call records). Empty dict on legacy fixtures.
    tool_outcomes: dict[str, Any] = field(default_factory=dict)
    # S84 m04 — the model id actually bound for this run (empty on legacy
    # fixtures; finalize telemetry falls back to the locked default).
    model_id: str = ""
    effective_config: dict[str, Any] = field(default_factory=dict)
    completion_control: dict[str, Any] = field(default_factory=dict)
    provider_attempts: list[dict[str, Any]] = field(default_factory=list)
    # Compatibility summary for consumers predating the tolerant projection.
    # Set only when unsafe/conflicting records were dropped; details live in
    # the bounded structural telemetry below.
    source_collection_error: str | None = None
    # Runtime-tolerant projection telemetry. Fixed reason codes and integer
    # counts only; defaults empty for legacy fixtures and external callers.
    source_collection_projection: dict[str, Any] = field(default_factory=dict)
    # True only when a terminal fallback returned one exact literal
    # ``final_answer(...)`` wrapper that the deterministic normalizer removed.
    output_wrapper_removed: bool = False
    # True only when the emergency max-step generation carries the provider's
    # authoritative ``finish_reason == "stop"``. Smolagents does not retain
    # that fallback chat message on ``model_output_message``, so the runner
    # captures it at the model-call boundary.
    max_steps_fallback_complete: bool = False
    # A mid-run AgentGenerationError may follow substantial paid work. These
    # fields retain content-free failure evidence even when same-memory final
    # answer recovery clears the fatal ``error``.
    generation_failure_error: str | None = None
    generation_failure_reason: str = ""
    generation_failure_reason_source: str = "unknown"
    generation_failure_root_class: str = "Exception"
    generation_failure_http_status: int | None = None
    generation_failure_transient: bool = False
    generation_failure_step: int = 0
    generation_failure_pass: int = 0
    # Count of pipeline passes that ended in a generation failure. In-process
    # zero-progress retry attempts are not separate synthesis passes.
    generation_failure_pass_count: int = 0
    generation_failure_consumed_tokens: int = 0
    generation_failure_retry_eligible: bool = False
    generation_failure_retry_suppressed_reason: str = "none"
    generation_recovery_attempted: bool = False
    generation_recovery_result: str = "not_attempted"
    generation_recovery_finish_reason: str = ""
    generation_error_recovered: bool = False


def _configure_max_steps_final_answer_prompt(
    agent: Any,
    system_instructions: str,
) -> None:
    """Keep the report contract visible to smolagents' max-step fallback.

    smolagents 1.24's ``provide_final_answer`` constructs a new system message
    from ``prompt_templates["final_answer"]`` and deliberately drops the
    CodeAgent's original system-prompt memory entry. Install a per-agent
    template that carries the already-rendered mission contract into that
    fallback call.
    """

    prompt_templates = getattr(agent, "prompt_templates", None)
    if not isinstance(prompt_templates, dict):
        # Lightweight test doubles do not initialise smolagents' template map.
        prompt_templates = {}
        agent.prompt_templates = prompt_templates
    prompt_templates["final_answer"] = {
        "pre_messages": (
            f"{system_instructions.rstrip()}\n\n"
            f"{_MAX_STEPS_FINAL_ANSWER_INSTRUCTIONS}"
        ),
        "post_messages": _MAX_STEPS_FINAL_ANSWER_POST_MESSAGE,
    }


def _unwrap_max_steps_final_answer(text: str) -> tuple[str, bool]:
    """Unwrap one exact, literal fallback ``final_answer`` call safely.

    The max-step model sometimes follows CodeAgent's normal tool-call syntax
    even though smolagents returns that fallback response verbatim. Accept only
    a whole-output call whose sole argument is a Python string literal,
    optionally enclosed by the two wrapper forms emitted by smolagents. No
    expression is evaluated, and every ambiguous output is preserved exactly.
    """

    original = text
    candidate = text.strip()
    if candidate.startswith("<code>") and candidate.endswith("</code>"):
        candidate = candidate[len("<code>") : -len("</code>")].strip()
    else:
        lines = candidate.splitlines()
        if (
            len(lines) >= 3
            and lines[0].strip() in {"```", "```py", "```python"}
            and lines[-1].strip() == "```"
        ):
            candidate = "\n".join(lines[1:-1]).strip()

    # This lexical boundary rejects prose, comments, trailing statements, and
    # non-call code before the AST performs the structural/literal checks.
    if not candidate.startswith("final_answer(") or not candidate.endswith(")"):
        return original, False
    try:
        parsed = ast.parse(candidate, mode="exec")
    except (SyntaxError, ValueError):
        return original, False
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Expr):
        return original, False
    call = parsed.body[0].value
    if (
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Name)
        or call.func.id != "final_answer"
        or len(call.args) != 1
        or call.keywords
    ):
        return original, False
    argument = call.args[0]
    if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
        return original, False
    return argument.value, True


def _has_cite_markers(text: str) -> bool:
    """Cheap check for at least one ``[cite:URL]`` marker."""
    return isinstance(text, str) and "[cite:" in text


def _has_unresolved_final_answer_wrapper(text: str) -> bool:
    """Return True when recovery output still looks like executable wrapping.

    A recovery response is accepted as raw Markdown or as the one exact
    literal ``final_answer("...")`` shape handled by
    :func:`_unwrap_max_steps_final_answer`.  Any remaining final_answer call is
    ambiguous code: citations inside a Python string do not make it a report.
    """

    candidate = text.strip()
    if candidate.startswith("<code>") or candidate.endswith("</code>"):
        return True
    lines = candidate.splitlines()
    if lines and (
        lines[0].strip().startswith("```")
        or lines[-1].strip() == "```"
    ):
        return True
    try:
        parsed = ast.parse(candidate, mode="exec")
    except (SyntaxError, ValueError):
        return candidate.startswith("final_answer")
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "final_answer"
        for node in ast.walk(parsed)
    ):
        return True
    # With citations present, raw Markdown normally is not valid Python. A
    # nonempty parsed module (assignment, import, expression, etc.) is code and
    # cannot be accepted as a report merely because a string literal contains
    # a cite marker. Markdown headings parse as comments and leave body empty.
    return bool(parsed.body)


def _recover_stranded_draft(scratchpad: Mapping[str, str]) -> str | None:
    """Return the best ``[cite:URL]``-bearing scratchpad draft, or None.

    Prefers the conventional :data:`SCRATCHPAD_RECOVERY_KEY` (``report_draft``)
    when it carries markers; otherwise falls back to the LONGEST cited value
    across the scratchpad (the agent may have stashed its assembled draft under
    a different key). Returns None when no scratchpad value carries a marker —
    nothing recoverable, leaving the empty/marker-less output for the m01 gate.
    """
    if not scratchpad:
        return None
    preferred = scratchpad.get(SCRATCHPAD_RECOVERY_KEY)
    if isinstance(preferred, str) and _has_cite_markers(preferred):
        return preferred
    best: str | None = None
    best_len = 0
    for value in scratchpad.values():
        if isinstance(value, str) and _has_cite_markers(value) and len(value) > best_len:
            best = value
            best_len = len(value)
    return best


def _consumed_tokens(agent: Any) -> int:
    """Total tokens billed across an agent's recorded steps."""
    steps = getattr(getattr(agent, "memory", None), "steps", None) or []
    total = 0
    for step in steps:
        usage = getattr(step, "token_usage", None)
        if usage is None:
            continue
        total += int(getattr(usage, "input_tokens", 0) or 0)
        total += int(getattr(usage, "output_tokens", 0) or 0)
    return total


def _has_meaningful_action_memory(agent: Any) -> bool:
    """Return True when an action step retains evidence of completed work.

    Token usage can be missing even after a paid model response.  A fresh-agent
    retry is safe only when memory contains neither billed tokens nor model/tool
    output from a completed action.
    """

    steps = getattr(getattr(agent, "memory", None), "steps", None) or []
    for step in steps:
        if type(step).__name__ != "ActionStep":
            continue
        for attribute in (
            "model_output",
            "model_output_message",
            "tool_calls",
            "observations",
            "action_output",
        ):
            value = getattr(step, attribute, None)
            if value is None:
                continue
            if isinstance(value, (str, bytes, tuple, list, dict, set)):
                if len(value) == 0:
                    continue
                return True
            # Unknown framework values are conservatively treated as retained
            # work; a false positive costs one same-memory fallback, while a
            # false negative can restart and double-charge a paid mission.
            if value is not None:
                return True
    return False


def _has_paid_progress(agent: Any) -> bool:
    """Return True when retrying from step one could discard paid progress."""

    return _consumed_tokens(agent) > 0 or _has_meaningful_action_memory(agent)


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Return a bounded, cycle-safe cause/context chain."""

    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and len(chain) < 8 and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        current = cause if isinstance(cause, BaseException) else context
        if not isinstance(current, BaseException):
            current = None
    return chain


@dataclass(frozen=True, slots=True)
class _GenerationFailureClassification:
    """Secret-safe, bounded explanation of one generation failure."""

    reason: str = "other"
    source: str = "unknown"
    root_class: str = "Exception"
    http_status: int | None = None


def _generation_failure_classification_unchecked(
    exc: BaseException,
) -> _GenerationFailureClassification:
    """Collapse a generation exception to bounded, content-free facts.

    Typed classes and HTTP status fields take precedence. Message inspection
    is an in-memory fallback for smolagents wrappers and is never persisted.
    """

    chain = _exception_chain(exc)
    root_class = safe_exception_details(chain[-1]).exception_class
    statuses: list[int] = []
    for item in chain:
        raw_status = getattr(item, "status_code", None)
        if raw_status is None:
            raw_status = getattr(getattr(item, "response", None), "status_code", None)
        if isinstance(raw_status, int) and 100 <= raw_status <= 599:
            statuses.append(raw_status)

    http_status = statuses[0] if statuses else None

    def classified(
        reason: str,
        source: str,
        *,
        status: int | None = http_status,
    ) -> _GenerationFailureClassification:
        return _GenerationFailureClassification(
            reason=reason,
            source=source,
            root_class=root_class,
            http_status=status,
        )

    if 429 in statuses:
        return classified("rate_limit", "status", status=429)
    server_status = next((status for status in statuses if status >= 500), None)
    if server_status is not None:
        return classified("server", "status", status=server_status)
    if 408 in statuses:
        return classified("timeout", "status", status=408)
    if 402 in statuses:
        return classified("balance", "status", status=402)
    auth_status = next(
        (status for status in statuses if status in {401, 403}), None
    )
    if auth_status is not None:
        return classified("authentication", "status", status=auth_status)

    message_text = " ".join(str(item).lower() for item in chain)
    generic_client_status = next(
        (status for status in statuses if 400 <= status < 500),
        None,
    )
    if generic_client_status is not None:
        # OpenAI-compatible APIs commonly encode context overflow as 400/413.
        # Preserve that specific deterministic cause, but never let a generic
        # 4xx body mentioning a transient word (for example "timeout") turn a
        # client rejection into a retryable outage.
        if generic_client_status in {400, 413} and any(
            marker in message_text
            for marker in (
                "context_length",
                "context length",
                "context window",
                "too many tokens",
            )
        ):
            return classified(
                "context_length",
                "message_marker",
                status=generic_client_status,
            )
        return classified(
            "invalid_request",
            "status",
            status=generic_client_status,
        )

    class_text = " ".join(type(item).__name__.lower() for item in chain)
    if "timeout" in class_text:
        return classified("timeout", "class")
    if "connection" in class_text or "connecterror" in class_text:
        return classified("connection", "class")
    if "ratelimit" in class_text:
        return classified("rate_limit", "class")
    if "authentication" in class_text or "permission" in class_text:
        return classified("authentication", "class")

    markers_by_reason = (
        (
            "context_length",
            ("context_length", "context length", "context window", "too many tokens"),
        ),
        ("balance", ("insufficient balance", "insufficient funds", "payment required")),
        (
            "authentication",
            ("authentication", "unauthorized", "invalid api key", "permission denied"),
        ),
        ("rate_limit", ("rate limit", "too many requests")),
        ("overloaded", ("overloaded",)),
        ("timeout", ("request timed out", "timed out", "timeout")),
        (
            "connection",
            (
                "connection error",
                "connection reset",
                "connection aborted",
                "network error",
                "dns",
            ),
        ),
        (
            "server",
            (
                "internal server error",
                "service unavailable",
                "temporarily unavailable",
                "bad gateway",
            ),
        ),
        ("invalid_request", ("invalid request", "bad request", "malformed request")),
    )
    for reason, markers in markers_by_reason:
        if any(marker in message_text for marker in markers):
            return classified(reason, "message_marker")
    if "badrequest" in class_text or "invalidrequest" in class_text:
        return classified("invalid_request", "class")
    return classified("other", "unknown")


def _generation_failure_classification(
    exc: BaseException,
) -> _GenerationFailureClassification:
    """Return failure facts without letting diagnostics break synthesis."""

    try:
        classification = _generation_failure_classification_unchecked(exc)
    except Exception:  # noqa: BLE001 — diagnostic classification must fail safe
        return _GenerationFailureClassification()
    if classification.reason not in _GENERATION_FAILURE_REASONS:
        return _GenerationFailureClassification(
            root_class=classification.root_class,
            http_status=classification.http_status,
        )
    return classification


def _generation_failure_reason(exc: BaseException) -> str:
    """Return a failure code without letting diagnostics break synthesis."""

    return _generation_failure_classification(exc).reason


def _is_retryable_generation_failure(exc: BaseException, agent: Any) -> bool:
    """Return True when a failed synthesis attempt is safe to run again.

    All three conditions must hold (S89):

    1. The exception is a smolagents ``AgentGenerationError`` — the model call
       itself failed, not the code the model generated. Matched by class name
       rather than ``isinstance`` because smolagents is imported lazily inside
       this module (it is absent from some local environments, and a
       module-scope import would break collection there).
    2. Typed cause/status classification identifies a transient transport or
       server condition. A deterministic failure — bad key, malformed request,
       context overflow — fails identically, so retrying only burns the lease.
    3. The attempt retains neither billed tokens nor meaningful action memory.
       Usage telemetry can be absent after a paid response; model/tool output
       is therefore independent evidence that restarting would discard work.

    Note: ``retry_with_backoff`` in ``deepsearch.agent.resilience`` is the
    house retry helper, but it decides purely on exception type and cannot see
    condition 3 — the guard that keeps this from re-running paid work. Hence
    the explicit loop in the caller.
    """
    if type(exc).__name__ != "AgentGenerationError":
        return False
    if _generation_failure_reason(exc) not in _TRANSIENT_GENERATION_REASONS:
        return False
    return not _has_paid_progress(agent)


def run_codeagent_synthesis(
    *,
    system_instructions: str,
    user_directive: str,
    model: Any | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    verbosity: int = SMOLAGENTS_VERBOSITY_OFF,
    deliverable_spec: Sequence[str] | None = None,
    excluded_domains: Sequence[str] | None = None,
    high_authority_domains: Sequence[str] | None = None,
    scope_admission: ScopeAdmission | None = None,
    report_word_budget_check: Callable[[str], dict[str, Any]] | None = None,
    cancel_event: CancellationEvent | None = None,
) -> SynthesisResult:
    """Run a single CodeAgent synthesis pass with the live retrieval tools.

    Synchronous (smolagents itself is sync). Wrap with ``asyncio.to_thread``
    when calling from an async context.

    Args:
        system_instructions: The agent's system prompt.
        user_directive: The agent's user-facing task message.
        model: Optional ``OpenAIServerModel`` instance (defaults to the locked
            effective provider bind via ``build_default_model``).
        max_steps: Maximum agent loop iterations (default ``DEFAULT_MAX_STEPS``,
            currently 40 per S79 m01.d).
        verbosity: smolagents log verbosity. Production callers pin this to
            ``SMOLAGENTS_VERBOSITY_OFF`` (-1).
        deliverable_spec: Optional list of deliverable bullet entries from the
            mission contract (S79 m02.a). Threaded into ``build_codeagent_tools``
            so the agent's ``completeness_check`` tool sees the same list it
            saw in its system prompt.
        excluded_domains: Mission-local retrieval exclusions.
        high_authority_domains: Mission-local primary-domain authority seeds.
        report_word_budget_check: Provider-free preview of the full rendered
            report, used for advisory length feedback within the current run.

    Raises ``CodeAgentRunnerError`` on hard failures: missing
    ``LINKUP_API_KEY`` (the agent has no retrieval surface without it) or
    missing selected-provider credential (raised inside ``build_default_model``). Soft
    failures (the agent itself errors mid-run) are captured on the returned
    ``SynthesisResult.error`` field as a bounded error category and exception
    class; the caller decides how to surface whatever partial state survived.
    """

    resolved_cancel_event = (
        cancel_event
        if cancel_event is not None
        else current_execution_cancel_event()
    )
    raise_if_execution_cancelled(resolved_cancel_event)

    if not isinstance(system_instructions, str) or not system_instructions.strip():
        raise CodeAgentRunnerError("system_instructions must be a non-empty string")
    if not isinstance(user_directive, str) or not user_directive.strip():
        raise CodeAgentRunnerError("user_directive must be a non-empty string")
    if not os.environ.get("LINKUP_API_KEY"):
        raise CodeAgentRunnerError(
            "LINKUP_API_KEY not set — the CodeAgent has no retrieval surface"
        )

    controller = CompletionController(
        max_steps, deliverable_spec, report_word_budget_check=report_word_budget_check,
    )
    bundle = build_codeagent_tools(
        deliverable_spec=deliverable_spec,
        excluded_domains=excluded_domains,
        high_authority_domains=high_authority_domains,
        cancel_event=resolved_cancel_event,
        completion_controller=controller,
        scope_admission=scope_admission,
    )
    raise_if_execution_cancelled(resolved_cancel_event)
    bound_model = model if model is not None else build_default_model()
    observed_attempts = getattr(bound_model, "provider_attempts", [])
    observed_attempts = observed_attempts if isinstance(observed_attempts, list) else []
    observation_start = len(observed_attempts)

    from smolagents import CodeAgent

    logger.info(
        "CodeAgent synthesis start: max_steps=%d verbosity=%d",
        max_steps,
        verbosity,
    )

    t0 = time.monotonic()
    error: str | None = None
    error_transient = False
    output_text: str = ""
    cum_input = cum_output = cum_total = 0
    attempts = max(1, SYNTHESIS_RETRY_ATTEMPTS)
    agent: Any = None
    fallback_finish_reason: str | None = None
    recovery_usage_observed = {"input": 0, "output": 0, "total": 0}
    recovery_usage_available = False
    prior_attempt_usage_complete = True
    run_result_usage_complete = False
    recovery_wrapper_removed = False
    generation_failure_error: str | None = None
    generation_failure_reason = ""
    generation_failure_reason_source = "unknown"
    generation_failure_root_class = "Exception"
    generation_failure_http_status: int | None = None
    generation_failure_transient = False
    generation_failure_step = 0
    generation_failure_pass = 0
    generation_failure_pass_count = 0
    generation_failure_consumed_tokens = 0
    generation_failure_retry_eligible = False
    generation_failure_retry_suppressed_reason = "none"
    generation_recovery_attempted = False
    generation_recovery_result = "not_attempted"
    generation_recovery_finish_reason = ""
    generation_error_recovered = False
    # S89 — a fresh CodeAgent per attempt so a retry starts from a clean step 1
    # with no memory carried over from the failed attempt. ``bundle.tools`` is
    # copied because smolagents may append its own final_answer tool to the
    # list it is handed. The scratchpad in ``bundle`` is intentionally shared:
    # a retry only happens when zero tokens were consumed, so nothing can have
    # been written to it yet.
    def cancel_after_step(_step: Any) -> None:
        raise_if_execution_cancelled(resolved_cancel_event)

    def completion_after_step(step: Any) -> None:
        controller.after_step(step, bundle.scratchpad)
        if isinstance(getattr(bound_model, "provider_attempts", None), list):
            bound_model.usage_step = controller.completed_steps + 1

    for attempt in range(1, attempts + 1):
        if isinstance(getattr(bound_model, "provider_attempts", None), list):
            bound_model.usage_attempt = attempt
            bound_model.usage_stage = "synthesis"
            bound_model.usage_step = 1
        raise_if_execution_cancelled(resolved_cancel_event)
        agent = CodeAgent(
            tools=list(bundle.tools),
            model=bound_model,
            max_steps=max_steps,
            instructions=system_instructions,
            verbosity_level=verbosity,
            step_callbacks=[cancel_after_step, completion_after_step],
        )
        _configure_max_steps_final_answer_prompt(agent, system_instructions)
        fallback_finish_reason = None
        original_provide_final_answer = getattr(agent, "provide_final_answer", None)
        if callable(original_provide_final_answer):

            def _capture_fallback_final_answer(task: str) -> Any:
                nonlocal fallback_finish_reason
                bound_model.usage_stage = "retained_memory_finalization"
                try:
                    message = original_provide_final_answer(task)
                finally:
                    bound_model.usage_stage = "synthesis"
                fallback_finish_reason = _message_finish_reason(message)
                return message

            agent.provide_final_answer = _capture_fallback_final_answer
        error = None
        error_transient = False
        recovery_usage_observed = {"input": 0, "output": 0, "total": 0}
        recovery_usage_available = False
        run_result_usage_complete = False
        recovery_wrapper_removed = False
        generation_failure_error = None
        generation_failure_reason = ""
        generation_failure_reason_source = "unknown"
        generation_failure_root_class = "Exception"
        generation_failure_http_status = None
        generation_failure_transient = False
        generation_failure_step = 0
        generation_failure_pass = 0
        generation_failure_pass_count = 0
        generation_failure_consumed_tokens = 0
        generation_failure_retry_eligible = False
        generation_failure_retry_suppressed_reason = "none"
        generation_recovery_attempted = False
        generation_recovery_result = "not_attempted"
        generation_recovery_finish_reason = ""
        generation_error_recovered = False
        try:
            result = agent.run(
                user_directive + "\n\n" + controller.notice(), return_full_result=True
            )
            raise_if_execution_cancelled(resolved_cancel_event)
            output_text = (
                result.output if isinstance(result.output, str) else str(result.output)
            )
            run_usage, run_result_usage_complete = (
                _normalise_token_usage_envelope(
                    getattr(result, "token_usage", None)
                )
            )
            cum_input = run_usage["input"]
            cum_output = run_usage["output"]
            cum_total = run_usage["total"]
            break
        except ExecutionCancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — agent errors are the load-bearing surface
            # A dispatched provider attempt that exits without a RunResult
            # cannot prove its usage total, even when a later retry succeeds.
            prior_attempt_usage_complete = False
            error = safe_exception_text(exc, category="execution")
            logger.warning("CodeAgent synthesis failed mid-run: %s", error)
            is_generation_failure = type(exc).__name__ == "AgentGenerationError"
            has_paid_progress = False
            if is_generation_failure:
                classification = _generation_failure_classification(exc)
                generation_failure_error = error
                generation_failure_reason = classification.reason
                generation_failure_reason_source = classification.source
                generation_failure_root_class = classification.root_class
                generation_failure_http_status = classification.http_status
                generation_failure_transient = (
                    generation_failure_reason in _TRANSIENT_GENERATION_REASONS
                )
                generation_failure_consumed_tokens = _consumed_tokens(agent)
                observed_progress = any(
                    event.get("status") == "returned"
                    or int((event.get("usage") or {}).get("total") or 0) > 0
                    for event in observed_attempts[observation_start:]
                )
                has_paid_progress = _has_paid_progress(agent) or observed_progress
                memory_steps = (
                    getattr(getattr(agent, "memory", None), "steps", None) or []
                )
                generation_failure_step = sum(
                    1 for step in memory_steps if type(step).__name__ == "ActionStep"
                )
                generation_failure_pass_count = 1
                generation_failure_retry_eligible = (
                    generation_failure_transient and not has_paid_progress
                )
                if has_paid_progress:
                    generation_failure_retry_suppressed_reason = "paid_progress"
                elif not generation_failure_transient:
                    generation_failure_retry_suppressed_reason = "deterministic"
            # S90 (#959) — the same verdict that decides an in-process retry
            # also decides whether the WORKER may requeue the mission across
            # attempts. Recorded on every failed attempt so the value that
            # survives the loop describes the attempt that actually ended it.
            error_transient = _is_retryable_generation_failure(exc, agent) and not has_paid_progress
            if attempt < attempts and error_transient:
                delay = SYNTHESIS_RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                logger.warning(
                    "CodeAgent synthesis: transient generation failure on attempt "
                    "%d/%d with zero tokens consumed; retrying in %.0fs",
                    attempt,
                    attempts,
                    delay,
                )
                wait_for_execution_cancel(delay, resolved_cancel_event)
                continue
            if is_generation_failure and error_transient:
                generation_failure_retry_suppressed_reason = "attempts_exhausted"

            # smolagents re-raises AgentGenerationError before its max-step
            # fallback, although all successful paid action memory survives.
            # Make exactly one final-answer call on that SAME agent. Only a
            # cited provider-stop response clears fatality; every weaker
            # artifact retains the original safe error for quality grading.
            final_answer = getattr(agent, "provide_final_answer", None)
            if (
                is_generation_failure
                and has_paid_progress
                and callable(final_answer)
            ):
                generation_recovery_attempted = True
                raise_if_execution_cancelled(resolved_cancel_event)
                fallback_finish_reason = None
                recovery_observation_start = len(observed_attempts)
                try:
                    recovery_message = final_answer(user_directive)
                except ExecutionCancelledError:
                    raise
                except Exception as recovery_exc:  # noqa: BLE001
                    logger.warning(
                        "CodeAgent retained-memory recovery failed: %s",
                        safe_exception_text(recovery_exc, category="execution"),
                    )
                    generation_recovery_result = "failed"
                    break
                raise_if_execution_cancelled(resolved_cancel_event)
                if any(
                    event.get("status") == "failed"
                    for event in observed_attempts[recovery_observation_start:]
                ):
                    # smolagents catches SDK errors and returns their text as a
                    # ChatMessage. The wire observation proves a failed request;
                    # never treat that message as a draft or an unknown finish.
                    generation_recovery_result = "provider_failed"
                    break
                generation_recovery_finish_reason = fallback_finish_reason or ""
                recovery_text = _message_text(recovery_message)
                normalized_text, recovery_wrapper_removed = (
                    _unwrap_max_steps_final_answer(recovery_text)
                )
                unresolved_wrapper = (
                    not recovery_wrapper_removed
                    and _has_unresolved_final_answer_wrapper(recovery_text)
                )
                try:
                    recovery_usage = (
                        recovery_message.get("token_usage")
                        if isinstance(recovery_message, Mapping)
                        else getattr(recovery_message, "token_usage", None)
                    )
                except Exception:  # noqa: BLE001 — usage remains incomplete
                    recovery_usage = None
                recovery_usage_observed, recovery_usage_available = (
                    _normalise_token_usage_envelope(recovery_usage)
                )

                # No finish reason is also how provide_final_answer represents
                # a caught provider exception. Never persist that error sentence
                # as if it were a report.
                if (
                    generation_recovery_finish_reason in {"stop", "length"}
                    and normalized_text.strip()
                    and not unresolved_wrapper
                ):
                    output_text = normalized_text
                if generation_recovery_finish_reason == "length":
                    generation_recovery_result = "truncated"
                elif generation_recovery_finish_reason != "stop":
                    generation_recovery_result = "unknown_finish"
                elif not normalized_text.strip():
                    generation_recovery_result = "empty"
                elif unresolved_wrapper:
                    generation_recovery_result = "invalid_wrapper"
                elif not _has_cite_markers(normalized_text):
                    generation_recovery_result = "uncited"
                else:
                    generation_recovery_result = "success"
                    generation_error_recovered = True
                    error = None
                    error_transient = False
                    logger.warning(
                        "CodeAgent recovered a cited provider-stop final answer "
                        "from retained paid memory"
                    )
            elif is_generation_failure and has_paid_progress:
                generation_recovery_result = "unavailable"
            break
    raise_if_execution_cancelled(resolved_cancel_event)
    runtime = time.monotonic() - t0

    steps = getattr(getattr(agent, "memory", None), "steps", None) or []
    step_dicts = [_step_to_dict(s, i) for i, s in enumerate(steps)]
    action_steps = [s for s in steps if type(s).__name__ == "ActionStep"]
    step_count = len(action_steps)
    self_terminated = bool(
        action_steps and getattr(action_steps[-1], "is_final_answer", False)
    )
    hit_max_steps = step_count >= max_steps and not self_terminated
    # S88 — the shipped report is emitted by final_answer(...) inside the last
    # ActionStep's generated code. If THAT generation hit max_tokens the report
    # is cut mid-content; finish_reason == "length" is the authoritative signal.
    if generation_recovery_attempted:
        final_finish_reason = generation_recovery_finish_reason
    elif hit_max_steps:
        final_finish_reason = fallback_finish_reason
    else:
        final_finish_reason = (
            _step_finish_reason(action_steps[-1]) if action_steps else None
        )
    output_truncated = final_finish_reason == "length"
    max_steps_fallback_complete = bool(
        hit_max_steps and fallback_finish_reason == "stop"
    )

    output_wrapper_removed = recovery_wrapper_removed
    if hit_max_steps:
        output_text, max_step_wrapper_removed = _unwrap_max_steps_final_answer(
            output_text
        )
        output_wrapper_removed = bool(
            recovery_wrapper_removed or max_step_wrapper_removed
        )
        if max_step_wrapper_removed:
            logger.warning(
                "CodeAgent synthesis: normalized a literal final_answer wrapper "
                "from the max-steps fallback"
            )

    # S82 m03 / S92 m07 — stranded-draft recovery. When max-step exhaustion or
    # a mid-run generation failure leaves no cited final_answer, salvage the
    # best [cite:URL]-bearing scratchpad draft instead of an empty stub.
    # This does NOT mask the failure: hit_max_steps/self_terminated are
    # unchanged, so the outcome gate retains the max-step warning while grading
    # the recovered draft from its independent citation/truncation evidence. If
    # nothing cited is recoverable, the empty/marker-less output stands and the
    # gate fails it.
    recovered_from_scratchpad = False
    if (
        hit_max_steps or generation_failure_error is not None
    ) and not _has_cite_markers(output_text):
        salvaged = controller.retained_draft or _recover_stranded_draft(bundle.scratchpad)
        if salvaged:
            logger.warning(
                "CodeAgent synthesis: no cited final_answer after terminal failure; "
                "recovered a %d-char scratchpad draft (recovered_from_scratchpad)",
                len(salvaged),
            )
            output_text = salvaged
            recovered_from_scratchpad = True

    action_usage = {"input": 0, "output": 0, "total": 0}
    action_step_usage_complete = bool(action_steps)
    for step in action_steps:
        try:
            raw_usage = getattr(step, "token_usage", None)
        except Exception:  # noqa: BLE001 — usage remains incomplete
            raw_usage = None
        step_usage, step_usage_complete = _normalise_token_usage_envelope(raw_usage)
        action_step_usage_complete = bool(
            action_step_usage_complete and step_usage_complete
        )
        for key in ("input", "output", "total"):
            action_usage[key] += step_usage[key]

    if not run_result_usage_complete:
        if action_step_usage_complete:
            # A partial RunResult is not authoritative. Exact per-step envelopes
            # cover the same calls and must replace, not augment, its lower bound.
            cum_input = action_usage["input"]
            cum_output = action_usage["output"]
            cum_total = action_usage["total"]
        else:
            # Both views are partial and overlap. Component-wise maxima retain
            # the strongest lower bound without double-counting the same calls.
            cum_input = max(cum_input, action_usage["input"])
            cum_output = max(cum_output, action_usage["output"])
            cum_total = max(
                cum_total,
                action_usage["total"],
                cum_input + cum_output,
            )
    if generation_recovery_attempted:
        # The manual fallback is absent from failed RunResult and is not
        # appended to agent.memory. Add its usage exactly once, after action
        # memory aggregation.
        cum_input += recovery_usage_observed["input"]
        cum_output += recovery_usage_observed["output"]
        cum_total += recovery_usage_observed["total"]
    provider_observations = observed_attempts[observation_start:]
    observed_usage_complete = all(
        event.get("usage", {}).get("complete") is True for event in provider_observations
    )
    if provider_observations:
        observed_input = sum(event["usage"].get("input") or 0 for event in provider_observations)
        observed_output = sum(event["usage"].get("output") or 0 for event in provider_observations)
        observed_total = sum(event["usage"].get("total") or 0 for event in provider_observations)
        # Both surfaces are lower bounds on incomplete failure paths, never additive.
        cum_input = max(cum_input, observed_input)
        cum_output = max(cum_output, observed_output)
        cum_total = max(cum_total, observed_total, cum_input + cum_output)

    token_usage_complete = bool(
        observed_usage_complete
        and prior_attempt_usage_complete
        and (run_result_usage_complete or action_step_usage_complete)
        and (
            not generation_recovery_attempted
            or recovery_usage_available
        )
    )

    canonical_sources, source_collection_projection = _project_sources_collected(
        bundle.sources_collected
    )
    source_collection_error: str | None = None
    if source_collection_projection["dropped_records"]:
        # Preserve the old structural-only signal for compatibility while the
        # bounded projection telemetry carries the actionable reason counts.
        source_collection_error = safe_exception_text(
            SourceCollectionValidationError(
                "sources_collected projection dropped unsafe records"
            ),
            category="validation",
        )
        logger.warning(
            "CodeAgent synthesis: source projection degraded: dropped=%d "
            "reasons=%s",
            source_collection_projection["dropped_records"],
            source_collection_projection["reason_counts"],
        )
    canonical_tiers: Counter[str] = Counter(
        str(source.get("authority_tier") or "low")
        for source in canonical_sources
    )
    retrieval_state = bundle.retrieval_state_summary()
    retrieval_state["grounding_sources"] = len(canonical_sources)

    completion_kind = (
        "generation_failure_recovery" if generation_recovery_attempted else
        "step_limit_fallback" if hit_max_steps else
        "normal_final_answer" if self_terminated else "failed"
    )
    controller.observe_draft(output_text, final=True)
    return SynthesisResult(
        output_markdown=output_text,
        step_count=step_count,
        self_terminated=self_terminated,
        hit_max_steps=hit_max_steps,
        runtime_seconds=round(runtime, 2),
        token_usage={"input": cum_input, "output": cum_output, "total": cum_total},
        tool_call_summary=bundle.tool_call_summary(),
        authority_tier_distribution={
            "high": int(canonical_tiers.get("high", 0)),
            "medium": int(canonical_tiers.get("medium", 0)),
            "low": int(canonical_tiers.get("low", 0)),
        },
        pool_size=bundle.pool_size,
        step_trace=step_dicts,
        token_usage_complete=token_usage_complete,
        error=error,
        error_transient=bool(error) and error_transient,
        sources_collected=canonical_sources,
        sources_discovered=list(bundle.sources_discovered),
        sources_fetched=list(bundle.sources_fetched),
        fetch_attempts=list(bundle.fetch_attempts),
        probes=list(bundle.probes),
        scope_telemetry=bundle.scope_admission.snapshot() if bundle.scope_admission else {},
        retrieval_state=retrieval_state,
        recovered_from_scratchpad=recovered_from_scratchpad,
        output_truncated=output_truncated,
        tool_outcomes=bundle.tool_outcomes(),
        # S84 m04 review fix: read the BOUND model (the production call site
        # passes model=None and the default is built internally) — reading
        # the raw parameter left this permanently empty.
        model_id=str(getattr(bound_model, "model_id", "") or ""),
        provider_attempts=list(observed_attempts[observation_start:]),
        completion_control=controller.telemetry(completion_kind),
        effective_config=(
            dict(bound_model.effective_config)
            if isinstance(getattr(bound_model, "effective_config", None), dict) else {}
        ),
        source_collection_error=source_collection_error,
        source_collection_projection=source_collection_projection,
        output_wrapper_removed=output_wrapper_removed,
        max_steps_fallback_complete=max_steps_fallback_complete,
        generation_failure_error=generation_failure_error,
        generation_failure_reason=generation_failure_reason,
        generation_failure_reason_source=generation_failure_reason_source,
        generation_failure_root_class=generation_failure_root_class,
        generation_failure_http_status=generation_failure_http_status,
        generation_failure_transient=generation_failure_transient,
        generation_failure_step=generation_failure_step,
        generation_failure_pass=generation_failure_pass,
        generation_failure_pass_count=generation_failure_pass_count,
        generation_failure_consumed_tokens=generation_failure_consumed_tokens,
        generation_failure_retry_eligible=generation_failure_retry_eligible,
        generation_failure_retry_suppressed_reason=(
            generation_failure_retry_suppressed_reason
        ),
        generation_recovery_attempted=generation_recovery_attempted,
        generation_recovery_result=generation_recovery_result,
        generation_recovery_finish_reason=generation_recovery_finish_reason,
        generation_error_recovered=generation_error_recovered,
    )


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_API_BASE",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "SMOLAGENTS_VERBOSITY_OFF",
    "LEDGER_RECORD_MAX_ENTRIES",
    "LEDGER_RECORD_URL_MAX_CHARS",
    "SOURCE_FETCH_MAX_TIMEOUT_S",
    "SOURCE_FETCH_MAX_REDIRECTS",
    "URL_LIVENESS_MAX_TIMEOUT_S",
    "URL_LIVENESS_MAX_REQUESTS",
    "SCHOLAR_BACKEND_TIMEOUT_S",
    "SCHOLAR_MAX_BACKENDS",
    "SCRATCHPAD_RECOVERY_KEY",
    "CodeAgentRunnerError",
    "CodeAgentToolBundle",
    "SynthesisResult",
    "build_codeagent_tools",
    "build_default_model",
    "derive_authority_tier",
    "normalise_pool",
    "run_codeagent_synthesis",
]
