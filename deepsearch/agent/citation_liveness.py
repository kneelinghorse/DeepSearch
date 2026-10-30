"""S68 URL liveness gate — batch wrapper around the S64 ``url_liveness`` probe.

Per ``cmos/reports/sprint_68_approach_and_plan.md`` PD3/PD4/PD5/PD7, with the
S84 m01 strong-evidence dead policy (decision #823 — the any-4xx=dead /
timeout=dead classification destroyed research the run had already fetched in
5/5 CAT runs):

* Reuses :func:`deepsearch.agent.tools.url_liveness.url_liveness` verbatim
  (HEAD-first; GET-on-405; browser-UA GET on any other 4xx so every reported
  4xx is GET-confirmed; 5s per-URL timeout; lenient 200–399; never raises).
  No new HTTP code in this module.
* Async fan-out via :class:`asyncio.Semaphore` (global ``N=10``, per-host
  ``N=2``). Sync probes run in worker threads via :func:`asyncio.to_thread`.
* Total wall-clock budget (default 45 s). Any URL not probed before the
  deadline is classified as ``unknown`` (``live=None``); we never penalise
  the agent for our budget.
* Per-call URL cache keyed by lowercased host+path with ``#fragment`` and
  known tracking params (``utm_*``, ``fbclid``, ``gclid``, ``ref``,
  ``ref_src``) stripped. **No module-level cache** — a fresh dict per call.
* S84 m01 — DEAD requires strong evidence: a GET-confirmed 404/410, or a
  connect-level failure (DNS NXDOMAIN / connection refused) on BOTH the
  first probe and the retry. Bot-wall/rate-limit answers (403/429/451, any
  other unconfirmed 4xx) are ``unknown`` with no retry; timeouts, 5xx, and
  transient failures retry once (500 ms backoff) and degrade to ``unknown``,
  never dead.
* S84 m01 — in-run body trust: a citation whose URL normalises into a
  ``sources_collected`` entry carrying a substantive body
  (≥ :data:`_SUBSTANTIVE_BODY_MIN_CHARS` chars) is LIVE regardless of what a
  probe would say — the run demonstrably read the page. Its probe is skipped.
* Mutates ``CitationAnnotation.live`` in place and returns the same list
  alongside a :class:`LivenessTelemetry` matching PD7's 10-field shape.
* Outer ``try/except`` so a top-level error still completes the mission
  with all annotations marked ``unknown`` (``live=None``).

Wired into :mod:`deepsearch.agent.codeagent_finalize_node` between the
S67 citation extractor + title resolver and the S66 critique LLM call.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AbstractSet, Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

from deepsearch.agent.quality_series import _has_readable_body
from deepsearch.agent.tools.url_liveness import url_liveness
from deepsearch.agent.url_utils import arxiv_identity
from deepsearch.mission.scope import ScopeAdmission
from deepsearch.report.citation import CitationAnnotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables — module-level constants so tests can monkeypatch the backoff to
# zero without redesigning the call site.
# ---------------------------------------------------------------------------

DEFAULT_BUDGET_SECONDS: float = 45.0
"""Total wall-clock budget for the gate. PD4."""

GLOBAL_CONCURRENCY: int = 10
"""Maximum concurrent HTTP probes across all hosts. PD4."""

PER_HOST_CONCURRENCY: int = 2
"""Maximum concurrent HTTP probes against a single host. PD4 — politeness
norm matching Common Crawl / Internet Archive defaults."""

_RETRY_BACKOFF_S: float = 0.5
"""Sleep between the first probe and the retry on 5xx / transient. PD4."""

_CITATION_HEAVY_THRESHOLD: int = 80
"""S82 m05 — above this many distinct URLs, log that the fixed wall-clock
budget may exhaust (orphans are now probed too). Observability only; the
budget remains the hard cap and unprobed URLs degrade to ``unknown``."""

_SUBSTANTIVE_BODY_MIN_CHARS: int = 200
"""S84 m01 — minimum in-run body length (chars, stripped) for the
collected-sources body-trust override: a citation whose URL was fetched
in-run with at least this much body/snippet text is LIVE regardless of probe
outcome. Sits above fixture/SERP noise (1-char snippets stay untrusted) and
below the smallest CAT-R002 false-dead body (478 chars)."""

_CONFIRMED_DEAD_STATUSES = frozenset({404, 410})
"""S84 m01 — the only HTTP statuses accepted as dead evidence. Since
``url_liveness`` GET-confirms every 4xx before reporting it, a payload
carrying one of these is a GET-confirmed gone."""

_ERROR_CONNECT = "connect"
"""Mirror of ``url_liveness``'s connect-error category (local literal to
preserve the imports-only-url_liveness AST guard). DNS NXDOMAIN / connection
refused — dead evidence only when it repeats across the retry."""

_TRACKING_PARAM_PREFIXES: Tuple[str, ...] = ("utm_",)
_TRACKING_PARAM_NAMES = frozenset({"fbclid", "gclid", "ref", "ref_src"})

_LEDGER_RECORD_MAX_ENTRIES = 1_000
_LEDGER_RECORD_URL_MAX_CHARS = 4_096
_LEDGER_RECORD_STATUS_CODE_MAX = 999
_LEDGER_ERROR_CATEGORIES = frozenset(
    {"connect", "excluded_domain", "invalid", "other", "timeout", "unexpected", "unknown"}
)
_HTTP_URL_ADAPTER = TypeAdapter(AnyHttpUrl)


# ---------------------------------------------------------------------------
# Internal probe-result dataclass + public telemetry dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LivenessProbeResult:
    """Internal per-URL outcome — the *classification* string is telemetry-
    only; only the boolean ``live`` lands on the persisted annotation.

    ``classification`` is one of:

    * ``"live"`` — probe returned ``alive=True``, or the URL carries a
      substantive in-run body in ``sources_collected`` (S84 m01 body trust —
      probe skipped).
    * ``"dead"`` — strong evidence only (S84 m01): GET-confirmed 404/410, or
      connect-level failure (NXDOMAIN / connection refused) on both the first
      probe and the retry.
    * ``"unknown"`` — everything else: bot-wall/rate-limit 4xx (403/429/451,
      any unconfirmed 4xx), timeout or 5xx surviving the retry, budget
      exhausted before the URL was probed, or ``url_liveness`` raised despite
      its no-raise contract. Never treated as dead downstream.
    """

    classification: str
    live: Optional[bool]
    status_code: int = 0
    final_url: str = ""
    retried: bool = False


@dataclass
class LivenessTelemetry:
    """The 10 fields of ``synthesis_telemetry["liveness_telemetry"]``. PD7."""

    checked: int = 0
    live: int = 0
    dead: int = 0
    orphan: int = 0
    unknown: int = 0
    hallucinated_url_count: int = 0
    distinct_urls_checked: int = 0
    wall_clock_ms: int = 0
    cache_hits: int = 0
    budget_exhausted: bool = False
    # LEDGER-2 internal transport. These fields deliberately stay out of
    # ``to_dict()`` so the public PD7 telemetry shape remains exactly ten
    # fields while finalize can retain every post-synthesis HTTP attempt.
    ledger_records: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    ledger_records_truncated: int = field(default=0, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict the finalize node merges into ``synthesis_telemetry``."""
        return {
            "checked": int(self.checked),
            "live": int(self.live),
            "dead": int(self.dead),
            "orphan": int(self.orphan),
            "unknown": int(self.unknown),
            "hallucinated_url_count": int(self.hallucinated_url_count),
            "distinct_urls_checked": int(self.distinct_urls_checked),
            "wall_clock_ms": int(self.wall_clock_ms),
            "cache_hits": int(self.cache_hits),
            "budget_exhausted": bool(self.budget_exhausted),
        }


@dataclass
class _LivenessLedgerCapture:
    """Ordered outcomes for one first-seen citation URL."""

    records: List[Dict[str, Any]] = field(default_factory=list)
    truncated: int = 0

    def append_result(self, url: str, result: Mapping[str, Any]) -> None:
        """Append one exact six-key outcome or mark the audit incomplete."""

        try:
            record = _strict_liveness_ledger_record(url, result)
        except (TypeError, ValueError):
            self.truncated += 1
            return
        self.records.append(record)


def _canonical_ledger_url(value: Any) -> str:
    """Return the canonical TraceLab HTTP(S) source identity."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("ledger URL must be non-empty text")
    normalized = value.strip()
    try:
        parsed = urlparse(normalized)
        parsed.port
    except ValueError as exc:
        raise ValueError("ledger URL has an invalid port") from exc
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("ledger URL userinfo is forbidden")
    try:
        canonical = str(_HTTP_URL_ADAPTER.validate_python(normalized))
    except ValidationError as exc:
        raise ValueError("ledger URL must be canonical HTTP(S)") from exc
    if len(canonical) > _LEDGER_RECORD_URL_MAX_CHARS:
        raise ValueError("ledger URL exceeds the 4096-character contract")
    return canonical


def _safe_error_category(value: Any, *, status: str) -> str | None:
    """Reduce a probe error to the finite secret-safe Ledger taxonomy."""

    if status == "ok":
        return None
    if not isinstance(value, str):
        return "unknown"
    category = value.strip().lower()
    return category if category in _LEDGER_ERROR_CATEGORIES else "unexpected"


def _strict_liveness_ledger_record(
    url: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project one actual HTTP probe onto TraceLab's exact six-key schema."""

    raw_status_code = result.get("status_code")
    if (
        not isinstance(raw_status_code, int)
        or isinstance(raw_status_code, bool)
        or not 0 <= raw_status_code <= _LEDGER_RECORD_STATUS_CODE_MAX
    ):
        raise ValueError("liveness status_code is outside 0..999")
    alive = result.get("alive")
    if not isinstance(alive, bool):
        raise ValueError("liveness alive verdict must be boolean")
    raw_error = result.get("error")
    status = "ok" if raw_error is None else "error"
    return {
        "tool": "url_liveness",
        "url": _canonical_ledger_url(url),
        "status": status,
        "status_code": raw_status_code,
        "error_category": _safe_error_category(raw_error, status=status),
        "alive": alive,
    }


def _flatten_ledger_captures(
    captures: Sequence[_LivenessLedgerCapture],
) -> tuple[List[Dict[str, Any]], int]:
    """Flatten by citation order, then attempt number, with a loud hard cap."""

    records: List[Dict[str, Any]] = []
    truncated = sum(capture.truncated for capture in captures)
    for capture in captures:
        for record in capture.records:
            if len(records) >= _LEDGER_RECORD_MAX_ENTRIES:
                truncated += 1
                continue
            records.append(record)
    return records, truncated


# ---------------------------------------------------------------------------
# URL normalisation — used both for cache keys and for the orphan check
# against ``sources_collected``. PD4 + risk-row "orphan check false-positives
# due to URL normalisation mismatch".
# ---------------------------------------------------------------------------


def _normalise_url_for_cache(url: str) -> str:
    """Return a canonical form that ignores fragment + tracking params.

    Lowercases scheme + host. Path, port, and remaining query are preserved
    verbatim. Returns ``""`` for falsy input.
    """

    if not url or not isinstance(url, str):
        return ""
    raw = url.strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except Exception:  # noqa: BLE001 — defensive
        return raw.lower()

    scheme = (parsed.scheme or "").lower()
    netloc = (parsed.netloc or "").lower()
    path = parsed.path or ""

    if parsed.query:
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        kept = [
            (k, v)
            for k, v in pairs
            if not any(k.lower().startswith(p) for p in _TRACKING_PARAM_PREFIXES)
            and k.lower() not in _TRACKING_PARAM_NAMES
        ]
        query = urlencode(kept)
    else:
        query = ""

    return urlunparse((scheme, netloc, path, parsed.params or "", query, ""))


def _host_of(url: str) -> str:
    """Return the lowercased hostname used as the per-host semaphore key."""
    if not url:
        return ""
    try:
        host = urlparse(url).hostname or ""
    except Exception:  # noqa: BLE001 — defensive
        return ""
    return host.lower()


# ---------------------------------------------------------------------------
# Inner probe — wraps the S64 sync ``url_liveness`` under the S84 m01
# strong-evidence dead policy. Defensive ``except Exception`` so a
# contract-violating raise surfaces as ``unknown`` rather than crashing the
# batch.
# ---------------------------------------------------------------------------


async def _run_probe(
    url: str,
    ledger_capture: _LivenessLedgerCapture,
    scope_admission: ScopeAdmission | None = None,
) -> Tuple[bool, int, str, Optional[str], bool]:
    """One ``url_liveness`` call → ``(alive, status, final_url, error, raised)``.

    ``error`` is the probe's transport category (``"timeout"`` /
    ``"connect"`` / ``"invalid"`` / ``"other"``) or ``None`` when an HTTP
    response was obtained; pre-S84 mocks without the field degrade to
    ``None`` (no strong evidence). ``raised`` marks a contract-violating
    raise from ``url_liveness``.
    """

    try:
        if scope_admission is None:
            result = await asyncio.to_thread(url_liveness, url)
        else:
            result = await asyncio.to_thread(url_liveness, url, scope_admission=scope_admission)
    except asyncio.CancelledError:
        # Cancelling the asyncio Future cannot stop an executor thread already
        # issuing HTTP. Its eventual result is unavailable once the batch
        # returns, so fail completeness loud instead of inventing an outcome.
        ledger_capture.truncated += 1
        raise
    except Exception as exc:  # noqa: BLE001 — defensive; primitive is no-raise
        # The primitive produced no outcome. Inventing ``alive=False`` would
        # turn an instrumentation failure into false rejected evidence, so the
        # only truthful representation is a loud incomplete envelope.
        ledger_capture.truncated += 1
        logger.error(
            "url_liveness raised unexpectedly (%s)",
            type(exc).__name__,
        )
        return False, 0, url, None, True

    if not isinstance(result, Mapping):
        ledger_capture.truncated += 1
        return False, 0, url, None, False
    ledger_capture.append_result(url, result)
    status = int(result.get("status_code") or 0)
    alive = bool(result.get("alive"))
    final_url = str(result.get("final_url") or url)
    error = result.get("error")
    return alive, status, final_url, (error if isinstance(error, str) else None), False


async def _probe_with_retry(
    url: str,
    ledger_capture: _LivenessLedgerCapture,
    scope_admission: ScopeAdmission | None = None,
) -> _LivenessProbeResult:
    """Classify one URL under the S84 m01 strong-evidence dead policy.

    DEAD requires strong evidence: a 404/410 whose probe chain completed
    (``error is None`` — ``url_liveness`` re-GETs every 4xx, so that payload
    is GET-confirmed), or a pure connect-level failure (DNS NXDOMAIN /
    connection refused, ``status == 0``) on BOTH the first probe and the
    retry. Bot-wall/rate-limit statuses (403/429/451, any other unconfirmed
    4xx) are UNKNOWN with no retry — a refusal to serve is not evidence the
    resource is gone; this includes payloads carrying an obtained 4xx plus a
    fallback-GET transport error. Timeouts, 5xx, and transient failures retry
    once and degrade to UNKNOWN, never dead.
    """

    alive, status, final_url, first_error, raised = await _run_probe(
        url,
        ledger_capture,
        scope_admission,
    )
    if raised:
        return _LivenessProbeResult(classification="unknown", live=None, final_url=url)
    if alive:
        return _LivenessProbeResult(
            classification="live", live=True, status_code=status, final_url=final_url
        )
    if status in _CONFIRMED_DEAD_STATUSES and first_error is None:
        # error is None means the probe chain completed — this 404/410 is the
        # confirming GET's own verdict. A 404 alongside a transport error is
        # HEAD-only evidence (the confirming GET never answered) → falls
        # through to the 4xx-unknown branch below.
        return _LivenessProbeResult(
            classification="dead", live=False, status_code=status, final_url=final_url
        )
    if 400 <= status < 500:
        # Bot-wall / rate-limit / auth-wall — either the browser-UA GET inside
        # url_liveness also got walled, or it failed at transport level (the
        # payload keeps the obtained 4xx + error category). A refusal to serve
        # is not evidence the resource is gone. Deterministic: no retry.
        return _LivenessProbeResult(
            classification="unknown",
            live=None,
            status_code=status,
            final_url=final_url,
        )

    # status 0 (timeout / connect / invalid / unexpected), 5xx, or a
    # non-standard status (e.g. 999 bot-block) — retry once, then classify
    # from the pair.
    await asyncio.sleep(_RETRY_BACKOFF_S)
    alive, status, final_url, second_error, raised = await _run_probe(
        url,
        ledger_capture,
        scope_admission,
    )
    if raised:
        return _LivenessProbeResult(
            classification="unknown", live=None, final_url=url, retried=True
        )
    if alive:
        return _LivenessProbeResult(
            classification="live",
            live=True,
            status_code=status,
            final_url=final_url,
            retried=True,
        )
    if status in _CONFIRMED_DEAD_STATUSES and second_error is None:
        return _LivenessProbeResult(
            classification="dead",
            live=False,
            status_code=status,
            final_url=final_url,
            retried=True,
        )
    if (
        status == 0
        and first_error == _ERROR_CONNECT
        and second_error == _ERROR_CONNECT
    ):
        # NXDOMAIN / connection refused on both attempts with NO HTTP response
        # ever obtained — the host is gone. (status == 0 keeps this rule off
        # any payload that carries an obtained 4xx alongside a fallback-GET
        # transport failure; those return unknown in the 4xx branch above.)
        return _LivenessProbeResult(
            classification="dead",
            live=False,
            status_code=status,
            final_url=final_url,
            retried=True,
        )
    return _LivenessProbeResult(
        classification="unknown",
        live=None,
        status_code=status,
        final_url=final_url,
        retried=True,
    )


# ---------------------------------------------------------------------------
# Public entry-point.
# ---------------------------------------------------------------------------


async def check_citations_liveness(
    citations: Sequence[CitationAnnotation],
    sources_collected: Sequence[Mapping[str, Any]] | None,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
    *,
    scope_admission: ScopeAdmission | None = None,
) -> Tuple[List[CitationAnnotation], LivenessTelemetry]:
    """Classify each citation as live / dead / orphan / unknown.

    Mutates ``CitationAnnotation.live`` in place on each entry of
    ``citations`` and returns the same list (now ``list``-typed) alongside
    a :class:`LivenessTelemetry`. Never raises — a top-level exception
    falls through with every annotation marked ``unknown`` and ``logger.
    error`` emitted.

    S84 m01 policy: a URL with a substantive in-run body in
    ``sources_collected`` is live without probing (body trust); dead requires
    strong evidence (GET-confirmed 404/410, or connect-level failure on both
    probe attempts); bot-wall/rate-limit/timeout evidence lands on unknown
    (``live=None``), which downstream quarantine and critique preserve.

    Args:
        citations: Annotations harvested by the S67 extractor (after title
            resolution).
        sources_collected: The retrieval-payload accumulator from the
            CodeAgent runner. Items must carry a ``url`` field; entries
            without one are ignored.
        budget_seconds: Total wall-clock budget. Any URL not probed before
            the deadline becomes ``unknown``. Defaults to
            :data:`DEFAULT_BUDGET_SECONDS` (45.0).

    Returns:
        Tuple ``(citations, telemetry)``. Each annotation has ``live`` set
        to ``True`` (live), ``False`` (dead-or-orphan), or ``None``
        (unknown — budget exhaustion or unexpected exception).
    """

    start = time.monotonic()
    citation_list: List[CitationAnnotation] = list(citations)
    ledger_captures: List[_LivenessLedgerCapture] = []
    tasks: Dict[asyncio.Task[Tuple[str, _LivenessProbeResult]], str] = {}

    if not citation_list:
        return citation_list, LivenessTelemetry()

    try:
        sources_url_set = build_source_url_set(sources_collected)
        substantive_url_set = _build_substantive_body_url_set(sources_collected)

        normalised_per_ann: List[str] = [
            _normalise_url_for_cache(ann.url) for ann in citation_list
        ]

        # First pass — collect the unique probe set. S82 m05 (PD135): probe
        # EVERY unique URL, including ones not in the collected-sources set.
        # Pre-m05 those short-circuited to "orphan" with NO HTTP probe, making
        # it impossible to tell a genuinely-reachable-but-uncollected URL from
        # true fabrication. Probing them yields the NXDOMAIN/404 signal that
        # splits orphan (resolves) from hallucinated (fails) in the tally below.
        # S84 m01 — in-run body trust: a URL whose sources_collected entry
        # carries a substantive body (fetched page text or a search-result
        # body) is LIVE regardless of what a probe would say — bot walls
        # routinely 403 the probe on URLs the run retrieved content for (the
        # CAT-R002 damage class). Seed the cache and skip its probe; this
        # also relieves the wall-clock budget.
        cache: Dict[str, _LivenessProbeResult] = {}
        unique_probe_urls: List[str] = []  # raw URLs (first-seen variant per normalisation key)
        scheduled_keys: set[str] = set()
        body_trusted_count = 0
        for ann, normalised in zip(citation_list, normalised_per_ann):
            if normalised in scheduled_keys:
                continue
            scheduled_keys.add(normalised)
            if scope_admission is not None and scope_admission.refusal(ann.url):
                scope_admission.reject(ann.url, "citation", "scope_source_not_allowed")
                cache[normalised] = _LivenessProbeResult(
                    classification="unknown", live=None, final_url=ann.url
                )
                continue
            if normalised in substantive_url_set:
                cache[normalised] = _LivenessProbeResult(
                    classification="live", live=True, final_url=ann.url
                )
                body_trusted_count += 1
                continue
            unique_probe_urls.append(ann.url)

        if len(unique_probe_urls) > _CITATION_HEAVY_THRESHOLD:
            logger.warning(
                "citation liveness: citation-heavy mission — %d distinct URLs to "
                "probe within a %.0fs budget; some may resolve to 'unknown' if the "
                "budget exhausts (tune budget_seconds per mission if needed)",
                len(unique_probe_urls),
                budget_seconds,
            )

        # Second pass — fan-out HTTP probes under the global + per-host
        # semaphore caps, with a single overall timeout for the whole gate.
        host_semaphores: Dict[str, asyncio.Semaphore] = {}
        global_sem = asyncio.Semaphore(GLOBAL_CONCURRENCY)

        async def _probe(
            raw_url: str,
            ledger_capture: _LivenessLedgerCapture,
        ) -> Tuple[str, _LivenessProbeResult]:
            host = _host_of(raw_url)
            host_sem = host_semaphores.setdefault(
                host, asyncio.Semaphore(PER_HOST_CONCURRENCY)
            )
            normalised = _normalise_url_for_cache(raw_url)
            try:
                async with global_sem:
                    async with host_sem:
                        result = await _probe_with_retry(raw_url, ledger_capture, scope_admission)
                return normalised, result
            except Exception as exc:  # noqa: BLE001 — defensive
                # The per-URL orchestration failed outside the individual
                # attempt recorder, so no exact outcome can be reconstructed.
                # Preserve the public unknown fallthrough but make the Ledger
                # envelope loudly incomplete.
                ledger_capture.truncated += 1
                logger.error(
                    "liveness probe inner exception (%s)",
                    type(exc).__name__,
                )
                return normalised, _LivenessProbeResult(
                    classification="unknown", live=None, final_url=raw_url
                )

        for raw_url in unique_probe_urls:
            ledger_capture = _LivenessLedgerCapture()
            ledger_captures.append(ledger_capture)
            t = asyncio.create_task(_probe(raw_url, ledger_capture))
            tasks[t] = _normalise_url_for_cache(raw_url)

        elapsed = time.monotonic() - start
        remaining = max(0.0, budget_seconds - elapsed)
        budget_exhausted = False

        if tasks:
            done, pending = await asyncio.wait(
                tasks.keys(), timeout=remaining
            )
        else:
            done, pending = set(), set()

        for t in done:
            try:
                normalised, probe_result = t.result()
                cache[normalised] = probe_result
            except Exception as exc:  # noqa: BLE001 — defensive
                normalised = tasks[t]
                logger.error(
                    "liveness probe task raised (%s)",
                    type(exc).__name__,
                )
                cache[normalised] = _LivenessProbeResult(
                    classification="unknown", live=None
                )

        unprobed_count = 0
        if pending:
            budget_exhausted = True
            for t in pending:
                t.cancel()
                normalised = tasks[t]
                if normalised not in cache:
                    cache[normalised] = _LivenessProbeResult(
                        classification="unknown", live=None
                    )
                    unprobed_count += 1
            # Drain the cancelled tasks so the event loop doesn't warn about
            # destroyed pending tasks. ``return_exceptions=True`` swallows
            # the CancelledError and any thread-side exceptions.
            try:
                await asyncio.gather(*pending, return_exceptions=True)
            except Exception:  # pragma: no cover — defense in depth
                pass

        # Third pass — apply cache to annotations and tally telemetry. The
        # first time a normalised key is seen counts as a *probe* for the
        # classification counters; subsequent occurrences count as cache
        # hits.
        live_count = 0
        dead_count = 0
        orphan_count = 0
        hallucinated_count = 0
        unknown_count = 0
        cache_hits = 0
        seen_for_counts: set[str] = set()

        for ann, normalised in zip(citation_list, normalised_per_ann):
            result = cache.get(normalised) or _LivenessProbeResult(
                classification="unknown", live=None
            )
            ann.live = result.live
        # S101 m03: which dead, uncollected URLs a collected body names needs
        # every citation's verdict first.
        source_named = source_named_dead_urls(citation_list, sources_collected)
        for ann, normalised in zip(citation_list, normalised_per_ann):
            if normalised in seen_for_counts:
                cache_hits += 1
                continue
            seen_for_counts.add(normalised)
            # S82 m05 — classify from the probe outcome + sources membership via
            # the shared classifier (critique uses the same one).
            cls = classify_citation(
                live=ann.live,
                url=ann.url,
                source_url_set=sources_url_set,
                source_named_urls=source_named,
            )
            if cls == CLASSIFICATION_LIVE:
                live_count += 1
            elif cls == CLASSIFICATION_DEAD:
                dead_count += 1
            elif cls == CLASSIFICATION_UNKNOWN:
                unknown_count += 1
            else:
                # ORPHAN (live, not in sources), HALLUCINATED (dead, not in
                # sources) or DEAD_UNCOLLECTED (dead, not in sources, named by
                # a collected body). All are set-membership orphans; only the
                # unnamed probe-failed subset is hallucinated.
                orphan_count += 1
                if cls == CLASSIFICATION_HALLUCINATED:
                    hallucinated_count += 1

        wall_clock_ms = int(round((time.monotonic() - start) * 1000))

        ledger_records, ledger_records_truncated = _flatten_ledger_captures(
            ledger_captures
        )
        telemetry = LivenessTelemetry(
            checked=len(citation_list),
            live=live_count,
            dead=dead_count,
            orphan=orphan_count,
            unknown=unknown_count,
            # S82 m05 — NO LONGER an alias for orphan_count. Counts only the
            # set-membership orphans whose HTTP probe actually FAILED
            # (NXDOMAIN/404) — true fabrication, not normalisation mismatch.
            hallucinated_url_count=hallucinated_count,
            distinct_urls_checked=len(seen_for_counts),
            wall_clock_ms=wall_clock_ms,
            cache_hits=cache_hits,
            budget_exhausted=budget_exhausted,
            ledger_records=ledger_records,
            ledger_records_truncated=ledger_records_truncated,
        )

        if budget_exhausted:
            # S84 m01 — report the URLs the budget actually cut off, not the
            # whole unknown tally (which now also holds probed bot-walls).
            logger.warning(
                "citation liveness: budget exhausted at %.2fs, %d URLs unprobed "
                "(%d unknown total)",
                wall_clock_ms / 1000.0,
                unprobed_count,
                unknown_count,
            )
        else:
            logger.info(
                "citation liveness: %d checked (%d live [%d body-trusted], "
                "%d dead, %d orphan [%d hallucinated], %d unknown), %.2fs",
                len(citation_list),
                live_count,
                body_trusted_count,
                dead_count,
                orphan_count,
                hallucinated_count,
                unknown_count,
                wall_clock_ms / 1000.0,
            )

        return citation_list, telemetry

    except Exception as exc:  # noqa: BLE001 — outer fallthrough is the contract
        logger.error(
            "citation liveness: outer exception (%s) — falling through with all unknown",
            type(exc).__name__,
        )
        # The public gate intentionally swallows internal failures, so finalize
        # cannot perform this cleanup or mark the audit incomplete for us.
        # Cancel and drain every task created by this batch. A cancelled
        # executor await adds its own per-attempt truncation in ``_run_probe``;
        # the structural ``+1`` below remains mandatory for failures before an
        # attempt starts or after a result becomes otherwise unprovable.
        if tasks:
            for task in tasks:
                if not task.done():
                    task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:  # pragma: no cover — cleanup cannot break no-raise
                pass
        for ann in citation_list:
            ann.live = None
        wall_clock_ms = int(round((time.monotonic() - start) * 1000))
        ledger_records, ledger_records_truncated = _flatten_ledger_captures(
            ledger_captures
        )
        return citation_list, LivenessTelemetry(
            checked=len(citation_list),
            unknown=len(citation_list),
            wall_clock_ms=wall_clock_ms,
            ledger_records=ledger_records,
            ledger_records_truncated=ledger_records_truncated + 1,
        )


# ---------------------------------------------------------------------------
# Citation quarantine — S84 m02: downgrade, never delete (decision #823).
#
# History: S82 m02 made DEAD removal deterministic in code; S83 m03 broadened
# the drop set to HALLUCINATED. That deletion turned out to be the AMPLIFIER
# of the CAT damage chain — false-dead verdicts became gutted reports. S84 m02
# reverses it: the drop set is HALLUCINATED only (``live is False`` AND URL
# NOT in sources — a probe-failed URL the model invented). DEAD-with-
# collected-source citations are DOWNGRADED: they stay in
# ``protocol.citations`` carrying ``live=False`` (post-m02 that value
# unambiguously means unverified-dead) and the finalize node flags their
# References entries. ORPHAN (``live is True``, not in sources — reachable
# but uncollected) and ``unknown`` (``live is None``) are preserved
# (decision #792). Gap notes are reserved for claims with NO collected
# source at all.
# ---------------------------------------------------------------------------

# Canonical critique gap-note form (mirrors critique.GAP_NOTE_DEFAULT; kept as a
# local literal to avoid a citation_liveness → critique import edge).
_DEAD_LINK_GAP_NOTE = "[Data unavailable in primary sources]"

# ``[cite:URL]`` marker (mirrors citation_extraction._CITE_MARKER_RE) — used
# only by the defensive body strip below.
_CITE_MARKER_RE = re.compile(r"\[cite:([^\[\]\s]+?)\]")


def _source_membership_key(url: str) -> str:
    """Collection-membership key: the arXiv paper identity, else the cache key.

    S98 m02 (decision 1002) — every arXiv URL form of one paper is one collected
    source. Only membership uses this key; the probe cache, body trust and the
    References audit keep :func:`_normalise_url_for_cache`.
    """
    return arxiv_identity(url) or _normalise_url_for_cache(url)


def build_source_url_set(
    sources_collected: Sequence[Mapping[str, Any]] | None,
) -> Set[str]:
    """Build the collected-sources membership set.

    The single source-set construction shared by the liveness gate, the m02
    quarantine, and (S82 m05) critique — so every consumer's orphan/dead
    classification agrees on what "in sources" means. Keys are
    :func:`_source_membership_key` values. S98 m06: a URL whose own successful
    fetch redirected to a collected source (``redirected_from``) is collected.
    """
    sources_url_set: Set[str] = set()
    for src in sources_collected or []:
        if not isinstance(src, Mapping):
            continue
        url = src.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        sources_url_set.add(_source_membership_key(url))
        aliases = src.get("redirected_from")
        for alias in aliases if isinstance(aliases, list) else []:
            if isinstance(alias, str) and alias.strip():
                sources_url_set.add(_source_membership_key(alias))
    return sources_url_set


def _build_substantive_body_url_set(
    sources_collected: Sequence[Mapping[str, Any]] | None,
) -> Set[str]:
    """Normalised URLs whose ``sources_collected`` entry has a substantive body.

    An entry qualifies only when its readable ``body`` holds at least
    :data:`_SUBSTANTIVE_BODY_MIN_CHARS` characters after stripping. Search
    snippets, liveness probes, and failed fetches never qualify — set
    membership alone is NOT body trust.

    A legacy entry flagged ``fetched_gone`` revokes body trust for the whole
    normalised URL. S87 m02 no longer writes failed fetches into grounding
    state, but preserving this read keeps older persisted protocols safe.
    """
    substantive: Set[str] = set()
    fetched_gone: Set[str] = set()
    for src in sources_collected or []:
        if not isinstance(src, Mapping):
            continue
        url = src.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        key = _normalise_url_for_cache(url)
        if src.get("fetched_gone") is True:
            fetched_gone.add(key)
        body = src.get("body") or ""
        if not isinstance(body, str):
            body = str(body)
        if len(body.strip()) >= _SUBSTANTIVE_BODY_MIN_CHARS:
            substantive.add(key)
    return substantive - fetched_gone


# ---------------------------------------------------------------------------
# S82 m05 (PD135) — shared 5-state citation classification.
#
# Pre-m05, hallucinated_url_count was a verbatim ALIAS for the set-membership
# orphan count (line ~431) and orphans short-circuited the HTTP probe entirely,
# so the metric structurally CONFLATED two very different things:
#   * a URL the model invented that does not resolve (true fabrication), and
#   * a genuinely-reachable URL we simply didn't record as a source, or one
#     that fails the normalised set-membership check (normalisation mismatch).
# m05 probes orphans too and splits the bucket:
#   ORPHAN       — not in collected sources, but the probe RESOLVES (live).
#   HALLUCINATED — not in collected sources AND the probe FAILS (NXDOMAIN/404).
# orphan_url_count (the ``orphan`` telemetry field) stays = set-membership total
# (orphan + hallucinated); hallucinated_url_count = the probe-failed subset.
#
# classify_citation is the SINGLE classifier shared by the liveness gate and
# critique._classify_citation_for_prompt, reconciling the two divergent orphan
# universes (the liveness gate normalised; critique used raw URLs pre-m05).
# ---------------------------------------------------------------------------

CLASSIFICATION_LIVE = "live"
CLASSIFICATION_DEAD = "dead"
CLASSIFICATION_ORPHAN = "orphan"
CLASSIFICATION_HALLUCINATED = "hallucinated"
CLASSIFICATION_UNKNOWN = "unknown"
# S101 m03: a dead, uncollected URL that a collected readable body names
# verbatim is dead, not fabricated. It survives, flagged unverified.
CLASSIFICATION_DEAD_UNCOLLECTED = "dead_uncollected"

_URL_BOUNDARY_OPENERS = "(<\"'[*_"
_URL_BOUNDARY_CLOSERS = ".,;:!?)]}>\"'*_"


def _named_at_url_boundary(body: str, url: str) -> bool:
    """Whether ``body`` holds the exact ``url`` string at a URL boundary.

    Before it: the start of the text, whitespace, or one of ``(<"'[*_``.
    After it: the end of the text, whitespace, or a run of ``.,;:!?)]}>"'*_``
    ending at either, where ``)`` or ``]`` does not close a URL holding more
    openers than closers. A URL followed directly by a letter is a safe miss.
    """
    closers = _URL_BOUNDARY_CLOSERS
    if url.count("(") > url.count(")"):
        closers = closers.replace(")", "")
    if url.count("[") > url.count("]"):
        closers = closers.replace("]", "")
    start = body.find(url)
    while start != -1:
        end = start + len(url)
        while end < len(body) and body[end] in closers:
            end += 1
        if (
            (start == 0 or body[start - 1].isspace() or body[start - 1] in _URL_BOUNDARY_OPENERS)
            and (end == len(body) or body[end].isspace())
        ):
            return True
        start = body.find(url, start + 1)
    return False


def _ends_in_stray_closer(url: str) -> bool:
    """Whether a cited URL ends in a closing character it does not itself open.

    ``…/Foo_(bar)`` balances its parenthesis; ``…/a)`` or ``…/a,`` carries a
    stray character, a miscopy whose 404 may come from that character alone.
    """
    last = url[-1:]
    if last == ")":
        return url.count("(") < url.count(")")
    if last == "]":
        return url.count("[") < url.count("]")
    return bool(last) and last in _URL_BOUNDARY_CLOSERS


def source_named_dead_urls(
    citations: Sequence[CitationAnnotation],
    sources_collected: Sequence[Mapping[str, Any]] | None,
) -> Set[str]:
    """Cited dead, uncollected URLs that a collected readable body names (S101 m03).

    A cited URL qualifies when its citation has ``live is False`` (decision
    824's dead policy), it is not collected under the membership key, and the
    exact string the agent cited (as written, or its canonical form) occurs at
    a URL boundary in the ``body`` of a source passing ``_has_readable_body``.
    A search snippet never qualifies, no URL is normalized for the match, and
    apex and ``www`` stay distinct. A URL ending in a stray closing character
    never qualifies. A named URL's normalized twins (fragment, tracking
    parameters, scheme or host case), which the probe cache and the References
    audit already read as one URL, share its class.
    """
    source_url_set = build_source_url_set(sources_collected)
    forms: Dict[str, Set[str]] = {}
    for annotation in citations:
        if annotation.live is not False or (
            _source_membership_key(annotation.url) in source_url_set
        ):
            continue
        written = annotation._cited_url or annotation.url
        forms.setdefault(annotation.url, set()).update(
            literal for literal in {annotation.url, written}
            if not _ends_in_stray_closer(literal)
        )
    if not any(forms.values()):
        return set()
    bodies = [
        source["body"]
        for source in sources_collected or []
        if isinstance(source, Mapping) and isinstance(source.get("body"), str)
        and _has_readable_body(source)
    ]
    named_keys = {
        _normalise_url_for_cache(url)
        for url, literals in forms.items()
        if any(_named_at_url_boundary(body, literal) for literal in literals for body in bodies)
    }
    return {url for url in forms if _normalise_url_for_cache(url) in named_keys}


def classify_citation(
    *,
    live: Optional[bool],
    url: str,
    source_url_set: Set[str],
    source_named_urls: AbstractSet[str] = frozenset(),
) -> str:
    """Classify a citation from its probe outcome + collected-sources membership.

    ``live`` is the probe result on the annotation (True/False/None). Returns
    one of the ``CLASSIFICATION_*`` constants. Set-membership uses the SAME
    key (:func:`_source_membership_key`) as :func:`build_source_url_set`, so
    the gate and critique never disagree. ``source_named_urls`` comes from
    :func:`source_named_dead_urls` over the same citations and sources: such
    a URL is dead-uncollected rather than hallucinated.
    """
    if live is None:
        return CLASSIFICATION_UNKNOWN
    in_sources = _source_membership_key(url) in source_url_set
    if live:
        return CLASSIFICATION_LIVE if in_sources else CLASSIFICATION_ORPHAN
    if in_sources:
        return CLASSIFICATION_DEAD
    if url in source_named_urls:
        return CLASSIFICATION_DEAD_UNCOLLECTED
    return CLASSIFICATION_HALLUCINATED


def quarantine_dead_citations(
    citations: Sequence[CitationAnnotation],
    sources_collected: Sequence[Mapping[str, Any]] | None,
) -> Tuple[List[CitationAnnotation], List[CitationAnnotation]]:
    """Partition citations into ``(surviving, quarantined)``.

    S84 m02 (decision #823 — quarantine was the amplifier that turned
    false-dead verdicts into gutted reports): the drop set narrows to
    HALLUCINATED only. A citation is quarantined ONLY when the shared
    classifier returns HALLUCINATED — ``live is False`` AND the URL is NOT
    in ``sources_collected`` (a probe-failed URL the model invented; a true
    fabrication never reaches ``protocol.citations``).

    DEAD (``live is False`` AND URL in ``sources_collected`` — genuinely
    retrieved, GET-confirmed gone under the s84-m01 policy) is DOWNGRADED,
    not deleted: the citation survives into ``protocol.citations`` carrying
    ``live=False``, which post-m02 unambiguously means unverified-dead (the
    finalize node flags it in the References list and counts it as
    ``markers_quarantined``). ORPHAN (``live is True``, not in sources) and
    UNKNOWN (``live is None``) survive unchanged (decision #792 / S83 m03).

    The function keeps its historical name — the call-site line in the
    finalize node is source-text-pinned by the S82/S83 wiring guards.
    Routing the decision through :func:`classify_citation` keeps the
    dead-vs-hallucinated split on the SAME classifier the gate + critique
    use. Pure function — annotations are reused by reference; no list is
    mutated.
    """
    citation_list = list(citations)
    if not citation_list:
        return citation_list, []

    sources_url_set = build_source_url_set(sources_collected)
    source_named = source_named_dead_urls(citation_list, sources_collected)

    surviving: List[CitationAnnotation] = []
    quarantined: List[CitationAnnotation] = []
    for ann in citation_list:
        cls = classify_citation(
            live=ann.live,
            url=ann.url,
            source_url_set=sources_url_set,
            source_named_urls=source_named,
        )
        if cls == CLASSIFICATION_HALLUCINATED:
            quarantined.append(ann)
        else:
            surviving.append(ann)
    return surviving, quarantined


def strip_dead_cite_markers(
    markdown: str,
    quarantined_citations: Sequence[CitationAnnotation],
    sources_collected: Sequence[Mapping[str, Any]] | None,
) -> Tuple[str, int]:
    """Annotate-mode leaked-marker cleanup (S84 m02).

    Normally a no-op: ``extract_citations`` strips ``[cite:URL]`` markers
    pre-critique, so none survive into the final markdown. This is
    belt-and-braces for a marker that ever leaks through (critique
    re-emission, a marker inside a code span the extractor skipped):

    * A leaked marker for a ``quarantined_citations`` URL that has NO
      collected source (HALLUCINATED — the only class the quarantine still
      drops) is replaced with the canonical gap note.
    * EVERY other leaked marker — including a DEAD-with-collected-source
      URL — is silently stripped, exactly like the pre-critique extractor
      would have done. The claim text survives untouched; the gap note is
      never inserted for a claim that has a collected source (s84-m02
      criterion 1). The unverified flagging for downgraded-dead citations
      happens in the finalize References pass, not here.

    ``sources_collected`` guards the gap-note path class-structurally: even
    a wrong caller cannot gap-note a collected-source URL. Returns
    ``(markdown, acted_count)`` counting gap-noted + stripped markers.
    """
    if not markdown or "[cite:" not in markdown:
        return markdown, 0

    sources_url_set = build_source_url_set(sources_collected)
    quarantined_normalised: Set[str] = {
        _normalise_url_for_cache(ann.url) for ann in quarantined_citations
    }
    acted = 0

    def _replace(match: "re.Match[str]") -> str:
        nonlocal acted
        acted += 1
        url = match.group(1)
        if (
            _normalise_url_for_cache(url) in quarantined_normalised
            and _source_membership_key(url) not in sources_url_set
        ):
            return _DEAD_LINK_GAP_NOTE
        return ""

    return _CITE_MARKER_RE.sub(_replace, markdown), acted


__all__ = [
    "DEFAULT_BUDGET_SECONDS",
    "GLOBAL_CONCURRENCY",
    "PER_HOST_CONCURRENCY",
    "CLASSIFICATION_LIVE",
    "CLASSIFICATION_DEAD",
    "CLASSIFICATION_ORPHAN",
    "CLASSIFICATION_HALLUCINATED",
    "CLASSIFICATION_UNKNOWN",
    "CLASSIFICATION_DEAD_UNCOLLECTED",
    "LivenessTelemetry",
    "build_source_url_set",
    "classify_citation",
    "check_citations_liveness",
    "quarantine_dead_citations",
    "source_named_dead_urls",
    "strip_dead_cite_markers",
]
