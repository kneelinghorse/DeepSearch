"""Unit tests for the S68 batch URL-liveness wrapper.

Per ``cmos/reports/sprint_68_approach_and_plan.md`` §"Success criteria"
and the s68-m01 test list. Mocks the S64 ``url_liveness`` primitive in the
``deepsearch.agent.citation_liveness`` namespace; the primitive itself is
covered by the S64 url-liveness suite.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import pytest

from deepsearch.agent import citation_liveness
from deepsearch.agent.citation_liveness import (
    GLOBAL_CONCURRENCY,
    LivenessTelemetry,
    _normalise_url_for_cache,
    check_citations_liveness,
)
from deepsearch.report.citation import CitationAnnotation

# ---------------------------------------------------------------------------
# Tiny helpers — keep test bodies focused on the property under test.
# ---------------------------------------------------------------------------


def _ann(url: str, *, start: int = 0, end: int = 5) -> CitationAnnotation:
    return CitationAnnotation(start_index=start, end_index=end, url=url)


def _src(url: str, title: str = "T") -> dict[str, Any]:
    return {"url": url, "title": title, "snippet": "s"}


@pytest.fixture(autouse=True)
def _fast_retry_backoff(monkeypatch: pytest.MonkeyPatch):
    """Retries still happen but the 500 ms backoff is collapsed to 0 for speed.

    Individual tests can re-patch the constant if they want to inspect timing.
    """

    monkeypatch.setattr(citation_liveness, "_RETRY_BACKOFF_S", 0.0)


def _make_url_liveness_returning(value: dict[str, Any]):
    """Sync mock returning a fixed payload, with call-count introspection."""

    calls: list[str] = []

    def fake(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
        calls.append(url)
        return {**value, "final_url": value.get("final_url") or url}

    fake.calls = calls  # type: ignore[attr-defined]
    return fake


def _alive() -> dict[str, Any]:
    return {"alive": True, "status_code": 200, "final_url": ""}


def _dead_404() -> dict[str, Any]:
    return {"alive": False, "status_code": 404, "final_url": ""}


# ---------------------------------------------------------------------------
# 1. Happy paths.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_5_citations_4_live_1_orphan(monkeypatch):
    """4 in sources → live; 1 not in sources, probed alive → orphan-live (S82 m05)."""

    fake = _make_url_liveness_returning(_alive())
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    sources = [_src(f"https://example.com/{i}") for i in range(4)]
    citations = [_ann(f"https://example.com/{i}") for i in range(4)]
    citations.append(_ann("https://orphan.example.com/x"))

    annotations, tel = await check_citations_liveness(citations, sources)

    assert isinstance(tel, LivenessTelemetry)
    assert tel.checked == 5
    assert tel.live == 4
    assert tel.dead == 0
    assert tel.orphan == 1  # set-membership (not in sources)
    assert tel.unknown == 0
    # S82 m05: hallucinated is NO LONGER an alias for orphan — the orphan was
    # probed and RESOLVED, so it is orphan-live, not fabricated.
    assert tel.hallucinated_url_count == 0
    assert tel.distinct_urls_checked == 5
    assert tel.cache_hits == 0
    assert tel.budget_exhausted is False

    # S82 m05: the orphan is now PROBED (short-circuit removed) and resolves.
    assert [a.live for a in annotations] == [True, True, True, True, True]
    assert len(fake.calls) == 5


@pytest.mark.asyncio
async def test_all_orphan_3_citations_probed_dead_are_hallucinated(monkeypatch):
    """No URL in sources + probe FAILS → 3 hallucinated (S82 m05 split).

    Pre-m05 these short-circuited to orphans with hallucinated as an alias.
    Post-m05 orphans are probed: a not-in-sources URL whose probe fails
    (NXDOMAIN/404) is true fabrication → hallucinated; orphan (set-membership)
    still counts all 3.
    """

    fake = _make_url_liveness_returning(_dead_404())
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    citations = [_ann(f"https://nope.example.com/{i}") for i in range(3)]
    annotations, tel = await check_citations_liveness(citations, [])

    assert tel.orphan == 3  # set-membership (not in sources)
    assert tel.live == 0
    assert tel.dead == 0  # dead requires in-sources; these are not in sources
    assert tel.unknown == 0
    assert tel.hallucinated_url_count == 3  # probe-failed orphans = fabrication
    assert all(a.live is False for a in annotations)
    assert len(fake.calls) == 3  # orphans are now probed (m05)


@pytest.mark.asyncio
async def test_dead_url_in_sources_collected(monkeypatch):
    """URL in sources_collected but probe says alive=False → classification=dead."""

    fake = _make_url_liveness_returning(_dead_404())
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    citations = [_ann("https://gone.example.com/x")]
    sources = [_src("https://gone.example.com/x")]

    annotations, tel = await check_citations_liveness(citations, sources)

    assert tel.dead == 1
    assert tel.live == 0
    assert tel.orphan == 0
    assert tel.unknown == 0
    assert tel.hallucinated_url_count == 0
    assert annotations[0].live is False
    assert len(fake.calls) == 1  # 4xx → no retry


@pytest.mark.asyncio
async def test_empty_citations_list(monkeypatch):
    fake = _make_url_liveness_returning(_alive())
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    annotations, tel = await check_citations_liveness([], [])
    assert annotations == []
    assert tel.checked == 0
    assert tel.live == 0
    assert tel.dead == 0
    assert tel.orphan == 0
    assert tel.unknown == 0
    assert tel.cache_hits == 0
    assert tel.budget_exhausted is False
    assert fake.calls == []


# ---------------------------------------------------------------------------
# 2. Cache + URL normalisation.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_same_url_twice(monkeypatch):
    fake = _make_url_liveness_returning(_alive())
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://example.com/page"
    citations = [_ann(url), _ann(url)]
    sources = [_src(url)]

    annotations, tel = await check_citations_liveness(citations, sources)

    assert tel.live == 1
    assert tel.cache_hits == 1
    assert tel.distinct_urls_checked == 1
    assert all(a.live is True for a in annotations)
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_url_normalisation_strips_fragment(monkeypatch):
    fake = _make_url_liveness_returning(_alive())
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    a = "https://Example.com/page#frag"
    b = "https://example.com/page"
    citations = [_ann(a), _ann(b)]
    sources = [_src(b)]

    _, tel = await check_citations_liveness(citations, sources)

    assert tel.live == 1
    assert tel.cache_hits == 1
    assert tel.distinct_urls_checked == 1


@pytest.mark.asyncio
async def test_url_normalisation_strips_tracking_params(monkeypatch):
    fake = _make_url_liveness_returning(_alive())
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    a = "https://example.com/page?utm_source=x&utm_campaign=y"
    b = "https://example.com/page"
    citations = [_ann(a), _ann(b)]
    sources = [_src(b)]

    _, tel = await check_citations_liveness(citations, sources)

    assert tel.live == 1
    assert tel.cache_hits == 1
    assert tel.distinct_urls_checked == 1


def test_normalisation_helper_strips_fbclid_gclid_ref():
    base = "https://example.com/page"
    for trail in ("?fbclid=abc", "?gclid=def", "?ref=footer", "?ref_src=share"):
        assert _normalise_url_for_cache(base + trail) == _normalise_url_for_cache(base)


def test_normalisation_helper_preserves_meaningful_query():
    a = _normalise_url_for_cache("https://example.com/page?id=1")
    b = _normalise_url_for_cache("https://example.com/page?id=2")
    assert a != b


def test_normalisation_helper_lowercases_scheme_and_host():
    assert _normalise_url_for_cache("HTTPS://Example.COM/Path") == "https://example.com/Path"


# ---------------------------------------------------------------------------
# 3. Concurrency caps.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_global_cap_at_least_10(monkeypatch):
    """10 distinct hosts run concurrently — barrier resolves only if all 10 reach it."""

    barrier = threading.Barrier(GLOBAL_CONCURRENCY, timeout=3.0)

    def fake(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
        barrier.wait()
        return {"alive": True, "status_code": 200, "final_url": url}

    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    # Python sizes its default async executor from the runner's CPU count. A
    # two-core CI runner gets fewer than ten workers, so isolate the semaphore
    # contract from that platform detail while preserving to_thread semantics.
    executor = ThreadPoolExecutor(max_workers=GLOBAL_CONCURRENCY)

    async def to_thread_at_contract_width(
        function: Any, /, *args: Any, **kwargs: Any
    ) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, partial(function, *args, **kwargs))

    monkeypatch.setattr(citation_liveness.asyncio, "to_thread", to_thread_at_contract_width)

    sources = [_src(f"https://host{i}.example.com/x") for i in range(GLOBAL_CONCURRENCY)]
    citations = [_ann(f"https://host{i}.example.com/x") for i in range(GLOBAL_CONCURRENCY)]

    try:
        _, tel = await check_citations_liveness(citations, sources)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert tel.live == GLOBAL_CONCURRENCY


@pytest.mark.asyncio
async def test_concurrency_per_host_cap_2(monkeypatch):
    """4 same-host probes — at most 2 in flight against that host at any moment."""

    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    def fake(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            if in_flight > max_in_flight:
                max_in_flight = in_flight
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        return {"alive": True, "status_code": 200, "final_url": url}

    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    host = "https://same.example.com"
    citations = [_ann(f"{host}/p{i}") for i in range(4)]
    sources = [_src(f"{host}/p{i}") for i in range(4)]

    _, tel = await check_citations_liveness(citations, sources)

    assert tel.live == 4
    assert max_in_flight <= 2, (
        f"per-host cap=2 violated: observed max_in_flight={max_in_flight}"
    )


# ---------------------------------------------------------------------------
# 4. Retry semantics.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_on_5xx_succeeds_on_second_try(monkeypatch):
    sequence = [
        {"alive": False, "status_code": 503, "final_url": ""},
        {"alive": True, "status_code": 200, "final_url": ""},
    ]
    calls: list[str] = []

    def fake(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
        calls.append(url)
        idx = min(len(calls) - 1, len(sequence) - 1)
        out = dict(sequence[idx])
        out["final_url"] = url
        return out

    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "HTTPS://Flaky.Example.COM:443/a/../x"
    canonical_url = "https://flaky.example.com/x"
    citations = [_ann(url)]
    sources = [_src(url)]

    annotations, tel = await check_citations_liveness(citations, sources)

    assert tel.live == 1
    assert tel.dead == 0
    assert tel.cache_hits == 0
    assert annotations[0].live is True
    assert len(calls) == 2
    assert tel.ledger_records == [
        {
            "tool": "url_liveness",
            "url": canonical_url,
            "status": "ok",
            "status_code": 503,
            "error_category": None,
            "alive": False,
        },
        {
            "tool": "url_liveness",
            "url": canonical_url,
            "status": "ok",
            "status_code": 200,
            "error_category": None,
            "alive": True,
        },
    ]
    assert tel.ledger_records_truncated == 0


@pytest.mark.asyncio
async def test_no_retry_on_4xx(monkeypatch):
    fake = _make_url_liveness_returning(_dead_404())
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://gone.example.com/x"
    citations = [_ann(url)]
    sources = [_src(url)]

    annotations, tel = await check_citations_liveness(citations, sources)

    assert annotations[0].live is False
    assert tel.dead == 1
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_retry_on_status_code_zero_transient(monkeypatch):
    """status_code=0 (timeout/connect-error in url_liveness) → one retry."""

    sequence = [
        {"alive": False, "status_code": 0, "final_url": ""},
        {"alive": True, "status_code": 200, "final_url": ""},
    ]
    calls: list[str] = []

    def fake(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
        calls.append(url)
        idx = min(len(calls) - 1, len(sequence) - 1)
        out = dict(sequence[idx])
        out["final_url"] = url
        return out

    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://transient.example.com/x"
    annotations, tel = await check_citations_liveness([_ann(url)], [_src(url)])

    assert annotations[0].live is True
    assert tel.live == 1
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_nonstandard_three_digit_status_preserves_both_retry_attempts(monkeypatch):
    """Bot-wall 999 responses are evidence, even though they are not registered."""

    monkeypatch.setattr(citation_liveness, "_RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(
        citation_liveness,
        "url_liveness",
        lambda url: {
            "alive": False,
            "status_code": 999,
            "final_url": url,
            "error": None,
        },
    )

    url = "https://bot-wall.example.test/source"
    annotations, telemetry = await check_citations_liveness(
        [_ann(url)],
        [_src(url)],
    )

    assert annotations[0].live is None
    assert telemetry.ledger_records_truncated == 0
    assert telemetry.ledger_records == [
        {
            "tool": "url_liveness",
            "url": url,
            "status": "ok",
            "status_code": 999,
            "error_category": None,
            "alive": False,
        },
        {
            "tool": "url_liveness",
            "url": url,
            "status": "ok",
            "status_code": 999,
            "error_category": None,
            "alive": False,
        },
    ]


# ---------------------------------------------------------------------------
# 5. Budget exhaustion.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_budget_exhausted(monkeypatch, caplog):
    def fake(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
        time.sleep(0.5)
        return {"alive": True, "status_code": 200, "final_url": url}

    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    citations = [_ann(f"https://slow.example.com/{i}") for i in range(3)]
    sources = [_src(f"https://slow.example.com/{i}") for i in range(3)]

    with caplog.at_level(logging.WARNING, logger=citation_liveness.__name__):
        annotations, tel = await check_citations_liveness(
            citations, sources, budget_seconds=0.1
        )

    assert tel.budget_exhausted is True
    assert tel.unknown >= 1
    assert tel.unknown + tel.live + tel.dead + tel.orphan == 3
    # All cancelled tasks → live=None on those annotations.
    unknown_anns = [a for a in annotations if a.live is None]
    assert len(unknown_anns) >= 1
    # logger.warning should have fired.
    assert any(
        "budget exhausted" in r.getMessage().lower() for r in caplog.records
    ), [r.getMessage() for r in caplog.records]
    # asyncio cancellation cannot stop an already-running executor request.
    # Its late result is unknowable when telemetry freezes, so the Ledger
    # envelope must fail loud rather than claim the audit was complete.
    assert tel.ledger_records_truncated >= 1


@pytest.mark.asyncio
async def test_malformed_probe_outcome_marks_ledger_incomplete(monkeypatch):
    """Classification stays no-raise while a non-strict record fails loud."""

    monkeypatch.setattr(
        citation_liveness,
        "url_liveness",
        lambda _url: {"alive": "yes", "status_code": True, "error": None},
    )

    _, telemetry = await check_citations_liveness(
        [_ann("https://malformed.example.test/source")],
        [_src("https://malformed.example.test/source")],
    )

    assert telemetry.ledger_records == []
    assert telemetry.ledger_records_truncated >= 1


# ---------------------------------------------------------------------------
# 6. Exception paths.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unexpected_exception_in_url_liveness(monkeypatch, caplog):
    """url_liveness raises despite no-raise contract → that citation = unknown."""

    def fake(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
        raise RuntimeError("credential=must-not-leak")

    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://broken.example.com/x?token=query-must-not-leak"
    with caplog.at_level(logging.ERROR, logger=citation_liveness.__name__):
        annotations, tel = await check_citations_liveness(
            [_ann(url)], [_src(url)]
        )

    assert annotations[0].live is None
    assert tel.unknown == 1
    assert tel.dead == 0
    assert tel.orphan == 0
    # mission still completes
    assert tel.checked == 1
    assert tel.ledger_records == []
    assert tel.ledger_records_truncated == 1
    # logger.error fired
    assert any(
        "url_liveness" in r.getMessage() and "RuntimeError" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]
    assert "credential" not in caplog.text
    assert "query-must-not-leak" not in caplog.text


@pytest.mark.asyncio
async def test_per_url_orchestrator_exception_marks_ledger_incomplete(
    monkeypatch,
    caplog,
):
    """The inner unknown fallthrough cannot claim lossless probe capture."""

    async def fail_probe_with_retry(_url, _ledger_capture, _scope_admission=None):
        raise RuntimeError("credential=inner-must-not-leak")

    monkeypatch.setattr(
        citation_liveness,
        "_probe_with_retry",
        fail_probe_with_retry,
    )
    url = "https://inner.example.test/source?token=query-must-not-leak"

    with caplog.at_level(logging.ERROR, logger=citation_liveness.__name__):
        annotations, telemetry = await check_citations_liveness(
            [_ann(url)],
            [_src(url)],
        )

    assert annotations[0].live is None
    assert telemetry.unknown == 1
    assert telemetry.ledger_records == []
    assert telemetry.ledger_records_truncated == 1
    assert "inner.example.test" not in caplog.text
    assert "query-must-not-leak" not in caplog.text
    assert "inner-must-not-leak" not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_outer_exception_falls_through(monkeypatch, caplog):
    """Patching a top-level helper to raise lands in the outer try/except."""

    def boom(*_a, **_kw):
        raise RuntimeError("outer-boom")

    monkeypatch.setattr(citation_liveness, "_normalise_url_for_cache", boom)

    citations = [_ann("https://x.example.com/1"), _ann("https://x.example.com/2")]
    sources = [_src("https://x.example.com/1"), _src("https://x.example.com/2")]

    with caplog.at_level(logging.ERROR, logger=citation_liveness.__name__):
        annotations, tel = await check_citations_liveness(citations, sources)

    assert all(a.live is None for a in annotations)
    assert tel.checked == 2
    assert tel.unknown == 2
    assert tel.live == 0
    assert tel.dead == 0
    assert tel.orphan == 0
    assert tel.ledger_records == []
    assert tel.ledger_records_truncated == 1
    assert any(
        "outer exception" in r.getMessage().lower() for r in caplog.records
    )


@pytest.mark.asyncio
async def test_outer_exception_after_scheduling_cancels_and_drains_tasks(
    monkeypatch,
    caplog,
):
    """A swallowed batch failure cannot leave probes or a complete audit."""

    started = threading.Event()
    release = threading.Event()
    created_tasks = []
    original_create_task = citation_liveness.asyncio.create_task

    def slow_probe(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
        started.set()
        release.wait(timeout=1.0)
        return {
            "alive": True,
            "status_code": 200,
            "final_url": url,
            "error": None,
        }

    def tracked_create_task(coro, *args, **kwargs):
        task = original_create_task(coro, *args, **kwargs)
        created_tasks.append(task)
        return task

    async def fail_wait(*_args, **_kwargs):
        while not started.is_set():
            await citation_liveness.asyncio.sleep(0)
        raise RuntimeError("token=internal-signed-value")

    monkeypatch.setattr(citation_liveness, "url_liveness", slow_probe)
    monkeypatch.setattr(
        citation_liveness.asyncio,
        "create_task",
        tracked_create_task,
    )
    monkeypatch.setattr(citation_liveness.asyncio, "wait", fail_wait)

    try:
        with caplog.at_level(logging.ERROR, logger=citation_liveness.__name__):
            annotations, telemetry = await check_citations_liveness(
                [_ann("https://pending.example.test/source")],
                [_src("https://pending.example.test/source")],
            )
    finally:
        release.set()

    assert annotations[0].live is None
    assert created_tasks and all(task.done() for task in created_tasks)
    assert telemetry.ledger_records == []
    # One lost in-flight attempt plus the swallowed structural batch failure.
    assert telemetry.ledger_records_truncated == 2
    assert "internal-signed-value" not in caplog.text


# ---------------------------------------------------------------------------
# 7. Logging + module-level state.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logger_info_summary_emitted(monkeypatch, caplog):
    fake = _make_url_liveness_returning(_alive())
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://summary.example.com/x"
    with caplog.at_level(logging.INFO, logger=citation_liveness.__name__):
        await check_citations_liveness([_ann(url)], [_src(url)])

    info_lines = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("citation liveness" in line and "checked" in line for line in info_lines), info_lines


@pytest.mark.asyncio
async def test_no_module_level_cache_state(monkeypatch):
    fake = _make_url_liveness_returning(_alive())
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://shared.example.com/x"

    _, tel1 = await check_citations_liveness([_ann(url)], [_src(url)])
    _, tel2 = await check_citations_liveness([_ann(url)], [_src(url)])

    # Each call gets a fresh cache → both probes hit the network mock.
    assert tel1.cache_hits == 0
    assert tel2.cache_hits == 0
    assert len(fake.calls) == 2


@pytest.mark.asyncio
async def test_telemetry_to_dict_shape(monkeypatch):
    """LivenessTelemetry.to_dict matches the PD7 ten-field shape."""

    fake = _make_url_liveness_returning(_alive())
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://shape.example.com/x"
    _, tel = await check_citations_liveness([_ann(url)], [_src(url)])
    data = tel.to_dict()

    assert set(data.keys()) == {
        "checked",
        "live",
        "dead",
        "orphan",
        "unknown",
        "hallucinated_url_count",
        "distinct_urls_checked",
        "wall_clock_ms",
        "cache_hits",
        "budget_exhausted",
    }
    assert isinstance(data["budget_exhausted"], bool)
    assert all(isinstance(data[k], int) for k in data if k != "budget_exhausted")
