"""S84 m01 invariants — liveness false-dead reclassification (decision #823).

WHY THIS FILE EXISTS: the 2026-07-16 fix-or-retire deep dive proved one
damage chain gutted 5/5 CAT runs — bot-UA HEAD probe → any 4xx/timeout =
deterministic dead → quarantine deletes citations → critique replaces claims
with gap notes → recall-biased gate certifies the husk. CAT-R002 alone lost
11 citations whose URLs ALL had substantive fetched bodies in
``sources_collected`` in the same run. These pins make that class of
destruction impossible to reintroduce silently:

1. The CAT-R002 11-URL fixture table — every URL the old policy killed must
   classify live (via in-run body trust), with zero probes issued and zero
   quarantine drops.
2. The strong-evidence dead policy — DEAD only on GET-confirmed 404/410 or
   connect-level failure on both probe attempts; every bot-wall/rate-limit/
   timeout signal lands on unknown, which quarantine and critique preserve.
3. The nngroup/lowes/zoro-style scenarios from the deep dive (the probe-level
   403→browser-UA-GET rescue is pinned in tests/unit/agent/tools/
   test_url_liveness.py; here we pin the batch-gate verdicts).
4. HALLUCINATED stays intact: a fabricated URL with a GET-confirmed 404 is
   still quarantined — only the DEAD verdict semantics changed.

New pins live in this NEW file because test_s83_m05_invariants pins the
collected count of the six s82 files at exactly 70.
"""

from __future__ import annotations

from typing import Any

import pytest

from deepsearch.agent import citation_liveness
from deepsearch.agent.citation_liveness import (
    _SUBSTANTIVE_BODY_MIN_CHARS,
    _build_substantive_body_url_set,
    check_citations_liveness,
    quarantine_dead_citations,
)
from deepsearch.report.citation import CitationAnnotation

# ---------------------------------------------------------------------------
# Helpers (mirroring test_citation_liveness.py conventions).
# ---------------------------------------------------------------------------


def _ann(url: str, *, start: int = 0, end: int = 5) -> CitationAnnotation:
    return CitationAnnotation(start_index=start, end_index=end, url=url)


@pytest.fixture(autouse=True)
def _fast_retry_backoff(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(citation_liveness, "_RETRY_BACKOFF_S", 0.0)


def _make_probe_returning(value: dict[str, Any]):
    """URL-agnostic sync mock with call introspection."""

    calls: list[str] = []

    def fake(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
        calls.append(url)
        return {**value, "final_url": value.get("final_url") or url}

    fake.calls = calls  # type: ignore[attr-defined]
    return fake


def _make_probe_sequence(payloads: list[dict[str, Any]]):
    """Sync mock returning payloads in call order (last one repeats)."""

    calls: list[str] = []

    def fake(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
        calls.append(url)
        out = dict(payloads[min(len(calls) - 1, len(payloads) - 1)])
        out["final_url"] = out.get("final_url") or url
        return out

    fake.calls = calls  # type: ignore[attr-defined]
    return fake


def _bot_wall(status: int) -> dict[str, Any]:
    return {"alive": False, "status_code": status, "final_url": "", "error": None}


def _transport(error: str) -> dict[str, Any]:
    return {"alive": False, "status_code": 0, "final_url": "", "error": error}


# ---------------------------------------------------------------------------
# 1. CAT-R002 fixture table — the 11 URLs the old policy killed.
#
# Source of record: forensic workflow wf_b1404b38-e16 (agent journal) +
# cached TraceLab execution record 50fdf0ae-61cd-45ba-b75c-1763dd23468f.
# Per-URL probe statuses were never persisted (telemetry stores counts only),
# so the mock feeds representative kill signals from the R001/R005 probe
# evidence: bot-wall 403/429/451 and timeout. Body char counts are the
# measured in-run body sizes.
# ---------------------------------------------------------------------------

CAT_R002_FALSE_DEAD: list[tuple[str, int]] = [
    ("https://www.partselect.com/Find-Your-Model-Number/", 1491),
    ("https://www.partselect.com/Appliance-Parts.htm", 1467),
    ("https://www.repairclinic.com/Repair-Library/Article/46/Tips-To-Find-Your-Model-Number", 1560),  # noqa: E501
    ("https://www.supplyhouse.com/Boiler-Parts-Finder-Tool", 1196),
    ("https://www.amazon.com/gp/help/customer/display.html?nodeId=GTTH6M3A9A3W8QUC", 557),
    ("https://www.motor.com/2023/11/auto-parts-return-rates-rank-as-the-highest-in-retail-how-e-commerce-options-aim-to-drop-that-number/", 794),  # noqa: E501
    ("https://hedgescompany.com/blog/2025/09/automotive-part-number-interchanges/", 1393),
    ("https://partsmax.co/blogs/news/installation-challenges-common-fitment-issues-with-aftermarket-parts", 1851),  # noqa: E501
    ("https://www.fyresite.com/aces-pies-sema-data/", 3547),
    ("https://www.darksidedevelopments.co.uk/Blog/vw-part-numbers-explained/", 1279),
    ("https://help.motilitysoftware.com/hc/en-us/articles/360060526752-Parts-Supersession-Chain-Succession-Parts", 478),  # noqa: E501
]

_KILL_SIGNALS: list[dict[str, Any]] = [
    _bot_wall(403),
    _bot_wall(429),
    _bot_wall(451),
    _transport("timeout"),
]


def _cat_r002_sources() -> list[dict[str, Any]]:
    return [
        {"url": url, "title": "T", "snippet": "x" * chars, "body": "x" * chars}
        for url, chars in CAT_R002_FALSE_DEAD
    ]


@pytest.mark.asyncio
async def test_cat_r002_fixture_table_zero_dead(monkeypatch):
    """Success criterion 3: the 11 CAT-R002 false-dead URLs classify live —
    zero dead — even when every probe answer is a kill signal the old policy
    treated as dead. Body trust must not even issue the probes."""

    calls: list[str] = []

    def fake(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
        calls.append(url)
        idx = len(calls) - 1
        out = dict(_KILL_SIGNALS[idx % len(_KILL_SIGNALS)])
        out["final_url"] = url
        return out

    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    sources = _cat_r002_sources()
    citations = [_ann(url) for url, _ in CAT_R002_FALSE_DEAD]

    annotations, tel = await check_citations_liveness(citations, sources)

    assert tel.dead == 0, "a CAT-R002 URL classified dead — the damage class is back"
    assert tel.live == len(CAT_R002_FALSE_DEAD)
    assert tel.unknown == 0
    assert tel.orphan == 0
    assert tel.hallucinated_url_count == 0
    assert [a.live for a in annotations] == [True] * len(CAT_R002_FALSE_DEAD)
    # Body trust skips the probe entirely — the kill signals were never
    # consulted, so this cannot regress via probe-policy drift either.
    assert calls == []


@pytest.mark.asyncio
async def test_cat_r002_quarantine_drops_nothing(monkeypatch):
    """End of the damage chain: after classification, quarantine must keep
    all 11 citations (pre-m01 it deleted them and critique gap-noted the
    claims)."""

    monkeypatch.setattr(
        citation_liveness, "url_liveness", _make_probe_returning(_bot_wall(403))
    )

    sources = _cat_r002_sources()
    citations = [_ann(url) for url, _ in CAT_R002_FALSE_DEAD]

    annotations, _ = await check_citations_liveness(citations, sources)
    surviving, quarantined = quarantine_dead_citations(annotations, sources)

    assert len(surviving) == len(CAT_R002_FALSE_DEAD)
    assert quarantined == []


def test_cat_r002_bodies_all_clear_the_substance_floor():
    """The fixture table stays meaningful only while every recorded body
    clears the floor — and the floor stays above trivial snippet noise."""

    assert all(chars >= _SUBSTANTIVE_BODY_MIN_CHARS for _, chars in CAT_R002_FALSE_DEAD)
    assert _SUBSTANTIVE_BODY_MIN_CHARS == 200
    # Smallest CAT-R002 body (motilitysoftware, 478 chars) has headroom.
    assert min(chars for _, chars in CAT_R002_FALSE_DEAD) == 478 > _SUBSTANTIVE_BODY_MIN_CHARS


# ---------------------------------------------------------------------------
# 2. Strong-evidence dead policy — success criterion 1: no 4xx-probed URL is
# dead without a GET-confirmed 404/410.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 405, 406, 418, 429, 451, 499])
async def test_unconfirmed_4xx_is_unknown_never_dead(monkeypatch, status):
    """Every non-404/410 4xx — bot wall, rate limit, legal block, auth wall —
    lands on unknown (live=None) with NO retry, and survives quarantine.
    Pre-m01 every one of these was deterministic dead."""

    fake = _make_probe_returning(_bot_wall(status))
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://wall.example.com/x"
    sources = [{"url": url, "title": "T", "snippet": "s"}]

    annotations, tel = await check_citations_liveness([_ann(url)], sources)

    assert annotations[0].live is None
    assert tel.unknown == 1
    assert tel.dead == 0
    assert len(fake.calls) == 1  # deterministic bot-wall answer: no retry

    surviving, quarantined = quarantine_dead_citations(annotations, sources)
    assert len(surviving) == 1 and quarantined == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 410])
async def test_get_confirmed_404_410_stays_dead(monkeypatch, status):
    """The probe GET-confirms every 4xx before reporting it, so a 404/410
    payload IS the strong evidence — dead, no retry. S84 m02: the dead
    citation is DOWNGRADED (kept, live=False = unverified), never deleted."""

    fake = _make_probe_returning(_bot_wall(status))
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://gone.example.com/x"
    sources = [{"url": url, "title": "T", "snippet": "s"}]

    annotations, tel = await check_citations_liveness([_ann(url)], sources)

    assert annotations[0].live is False
    assert tel.dead == 1
    assert len(fake.calls) == 1

    surviving, quarantined = quarantine_dead_citations(annotations, sources)
    assert len(surviving) == 1 and quarantined == []
    assert surviving[0].live is False  # the unverified flag


@pytest.mark.asyncio
async def test_timeout_after_retry_is_unknown_never_dead(monkeypatch):
    """Timeout twice → unknown. Pre-m01: retry then DEAD — a slow host cost
    the mission its citation."""

    fake = _make_probe_returning(_transport("timeout"))
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://slow.example.com/x"
    annotations, tel = await check_citations_liveness(
        [_ann(url)], [{"url": url, "title": "T", "snippet": "s"}]
    )

    assert annotations[0].live is None
    assert tel.unknown == 1
    assert tel.dead == 0
    assert len(fake.calls) == 2  # transient signals still get the one retry


@pytest.mark.asyncio
async def test_5xx_after_retry_is_unknown_never_dead(monkeypatch):
    """5xx twice → unknown (a broken server is not a gone resource).
    Pre-m01 this tail was dead and was never pinned."""

    fake = _make_probe_returning(
        {"alive": False, "status_code": 503, "final_url": "", "error": None}
    )
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://flaky.example.com/x"
    annotations, tel = await check_citations_liveness(
        [_ann(url)], [{"url": url, "title": "T", "snippet": "s"}]
    )

    assert annotations[0].live is None
    assert tel.unknown == 1
    assert tel.dead == 0
    assert len(fake.calls) == 2


@pytest.mark.asyncio
async def test_connect_failure_on_both_attempts_is_dead(monkeypatch):
    """NXDOMAIN / connection refused twice IS strong dead evidence — the
    host is gone. This keeps true link-rot detection alive under the new
    policy."""

    fake = _make_probe_returning(_transport("connect"))
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://nxdomain.example.invalid/x"
    sources = [{"url": url, "title": "T", "snippet": "s"}]
    annotations, tel = await check_citations_liveness([_ann(url)], sources)

    assert annotations[0].live is False
    assert tel.dead == 1
    assert len(fake.calls) == 2  # the confirming retry is mandatory


@pytest.mark.asyncio
async def test_connect_then_timeout_is_unknown(monkeypatch):
    """A flapping failure mode (connect error, then timeout) is NOT strong
    evidence — only a repeated connect-level failure is."""

    fake = _make_probe_sequence([_transport("connect"), _transport("timeout")])
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://flap.example.com/x"
    annotations, tel = await check_citations_liveness(
        [_ann(url)], [{"url": url, "title": "T", "snippet": "s"}]
    )

    assert annotations[0].live is None
    assert tel.unknown == 1
    assert tel.dead == 0


@pytest.mark.asyncio
async def test_legacy_status_zero_without_error_field_is_unknown(monkeypatch):
    """A status-0 payload WITHOUT the S84 error field (pre-S84 mock shape, or
    a defensive fallback) carries no strong evidence → unknown, never dead."""

    fake = _make_probe_returning({"alive": False, "status_code": 0, "final_url": ""})
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://legacy.example.com/x"
    annotations, tel = await check_citations_liveness(
        [_ann(url)], [{"url": url, "title": "T", "snippet": "s"}]
    )

    assert annotations[0].live is None
    assert tel.unknown == 1
    assert tel.dead == 0


@pytest.mark.asyncio
async def test_nonstandard_status_999_is_unknown(monkeypatch):
    """LinkedIn-style 999 bot blocks are neither alive nor dead evidence."""

    fake = _make_probe_returning(
        {"alive": False, "status_code": 999, "final_url": "", "error": None}
    )
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://www.linkedin.com/pulse/x"
    annotations, tel = await check_citations_liveness(
        [_ann(url)], [{"url": url, "title": "T", "snippet": "s"}]
    )

    assert annotations[0].live is None
    assert tel.unknown == 1
    assert tel.dead == 0


@pytest.mark.asyncio
async def test_bot_wall_403_then_refused_get_is_unknown_never_dead(monkeypatch):
    """ADVERSARIAL-REVIEW FIX PIN: a wall that 403s the HEAD then refuses the
    fallback GET's connection produces {status 403, error 'connect'} — which
    must land on unknown via the 4xx branch, NEVER reach the connect-twice
    dead rule. Pre-fix the probe reported {status 0, error 'connect'} here
    and two attempts classified DEAD — a 4xx-probed URL deleted without any
    GET-confirmed 404/410 (the exact criterion-1 violation)."""

    fake = _make_probe_returning(
        {"alive": False, "status_code": 403, "final_url": "", "error": "connect"}
    )
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://escalating-wall.example.com/x"
    sources = [{"url": url, "title": "T", "snippet": "s"}]
    annotations, tel = await check_citations_liveness([_ann(url)], sources)

    assert annotations[0].live is None
    assert tel.unknown == 1
    assert tel.dead == 0
    assert len(fake.calls) == 1  # deterministic 4xx answer: no retry

    surviving, quarantined = quarantine_dead_citations(annotations, sources)
    assert len(surviving) == 1 and quarantined == []


@pytest.mark.asyncio
async def test_head_only_404_with_transport_error_is_unknown(monkeypatch):
    """A 404 alongside a transport error is HEAD-only evidence — the
    confirming GET never answered — so it must NOT classify dead."""

    fake = _make_probe_returning(
        {"alive": False, "status_code": 404, "final_url": "", "error": "timeout"}
    )
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://tarpit-404.example.com/x"
    annotations, tel = await check_citations_liveness(
        [_ann(url)], [{"url": url, "title": "T", "snippet": "s"}]
    )

    assert annotations[0].live is None
    assert tel.dead == 0
    assert tel.unknown == 1


def test_error_connect_constant_matches_probe_module():
    """The wrapper's _ERROR_CONNECT literal mirrors url_liveness.ERROR_CONNECT
    (kept local to satisfy the imports-only AST guard) — a rename in either
    module would silently disable connect-twice dead detection."""

    import deepsearch.agent.tools.url_liveness as url_liveness_mod

    assert citation_liveness._ERROR_CONNECT == url_liveness_mod.ERROR_CONNECT


# ---------------------------------------------------------------------------
# 3. Deep-dive scenario pins (nngroup / lowes / zoro style — deliverable 5).
# The probe-level 403-HEAD→200-browser-GET rescue is pinned in
# tests/unit/agent/tools/test_url_liveness.py; these pin the batch verdicts.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bot_wall_rescued_by_browser_get_is_live(monkeypatch):
    """nngroup-style: the probe's browser-UA GET rescue already flipped the
    payload to alive — the gate lands on live and quarantine keeps it."""

    fake = _make_probe_returning(
        {"alive": True, "status_code": 200, "final_url": "", "error": None}
    )
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://www.nngroup.com/articles/faceted-search/"
    sources = [{"url": url, "title": "NN/g", "snippet": "s"}]
    annotations, tel = await check_citations_liveness([_ann(url)], sources)

    assert annotations[0].live is True
    assert tel.live == 1


@pytest.mark.asyncio
async def test_bot_wall_on_both_uas_with_in_run_body_is_live(monkeypatch):
    """zoro/lowes-style: 403 to BOTH user agents, but the run fetched the page
    — the in-run body wins and the probe is never issued."""

    fake = _make_probe_returning(_bot_wall(403))
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://www.zoro.com/supply-chain-insights"
    sources = [
        {
            "url": url,
            "title": "Zoro",
            "snippet": "short search preview",
            "body": "z" * _SUBSTANTIVE_BODY_MIN_CHARS,
            "content_quality": "readable_body",
        }
    ]
    annotations, tel = await check_citations_liveness([_ann(url)], sources)

    assert annotations[0].live is True
    assert tel.live == 1
    assert tel.dead == 0
    assert fake.calls == []  # body trust skipped the probe


@pytest.mark.asyncio
async def test_bot_wall_on_both_uas_without_body_is_unknown(monkeypatch):
    """Same double-403 but the run never fetched the page: unknown — enough
    to preserve the citation, not enough to call it verified."""

    fake = _make_probe_returning(_bot_wall(403))
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://www.lowes.com/l/about"
    sources = [{"url": url, "title": "Lowes", "snippet": "s"}]
    annotations, tel = await check_citations_liveness([_ann(url)], sources)

    assert annotations[0].live is None
    assert tel.unknown == 1


@pytest.mark.asyncio
async def test_true_404_in_sources_still_classifies_dead(monkeypatch):
    """A genuinely gone URL (GET-confirmed 404) with only a thin snippet in
    sources still CLASSIFIES dead — recall bias must not blur the verdict.
    S84 m02: the verdict downgrades (kept, live=False = unverified) instead
    of deleting; hard quarantine is reserved for HALLUCINATED."""

    fake = _make_probe_returning(_bot_wall(404))
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://real-404.example.com/x"
    sources = [{"url": url, "title": "T", "snippet": "s"}]
    annotations, tel = await check_citations_liveness([_ann(url)], sources)

    assert annotations[0].live is False
    assert tel.dead == 1
    surviving, quarantined = quarantine_dead_citations(annotations, sources)
    assert len(surviving) == 1 and quarantined == []
    assert surviving[0].live is False  # the unverified flag


# ---------------------------------------------------------------------------
# 4. HALLUCINATED intact (mission hard requirement) + the documented m01
# trade-off for bot-blocked fabrications.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hallucinated_confirmed_404_still_quarantined(monkeypatch):
    """live=False AND not-in-sources (GET-confirmed 404) must keep the S83 m03
    hard quarantine — only DEAD verdict semantics changed in m01."""

    fake = _make_probe_returning(_bot_wall(404))
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://fabricated.example.com/paper"
    annotations, tel = await check_citations_liveness([_ann(url)], [])

    assert annotations[0].live is False
    assert tel.orphan == 1  # set-membership total
    assert tel.hallucinated_url_count == 1

    surviving, quarantined = quarantine_dead_citations(annotations, [])
    assert surviving == [] and len(quarantined) == 1


@pytest.mark.asyncio
async def test_bot_blocked_fabrication_is_unknown_documented_tradeoff(monkeypatch):
    """DOCUMENTED m01 TRADE-OFF: a fabricated URL answering 403 is unknown
    (preserved) — never-dead-on-bot-wall applies to every URL. The
    hallucination signal for this subset is deferred to the s84-m02
    unverified tier + S85 critique containment. If this pin breaks because
    the URL got quarantined, the never-dead-on-bot-wall guarantee broke."""

    fake = _make_probe_returning(_bot_wall(403))
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://fabricated-wall.example.com/paper"
    annotations, tel = await check_citations_liveness([_ann(url)], [])

    assert annotations[0].live is None
    assert tel.unknown == 1
    assert tel.hallucinated_url_count == 0

    surviving, quarantined = quarantine_dead_citations(annotations, [])
    assert len(surviving) == 1 and quarantined == []


# ---------------------------------------------------------------------------
# 5. Body-trust floor boundaries.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_body_at_floor_is_trusted_below_floor_is_probed(monkeypatch):
    """Exactly-at-floor body → live without probe; one char short → the probe
    runs and its (bot-wall) answer lands on unknown."""

    fake = _make_probe_returning(_bot_wall(403))
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    at_floor = "https://at-floor.example.com/x"
    below_floor = "https://below-floor.example.com/x"
    sources = [
        {"url": at_floor, "title": "T", "body": "a" * _SUBSTANTIVE_BODY_MIN_CHARS},
        {"url": below_floor, "title": "T", "body": "b" * (_SUBSTANTIVE_BODY_MIN_CHARS - 1)},
    ]

    annotations, tel = await check_citations_liveness(
        [_ann(at_floor), _ann(below_floor)], sources
    )

    assert annotations[0].live is True
    assert annotations[1].live is None
    assert tel.live == 1 and tel.unknown == 1
    assert fake.calls == [below_floor]


def test_liveness_probe_only_entry_is_not_substantive():
    """A url_liveness-tool accumulator entry (empty snippet, alive flag, no
    body) must NOT qualify as body trust — set membership alone is not
    evidence the run read the page. Whitespace padding doesn't count either."""

    entries = [
        {"url": "https://probed-only.example.com/x", "title": "", "snippet": "", "alive": True},
        {"url": "https://ws.example.com/x", "snippet": " " * (_SUBSTANTIVE_BODY_MIN_CHARS + 50)},
        {"url": "https://real.example.com/x", "body": "r" * _SUBSTANTIVE_BODY_MIN_CHARS},
    ]
    out = _build_substantive_body_url_set(entries)
    assert out == {citation_liveness._normalise_url_for_cache("https://real.example.com/x")}


@pytest.mark.asyncio
async def test_in_run_fetched_gone_revokes_body_trust(monkeypatch):
    """ADVERSARIAL-REVIEW FIX PIN: the run's own GET evidence outranks a SERP
    snippet. A web_search entry left a 500-char body on the URL, but an
    in-run source_fetch saw 404 (fetched_gone marker) — body trust must NOT
    apply; the probe runs and its GET-confirmed 404 classifies dead."""

    fake = _make_probe_returning(_bot_wall(404))
    monkeypatch.setattr(citation_liveness, "url_liveness", fake)

    url = "https://stale-serp.example.com/deleted-page"
    sources = [
        {"url": url, "title": "T", "snippet": "x" * 500, "body": "x" * 500, "fetched_gone": True}
    ]
    annotations, tel = await check_citations_liveness([_ann(url)], sources)

    assert fake.calls == [url]  # probe ran — trust was revoked
    assert annotations[0].live is False
    assert tel.dead == 1
    assert tel.live == 0


def test_fetched_gone_revokes_at_the_normalised_url_level():
    """The gone marker binds through normalisation: a marker recorded on a
    tracking-param variant revokes trust for the canonical URL too, even when
    the body lives on a different entry."""

    body = "x" * _SUBSTANTIVE_BODY_MIN_CHARS
    entries = [
        {"url": "https://example.com/page?utm_source=serp", "body": body},
        {"url": "https://example.com/page", "snippet": "", "fetched_gone": True},
    ]
    assert _build_substantive_body_url_set(entries) == set()

    # And a False marker (later successful fetch) does not revoke.
    entries_ok = [
        {"url": "https://example.com/page", "body": body, "fetched_gone": False},
    ]
    assert _build_substantive_body_url_set(entries_ok) == {
        citation_liveness._normalise_url_for_cache("https://example.com/page")
    }


def test_substantive_set_uses_shared_normalisation():
    """Body trust must bind through the SAME normalisation as the sources set
    (tracking params + fragment stripped) or a utm-tagged citation of a
    fetched page would silently lose its body trust."""

    body = "x" * _SUBSTANTIVE_BODY_MIN_CHARS
    entries = [{"url": "https://example.com/page?utm_source=news#frag", "body": body}]
    out = _build_substantive_body_url_set(entries)
    assert citation_liveness._normalise_url_for_cache("https://example.com/page") in out
