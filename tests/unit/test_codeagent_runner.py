"""Unit tests for `deepsearch.agent.codeagent_runner` — S64 live-tool surface.

Coverage:
- ``derive_authority_tier`` matches the S62 heuristic (high/medium/low).
- ``normalise_pool`` accepts both spike-shape and production-shape source dicts
  (kept for callers that still build a normalised view of upstream sources).
- ``build_codeagent_tools`` (no args) produces 5 tools — web_search,
  source_fetch, url_liveness, scratchpad_write, scratchpad_read — and routes
  each through smolagents @tool with a closure-captured call log.
- ``run_codeagent_synthesis`` requires ``LINKUP_API_KEY`` and ``DEEPSEEK_API_KEY``,
  invokes the smolagents agent without an injected pool, and surfaces telemetry.
  On agent error it returns a result with ``error`` populated rather than raising.

Smolagents itself is mocked at the import boundary; the four live tool
implementations are stubbed via ``deepsearch.agent.tools.*`` patches — these
tests do not call DeepSeek or Linkup or fetch URLs.
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
import types
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Imports under test (use real module — smolagents is installed locally).
# ---------------------------------------------------------------------------

from deepsearch.agent import codeagent_runner as runner
from deepsearch.agent.execution_control import ExecutionCancelledError


def _assert_safe_error(
    value: str | None,
    *,
    category: str,
    exception_class: str,
) -> None:
    """Pin the secret-safe error boundary instead of exception messages."""

    assert value is not None
    assert json.loads(value) == {
        "category": category,
        "exception_class": exception_class,
    }


# ---------------------------------------------------------------------------
# Authority-tier classifier (kept — carries forward to S65 retrieval).
# ---------------------------------------------------------------------------


def test_derive_authority_tier_high():
    assert runner.derive_authority_tier("https://docs.aws.amazon.com/lambda/latest/dg/foo.html") == "high"
    assert runner.derive_authority_tier("https://cloud.google.com/run/docs/configuring/cpu") == "high"
    assert runner.derive_authority_tier("https://vercel.com/docs/functions") == "high"
    assert runner.derive_authority_tier("https://docs.fly.io/some-page") == "high"
    assert runner.derive_authority_tier("https://docs.railway.app/deploy") == "high"


def test_derive_authority_tier_medium():
    assert runner.derive_authority_tier("https://aws.amazon.com/blogs/compute/some-post") == "medium"
    assert runner.derive_authority_tier("https://vercel.com/pricing") == "medium"
    assert runner.derive_authority_tier("https://github.com/aws/some-repo") == "medium"
    assert runner.derive_authority_tier("https://fly.io/changelog") == "medium"


def test_derive_authority_tier_low():
    assert runner.derive_authority_tier("https://stackoverflow.com/questions/12345") == "low"
    assert runner.derive_authority_tier("https://medium.com/@someone/post") == "low"
    assert runner.derive_authority_tier("https://reddit.com/r/aws/comments/abc") == "low"
    assert runner.derive_authority_tier("https://random-blog.example.com/post") == "low"
    assert runner.derive_authority_tier("") == "low"


# ---------------------------------------------------------------------------
# S78 m01.b (PD80) — generic first-party documentation host pattern.
# Recognizes `developers.<X>` / `developer.<X>` / `docs.<X>` / `help.<X>` /
# `support.<X>` shapes as high-tier. Closes the OODS-PLUGIN-REVIEW-01 gap
# where Figma's first-party docs (`developers.figma.com`, `help.figma.com`)
# fell through to low because they weren't on the explicit cloud-infra
# allowlist. Existing allowlist + low-host short-circuits remain.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        # The OODS-PLUGIN-REVIEW-01 motivating cases.
        "https://developers.figma.com/docs/plugins/api/",
        "https://help.figma.com/hc/en-us/articles/360039827834",
        # Other first-party docs that previously fell through to low.
        "https://docs.stripe.com/api",
        "https://developers.notion.com/reference/post-database-query",
        "https://docs.snowflake.com/sql-reference",
        "https://help.github.com/articles/about-pull-requests",
        "https://support.apple.com/guide/safari",
        # Two-label TLD (e.g. .co.uk) — generic prefix still matches because
        # the remainder still contains a dot.
        "https://docs.example.co.uk/api",
    ],
)
def test_derive_authority_tier_generic_high_pattern(url):
    assert runner.derive_authority_tier(url) == "high", (
        f"{url} must classify as high under the generic developers./docs./help. shape"
    )


def test_derive_authority_tier_generic_pattern_does_not_collide_with_explicit_low():
    """`_LOW_HOSTS` short-circuits BEFORE the generic high pattern.

    Hypothetical low-host with a docs-like subdomain must remain low.
    There is currently no such host in `_LOW_HOSTS` (no `docs.medium.com`
    exists), but the precedence order is the load-bearing invariant — a
    future addition of e.g. `docs.somecommunity.com` to `_LOW_HOSTS` must
    still classify as low.
    """

    # Surface community / aggregator hosts must NOT be promoted to high.
    assert runner.derive_authority_tier("https://reddit.com/r/programming") == "low"
    assert runner.derive_authority_tier("https://medium.com/some-blog/post") == "low"


@pytest.mark.parametrize(
    "url,expected",
    [
        # Vendor-specific path matchers in `_HIGH_HOST_PATTERNS` keep precedence.
        ("https://developer.mozilla.org/en-US/docs/Web/JavaScript", "high"),
        ("https://developers.cloudflare.com/workers/platform/limits/", "high"),
        ("https://docs.aws.amazon.com/lambda/latest/dg/foo.html", "high"),
        # `_MEDIUM_HOST_PATTERNS` keep precedence over the generic low fallback.
        ("https://github.com/owner/repo", "medium"),
        ("https://aws.amazon.com/blogs/compute/some-post", "medium"),
        # `_LOW_HOSTS` exact-match short-circuit holds.
        ("https://stackoverflow.com/questions/12345", "low"),
        ("https://reddit.com/r/aws", "low"),
        # Community/forum sub-hosts of vendors not in our allowlist remain low —
        # the generic prefix only matches `developers.` / `docs.` / `help.` etc.
        ("https://forum.figma.com/t/some-thread/1234", "low"),
        ("https://community.notion.com/t/topic", "low"),
    ],
)
def test_derive_authority_tier_regression_invariants(url, expected):
    assert runner.derive_authority_tier(url) == expected, (
        f"{url} must remain {expected!r} after S78 m01.b generalization"
    )


# ---------------------------------------------------------------------------
# S79 m01.b (PD94) — academic + standards tier patches.
#
# 25 explicit hosts appended to ``_HIGH_HOST_PATTERNS`` plus a generic ``.edu``
# TLD shape. Closes the structural retrieval bias that classified arXiv,
# tools.ietf.org, w3.org, oauth.net, etc. as "low" — both post-S78 missions
# (PLUGIN-BACKEND-AUTH-01 / STRUCTURED-OUTPUT-VIEWER-01) returned 0 academic
# / 0 standards / 0 RFC sources because the rerank tier multiplier was
# ×0.7 against authoritative literature.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        # PD102 verification expectations #2, #3, #4 — the canonical probes.
        "https://tools.ietf.org/html/rfc6749",
        "https://arxiv.org/abs/2104.04473",
        "https://cs.stanford.edu/research/papers/x.pdf",
        # Other academic publishers + preprint servers.
        "https://acm.org/conferences/dis25",
        "https://dl.acm.org/doi/10.1145/3411764.3445693",
        "https://ieee.org/standards-development/scc.html",
        "https://ieeexplore.ieee.org/document/9876543",
        "https://link.springer.com/article/10.1007/s10994-025-06234-1",
        "https://nature.com/articles/s41586-025-99999-9",
        "https://sciencedirect.com/science/article/pii/S0004370225000423",
        "https://openreview.net/forum?id=abcdefg",
        "https://plos.org/article/info:doi/10.1371/journal.pone.0290000",
        "https://biorxiv.org/content/10.1101/2024.10.20.123456v2",
        # Standards bodies (IETF / W3C / NIST / ISO).
        "https://datatracker.ietf.org/doc/html/rfc8252",
        "https://rfc-editor.org/rfc/rfc7636",
        "https://w3.org/TR/oauth-2.0-token-binding/",
        "https://oauth.net/2/grant-types/authorization-code/",
        "https://openid.net/specs/openid-connect-core-1_0.html",
        "https://nist.gov/publications/authoritative-document",
        "https://iso.org/standard/12345.html",
        # Scholar-search vendor surface (the new tool's own backends).
        "https://api.openalex.org/works/W12345",
        "https://semanticscholar.org/paper/some-id",
        # MIT.edu — PD102 verification expectation #4 generic.
        "https://web.mit.edu/research/labs/csail/",
        # Subdomain of an .edu host — generic shape still matches.
        "https://library.cs.harvard.edu/papers/foo.pdf",
    ],
)
def test_derive_authority_tier_academic_standards_high(url: str) -> None:
    assert runner.derive_authority_tier(url) == "high", (
        f"{url} must classify as high under the S79 m01.b academic+standards "
        f"tier patches (PD94)"
    )


@pytest.mark.parametrize(
    "url",
    [
        # Vanity / commercial impostor sites — exact host match prevents
        # promotion. "fakedomain-arxiv.org" is not "arxiv.org".
        "https://fakedomain-arxiv.org/listing",
        # ``.io`` is not ``.edu`` — generic TLD shape rejects.
        "https://university-vanity.io/papers",
        # ``.eduo`` is not ``.edu`` — suffix tail-match must be exact.
        "https://impostor.eduo/site",
    ],
)
def test_derive_authority_tier_academic_negative_cases(url: str) -> None:
    """Vanity + impostor URLs must NOT be promoted to high under m01.b."""

    assert runner.derive_authority_tier(url) != "high", (
        f"{url} must NOT classify as high — m01.b PD94 risk #2 mitigation"
    )


def test_default_max_steps_is_forty() -> None:
    """S79 m01.d (PD96) — DEFAULT_MAX_STEPS bumped 25 → 40.

    Type A failure mode (Mission 1 PLUGIN-BACKEND-AUTH-01 hit_max at 26/25)
    needed headroom; bump is the load-bearing config edit. m01.e clarifies
    in the smolagents bind doc that ``max_steps`` is reviewable, not part
    of the bind contract.
    """
    assert runner.DEFAULT_MAX_STEPS == 40


# ---------------------------------------------------------------------------
# Pool normalisation (kept — still used by codeagent_finalize_node until s64-m03).
# ---------------------------------------------------------------------------


def test_normalise_pool_accepts_spike_shape():
    spike_sources = [
        {
            "index": 1,
            "evidence_id": "EV-001",
            "url": "https://docs.aws.amazon.com/lambda/latest/dg/scaling.html",
            "title_hint": "scaling",
            "snippet": "Concurrency is the number of in-flight requests...",
            "tags": {"slots": ["slot-aws"], "domains": ["docs.aws.amazon.com"]},
        },
        {
            "index": 2,
            "evidence_id": "EV-002",
            "url": "https://stackoverflow.com/questions/12345",
            "title_hint": "lambda question",
            "snippet": "Quick question about lambda",
            "tags": {"slots": [], "domains": ["stackoverflow.com"]},
        },
    ]
    pool = runner.normalise_pool(spike_sources)
    assert len(pool) == 2
    assert pool[0].index == 1
    assert pool[0].authority_tier == "high"
    assert pool[1].authority_tier == "low"
    assert pool[0].slots == ("slot-aws",)


def test_normalise_pool_accepts_production_shape():
    """WorkingSet.sources_found typically uses ``title``/``summary`` not ``title_hint``/``snippet``."""

    prod_sources = [
        {
            "url": "https://cloud.google.com/run/docs/services",
            "title": "Cloud Run Services",
            "summary": "Cloud Run lets you run containers...",
        },
        {
            "url": "https://vercel.com/docs/functions/serverless-functions/runtimes/node-js",
            "title": "Vercel Functions Node.js Runtime",
            "summary": "Vercel Functions runtime details...",
        },
    ]
    pool = runner.normalise_pool(prod_sources)
    assert len(pool) == 2
    # 1-based index when input doesn't already carry one.
    assert [entry.index for entry in pool] == [1, 2]
    assert pool[0].title_hint == "Cloud Run Services"
    assert "Cloud Run lets you" in pool[0].snippet
    assert pool[0].authority_tier == "high"


def test_normalise_pool_drops_sourceless_entries():
    pool = runner.normalise_pool(
        [
            {"url": "https://docs.aws.amazon.com/lambda/x"},
            {"url": ""},  # dropped
            {"url": None},  # dropped
            {"title": "no url"},  # dropped
        ]
    )
    assert len(pool) == 1
    assert pool[0].url == "https://docs.aws.amazon.com/lambda/x"


# ---------------------------------------------------------------------------
# Tool factory (S64 live-tool surface — no admitted_pool argument).
# ---------------------------------------------------------------------------


def _install_fake_smolagents(monkeypatch: pytest.MonkeyPatch):
    fake = types.ModuleType("smolagents")
    fake.CodeAgent = _MockCodeAgent
    fake.OpenAIServerModel = MagicMock()
    fake.tool = lambda f: f  # passthrough — keeps callables directly callable in tests
    monkeypatch.setitem(sys.modules, "smolagents", fake)
    importlib.reload(runner)
    return fake


def test_build_codeagent_tools_returns_seven_live_tools(monkeypatch: pytest.MonkeyPatch):
    """S79 m01.a + m02.a appended ``scholar_search`` and ``completeness_check`` —
    total 7 tools on the agent's bag.
    """

    _install_fake_smolagents(monkeypatch)

    bundle = runner.build_codeagent_tools()

    assert len(bundle.tools) == 7
    names = [t.__name__ for t in bundle.tools]
    assert names == [
        "web_search",
        "source_fetch",
        "url_liveness",
        "scratchpad_write",
        "scratchpad_read",
        "scholar_search",
        "completeness_check",
    ]
    # No upstream pool — pool_size is 0 and tier summary reports zeros.
    assert bundle.pool_size == 0
    assert bundle.authority_tier_summary() == {"high": 0, "medium": 0, "low": 0}


def test_build_codeagent_tools_scratchpad_roundtrip(monkeypatch: pytest.MonkeyPatch):
    _install_fake_smolagents(monkeypatch)

    bundle = runner.build_codeagent_tools()
    _, _, _, scratchpad_write, scratchpad_read, _, _ = bundle.tools

    confirmation = scratchpad_write("notes_aws", "lambda timeout = 15 minutes")
    assert "scratchpad['notes_aws']" in confirmation
    assert scratchpad_read("notes_aws") == "lambda timeout = 15 minutes"
    assert scratchpad_read("nonexistent") == ""


def test_build_codeagent_tools_web_search_routes_to_impl(monkeypatch: pytest.MonkeyPatch):
    _install_fake_smolagents(monkeypatch)

    captured = {}

    def fake_web_search(query, max_results=20):
        captured["query"] = query
        captured["max_results"] = max_results
        return [{"title": "T", "url": "https://x", "snippet": "S"}]

    # Patch the impl referenced inside build_codeagent_tools' lazy imports.
    import deepsearch.agent.tools.web_search as ws_mod

    monkeypatch.setattr(ws_mod, "web_search", fake_web_search)

    bundle = runner.build_codeagent_tools()
    web_search_tool = bundle.tools[0]
    out = web_search_tool("aws lambda", 5)

    assert captured == {"query": "aws lambda", "max_results": 5}
    assert out == [{"title": "T", "url": "https://x", "snippet": "S"}]


def test_build_codeagent_tools_source_fetch_routes_to_impl(monkeypatch: pytest.MonkeyPatch):
    _install_fake_smolagents(monkeypatch)

    captured = {}

    def fake_source_fetch(url, timeout_s=15.0):
        captured["url"] = url
        captured["timeout_s"] = timeout_s
        return {"text": "body", "status": "ok", "status_code": 200, "fetched_at": "now", "final_url": url}

    import deepsearch.agent.tools.source_fetch as sf_mod

    monkeypatch.setattr(sf_mod, "source_fetch", fake_source_fetch)

    bundle = runner.build_codeagent_tools()
    source_fetch_tool = bundle.tools[1]
    out = source_fetch_tool("https://docs.aws.amazon.com/x", 10.0)

    assert captured == {"url": "https://docs.aws.amazon.com/x", "timeout_s": 10.0}
    assert out["status"] == "ok"


def test_build_codeagent_tools_url_liveness_routes_to_impl(monkeypatch: pytest.MonkeyPatch):
    _install_fake_smolagents(monkeypatch)

    captured = {}

    def fake_url_liveness(url, timeout_s=5.0):
        captured["url"] = url
        return {"alive": True, "status_code": 200, "final_url": url}

    import deepsearch.agent.tools.url_liveness as ul_mod

    monkeypatch.setattr(ul_mod, "url_liveness", fake_url_liveness)

    bundle = runner.build_codeagent_tools()
    url_liveness_tool = bundle.tools[2]
    out = url_liveness_tool("https://docs.aws.amazon.com/x")

    assert captured == {"url": "https://docs.aws.amazon.com/x"}
    assert out["alive"] is True


def test_actual_tool_wrappers_preserve_late_long_ledger_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production wrappers, not a hand-built call log, feed the ledger.

    A relevant call after generic diagnostic row 250 must retain its full
    canonical URL and exact strict envelope through both fetch and liveness.
    """

    _install_fake_smolagents(monkeypatch)

    import deepsearch.agent.tools.source_fetch as sf_mod
    import deepsearch.agent.tools.url_liveness as ul_mod
    import deepsearch.agent.tools.web_search as ws_mod

    raw_url = (
        "HTTPS://LATE.Example.TEST:443/source-"
        + ("a" * 240)
        + "?view=complete&revision=1"
    )
    canonical_url = (
        "https://late.example.test/source-"
        + ("a" * 240)
        + "?view=complete&revision=1"
    )
    monkeypatch.setattr(ws_mod, "web_search", lambda query, max_results=20: [])
    monkeypatch.setattr(
        sf_mod,
        "source_fetch",
        lambda url, timeout_s=15.0: {
            "text": "body",
            "status": "ok",
            "status_code": 200,
            "fetched_at": "now",
            "final_url": url,
        },
    )
    monkeypatch.setattr(
        ul_mod,
        "url_liveness",
        lambda url, timeout_s=5.0: {
            "alive": True,
            "status_code": 200,
            "final_url": url,
            "error": None,
        },
    )

    bundle = runner.build_codeagent_tools()
    web_search, source_fetch, url_liveness, *_ = bundle.tools
    for index in range(251):
        web_search(f"diagnostic-{index}", 1)
    source_fetch(raw_url)
    url_liveness(raw_url)

    outcomes = bundle.tool_outcomes()

    assert len(canonical_url) > 200
    assert len(outcomes["records"]) == 250
    assert outcomes["records_truncated"] == 3
    assert outcomes["ledger_records_truncated"] == 0
    assert outcomes["ledger_records"] == [
        {
            "tool": "source_fetch",
            "url": canonical_url,
            "status": "ok",
            "status_code": 200,
            "error_category": None,
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


def test_model_supplied_tool_timeouts_are_clamped_to_runtime_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_smolagents(monkeypatch)
    captured: dict[str, float] = {}

    import deepsearch.agent.tools.source_fetch as sf_mod
    import deepsearch.agent.tools.url_liveness as ul_mod
    from deepsearch.tools.openalex_search import OpenAlexClient
    from deepsearch.tools.semantic_scholar_search import SemanticScholarClient

    def fake_source_fetch(url, timeout_s=15.0):
        captured["source_fetch"] = timeout_s
        return {
            "text": "body",
            "status": "ok",
            "status_code": 200,
            "fetched_at": "now",
            "final_url": url,
        }

    def fake_url_liveness(url, timeout_s=5.0):
        captured["url_liveness"] = timeout_s
        return {"alive": True, "status_code": 200, "final_url": url}

    monkeypatch.setattr(sf_mod, "source_fetch", fake_source_fetch)
    monkeypatch.setattr(ul_mod, "url_liveness", fake_url_liveness)
    bundle = runner.build_codeagent_tools()

    bundle.tools[1]("https://example.test/source", 999.0)
    bundle.tools[2]("https://example.test/live", float("nan"))

    assert captured == {
        "source_fetch": runner.SOURCE_FETCH_MAX_TIMEOUT_S,
        "url_liveness": runner.URL_LIVENESS_MAX_TIMEOUT_S,
    }
    assert runner.SOURCE_FETCH_MAX_TIMEOUT_S == sf_mod.DEFAULT_TIMEOUT_S
    assert runner.SOURCE_FETCH_MAX_REDIRECTS == sf_mod.MAX_REDIRECTS
    assert runner.URL_LIVENESS_MAX_TIMEOUT_S == ul_mod.DEFAULT_TIMEOUT_S
    assert runner.SCHOLAR_BACKEND_TIMEOUT_S == OpenAlexClient().timeout
    assert runner.SCHOLAR_BACKEND_TIMEOUT_S == SemanticScholarClient().timeout


def test_tool_call_summary_increments(monkeypatch: pytest.MonkeyPatch):
    _install_fake_smolagents(monkeypatch)

    import deepsearch.agent.tools.web_search as ws_mod
    import deepsearch.agent.tools.source_fetch as sf_mod
    import deepsearch.agent.tools.url_liveness as ul_mod

    monkeypatch.setattr(ws_mod, "web_search", lambda q, m=20: [])
    monkeypatch.setattr(
        sf_mod,
        "source_fetch",
        lambda u, t=15.0: {"text": "", "status": "ok", "status_code": 200, "fetched_at": "", "final_url": u},
    )
    monkeypatch.setattr(
        ul_mod,
        "url_liveness",
        lambda u, t=5.0: {"alive": True, "status_code": 200, "final_url": u},
    )

    bundle = runner.build_codeagent_tools()
    (
        web_search,
        source_fetch,
        url_liveness,
        scratchpad_write,
        scratchpad_read,
        _scholar_search,
        _completeness_check,
    ) = bundle.tools

    web_search("q1", 5)
    web_search("q2", 5)
    source_fetch("https://x", 5.0)
    url_liveness("https://x")
    scratchpad_write("k", "v")
    scratchpad_read("k")

    summary = bundle.tool_call_summary()
    assert summary["total_calls"] == 6
    assert summary["by_tool"]["web_search"] == 2
    assert summary["by_tool"]["source_fetch"] == 1
    assert summary["by_tool"]["url_liveness"] == 1
    assert summary["by_tool"]["scratchpad_write"] == 1
    assert summary["by_tool"]["scratchpad_read"] == 1


def test_build_codeagent_tools_scratchpad_invalid_inputs(monkeypatch: pytest.MonkeyPatch):
    _install_fake_smolagents(monkeypatch)

    bundle = runner.build_codeagent_tools()
    _, _, _, scratchpad_write, scratchpad_read, _, _ = bundle.tools

    with pytest.raises(ValueError):
        scratchpad_write("", "value")
    with pytest.raises(ValueError):
        scratchpad_read(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Max-step fallback — smolagents returns its fallback model text verbatim.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wrapped", "expected"),
    (
        (
            'final_answer("# Report\\n\\nClaim [cite:https://example.test/a]")',
            "# Report\n\nClaim [cite:https://example.test/a]",
        ),
        (
            "<code>\nfinal_answer('## Findings\\n\\nSupported [cite:https://example.test/b]')\n</code>",
            "## Findings\n\nSupported [cite:https://example.test/b]",
        ),
        (
            '```python\nfinal_answer("""# Result\n\nEvidence [cite:https://example.test/c]""")\n```',
            "# Result\n\nEvidence [cite:https://example.test/c]",
        ),
    ),
    ids=("bare_call", "smolagents_code_tag", "python_fence"),
)
def test_unwrap_max_steps_final_answer_accepts_only_literal_report(
    wrapped: str,
    expected: str,
) -> None:
    """Known framework wrappers must not become user-visible report text."""

    assert runner._unwrap_max_steps_final_answer(wrapped) == (expected, True)


@pytest.mark.parametrize(
    "ambiguous",
    (
        "# Raw report\n\nfinal_answer is discussed as prose.",
        'final_answer(build_report())',
        'prefix = "unsafe"\nfinal_answer("# Report")',
        'The answer is final_answer("# Report")',
        '<code>final_answer("# Report")\nprint("extra statement")</code>',
        '```python\n# extra comment\nfinal_answer("# Report")\n```',
        'final_answer("# Report", metadata=True)',
    ),
)
def test_unwrap_max_steps_final_answer_preserves_ambiguous_output(
    ambiguous: str,
) -> None:
    """Normalization must never execute or reinterpret non-literal model code."""

    assert runner._unwrap_max_steps_final_answer(ambiguous) == (ambiguous, False)


# ---------------------------------------------------------------------------
# Runner — mocks smolagents.CodeAgent to avoid real LLM calls.
# ---------------------------------------------------------------------------


@dataclass
class _MockTokenUsage:
    input_tokens: int = 100
    output_tokens: int = 200
    total_tokens: int = 300


@dataclass
class _MockResult:
    output: str
    token_usage: _MockTokenUsage


class ActionStep:
    """Named to match the smolagents class — runner counts steps via type().__name__."""

    def __init__(self, *, is_final: bool = True):
        self.is_final_answer = is_final
        self.token_usage = _MockTokenUsage()
        self.timing = None
        self.tool_calls = None
        self.model_output = "model output text"


class _MockMemory:
    def __init__(self, steps):
        self.steps = steps


class _MockCodeAgent:
    def __init__(
        self,
        *,
        tools,
        model,
        max_steps,
        instructions,
        verbosity_level,
        step_callbacks=None,
    ):  # noqa: ARG002
        self.tools = tools
        self.model = model
        self.max_steps = max_steps
        self.instructions = instructions
        self.step_callbacks = list(step_callbacks or [])
        self.memory = _MockMemory([ActionStep()])

    def run(self, user_directive, return_full_result=True):  # noqa: ARG002
        return _MockResult(
            output="# Synthetic CodeAgent Output\n\nReport body with [1] citation.",
            token_usage=_MockTokenUsage(input_tokens=100, output_tokens=200, total_tokens=300),
        )


def _install_usage_agent(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result_usage: object,
    step_usage: object,
) -> None:
    """Install a successful agent with independently shaped usage envelopes."""

    class _UsageCodeAgent(_MockCodeAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            step = ActionStep()
            step.token_usage = step_usage
            self.memory = _MockMemory([step])

        def run(self, user_directive, return_full_result=True):  # noqa: ARG002
            return _MockResult(
                output="# Usage accounting\n\nCited claim [cite:https://example.test/a]",
                token_usage=result_usage,
            )

    fake = types.ModuleType("smolagents")
    fake.CodeAgent = _UsageCodeAgent
    fake.OpenAIServerModel = MagicMock()
    fake.tool = lambda f: f
    monkeypatch.setitem(sys.modules, "smolagents", fake)
    importlib.reload(runner)


def test_run_codeagent_synthesis_success(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    _install_fake_smolagents(monkeypatch)

    fake_model = MagicMock(name="fake-model")
    result = runner.run_codeagent_synthesis(
        system_instructions="system",
        user_directive="user",
        model=fake_model,
        max_steps=5,
    )
    assert "Synthetic CodeAgent Output" in result.output_markdown
    assert result.error is None
    assert result.pool_size == 0  # No upstream pool with live tools
    assert result.token_usage["total"] == 300
    assert result.token_usage_complete is True
    assert result.step_count == 1
    assert result.self_terminated is True


def test_partial_run_usage_is_replaced_by_complete_action_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlapping RunResult and ActionStep views must never be added together."""

    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    _install_usage_agent(
        monkeypatch,
        result_usage={"input_tokens": 7},
        step_usage={"input_tokens": 40, "output_tokens": 60, "total_tokens": 100},
    )

    result = runner.run_codeagent_synthesis(
        system_instructions="system",
        user_directive="user",
        model=MagicMock(),
        max_steps=5,
    )

    assert result.token_usage == {"input": 40, "output": 60, "total": 100}
    assert result.token_usage_complete is True


def test_mapping_run_result_usage_is_counted_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted mapping envelopes must not be read through attribute-only access."""

    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    _install_usage_agent(
        monkeypatch,
        result_usage={"input_tokens": 13, "output_tokens": 5, "total_tokens": 18},
        step_usage=_MockTokenUsage(),
    )

    result = runner.run_codeagent_synthesis(
        system_instructions="system",
        user_directive="user",
        model=MagicMock(),
        max_steps=5,
    )

    assert result.token_usage == {"input": 13, "output": 5, "total": 18}
    assert result.token_usage_complete is True


def test_run_codeagent_synthesis_suppresses_smolagents_sensitive_console(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Production agent logging must not expose tool arguments or fetched bodies.

    smolagents 1.24 logs generated code and Python execution output at INFO.
    This exercises its real ``CodeAgent`` and the production ``source_fetch``
    wrapper, replacing only the network implementation and model transport.
    Python root-logger DEBUG capture proves worker logging cannot re-enable the
    independent Rich console after the agent is constructed at OFF (-1).
    """

    from smolagents import ChatMessage, Model
    from smolagents.models import MessageRole
    from smolagents.monitoring import LogLevel, TokenUsage

    import deepsearch.agent.tools.source_fetch as source_fetch_mod

    signed_url = (
        "https://signed.example/private?signature="
        "LEAK_SENTINEL_URL_92"
    )
    body_marker = "LEAK_SENTINEL_BODY_92"
    fetched_body = body_marker + ("x" * 12_000)

    def fake_source_fetch(url: str, timeout_s: float = 15.0) -> dict[str, object]:
        del timeout_s
        return {
            "text": fetched_body,
            "status": "ok",
            "status_code": 200,
            "fetched_at": "2026-08-21T00:00:00+00:00",
            "final_url": url,
        }

    class _SensitiveToolModel(Model):
        def __init__(self) -> None:
            super().__init__(model_id="sensitive-log-regression")
            self.calls = 0

        def generate(
            self,
            messages: list[ChatMessage],
            **kwargs: object,
        ) -> ChatMessage:
            del messages, kwargs
            self.calls += 1
            if self.calls == 1:
                content = (
                    f'<code>page = source_fetch("{signed_url}")\n'
                    'print(page["text"])</code>'
                )
            else:
                content = (
                    '<code>final_answer("# Safe report\\n\\n'
                    'Verified [cite:https://example.com/source]")</code>'
                )
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                raw=types.SimpleNamespace(
                    choices=[types.SimpleNamespace(finish_reason="stop")]
                ),
                token_usage=TokenUsage(input_tokens=1, output_tokens=1),
            )

    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    monkeypatch.setattr(source_fetch_mod, "source_fetch", fake_source_fetch)
    model = _SensitiveToolModel()

    with caplog.at_level("DEBUG"):
        result = runner.run_codeagent_synthesis(
            system_instructions="Use the retrieval tool, then answer.",
            user_directive="Fetch the supplied private source and summarize it.",
            model=model,
            max_steps=3,
        )

    captured = capsys.readouterr()
    emitted = "\n".join((captured.out, captured.err, caplog.text))
    assert runner.SMOLAGENTS_VERBOSITY_OFF == int(LogLevel.OFF) == -1
    assert signed_url not in emitted
    assert body_marker not in emitted
    assert result.error is None
    assert result.output_markdown.startswith("# Safe report")
    assert result.tool_call_summary["by_tool"]["source_fetch"] == 1
    assert model.calls == 2


def test_run_codeagent_synthesis_repairs_real_max_steps_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real 1.24 fallback must retain the contract and return raw Markdown.

    ``provide_final_answer`` replaces the normal CodeAgent system prompt and
    returns its model response verbatim. Exercise that installed-framework seam
    offline so either a future prompt reset or wrapper-regression fails here.
    """

    from smolagents import ChatMessage, Model
    from smolagents.models import MessageRole
    from smolagents.monitoring import TokenUsage

    report = (
        "# Fallback report\n\n"
        "Supported claim [cite:https://example.test/source]\n\n"
        "## References\n\n"
        "[1] Test source — https://example.test/source"
    )

    class _MaxStepsModel(Model):
        def __init__(self) -> None:
            super().__init__(model_id="max-steps-final-answer-regression")
            self.calls = 0
            self.message_batches: list[list[ChatMessage]] = []

        def generate(
            self,
            messages: list[ChatMessage],
            **kwargs: object,
        ) -> ChatMessage:
            del kwargs
            self.calls += 1
            self.message_batches.append(list(messages))
            content = (
                '<code>draft = "unfinished"\nprint(draft)</code>'
                if self.calls == 1
                else f"<code>final_answer({report!r})</code>"
            )
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                raw=types.SimpleNamespace(
                    choices=[types.SimpleNamespace(finish_reason="stop")]
                ),
                token_usage=TokenUsage(input_tokens=1, output_tokens=1),
            )

    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    model = _MaxStepsModel()
    contract_sentinel = "MISSION CONTRACT: every supported claim requires [cite:URL]."

    result = runner.run_codeagent_synthesis(
        system_instructions=contract_sentinel,
        user_directive="Produce the requested report.",
        model=model,
        max_steps=1,
    )

    fallback_messages = model.message_batches[-1]
    fallback_system_text = str(fallback_messages[0].content)
    fallback_post_text = str(fallback_messages[-1].content)
    assert model.calls == 2, "one action plus the framework's max-step fallback"
    assert contract_sentinel in fallback_system_text
    assert "Return only the raw Markdown report" in fallback_system_text
    assert "not code or a final_answer(...) wrapper" in fallback_post_text
    assert result.output_markdown == report
    assert result.hit_max_steps is True
    assert result.self_terminated is False
    assert result.output_wrapper_removed is True
    assert result.max_steps_fallback_complete is True
    assert result.output_truncated is False
    assert result.error is None


def test_paid_generation_error_finalizes_once_on_same_agent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A billed generation failure must finalize retained memory, not restart.

    This exercises the real smolagents 1.24 failure seam offline: the second
    action generation raises, smolagents retains the first paid ActionStep and
    re-raises AgentGenerationError, then the runner makes one final-answer call
    on that same memory.
    """

    from smolagents import ChatMessage, Model
    from smolagents.models import MessageRole
    from smolagents.monitoring import TokenUsage

    report = (
        "# Recovered report\n\n"
        "Supported finding [cite:https://example.test/source]\n\n"
        "## References\n\n"
        "[1] Source — https://example.test/source"
    )
    secret = "https://signed.example/private?token=DO_NOT_PERSIST"

    class _MidRunFailureModel(Model):
        def __init__(self) -> None:
            super().__init__(model_id="mid-run-recovery-regression")
            self.calls = 0
            self.message_batches: list[list[ChatMessage]] = []

        def generate(
            self,
            messages: list[ChatMessage],
            **kwargs: object,
        ) -> ChatMessage:
            del kwargs
            self.calls += 1
            self.message_batches.append(list(messages))
            if self.calls == 1:
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content='<code>print("paid-memory-sentinel")</code>',
                    raw=types.SimpleNamespace(
                        choices=[types.SimpleNamespace(finish_reason="stop")]
                    ),
                    token_usage=TokenUsage(input_tokens=11, output_tokens=5),
                )
            if self.calls == 2:
                raise TimeoutError(f"request failed for {secret}")
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=f"<code>final_answer({report!r})</code>",
                raw=types.SimpleNamespace(
                    choices=[types.SimpleNamespace(finish_reason="stop")]
                ),
                token_usage=TokenUsage(input_tokens=7, output_tokens=3),
            )

    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    model = _MidRunFailureModel()

    result = runner.run_codeagent_synthesis(
        system_instructions="Every supported claim requires [cite:URL].",
        user_directive="Produce the requested report.",
        model=model,
        max_steps=5,
    )

    recovery_messages = model.message_batches[-1]
    assert model.calls == 3, "two action calls plus exactly one memory finalization"
    assert "paid-memory-sentinel" in str(recovery_messages)
    assert result.output_markdown == report
    assert result.error is None
    assert result.generation_failure_error is not None
    assert result.generation_failure_reason == "timeout"
    assert result.generation_failure_reason_source == "class"
    assert result.generation_failure_root_class == "TimeoutError"
    assert result.generation_failure_http_status is None
    assert result.generation_failure_transient is True
    assert result.generation_failure_step == 2
    assert result.generation_failure_pass == 0
    assert result.generation_failure_pass_count == 1
    assert result.generation_failure_consumed_tokens == 16
    assert result.generation_failure_retry_eligible is False
    assert result.generation_failure_retry_suppressed_reason == "paid_progress"
    assert result.generation_recovery_attempted is True
    assert result.generation_recovery_result == "success"
    assert result.generation_recovery_finish_reason == "stop"
    assert result.generation_error_recovered is True
    assert result.output_wrapper_removed is True
    assert result.hit_max_steps is False
    assert result.self_terminated is False
    assert result.max_steps_fallback_complete is False
    assert result.token_usage == {"input": 18, "output": 8, "total": 26}
    assert result.token_usage_complete is False
    assert secret not in repr(result)
    assert secret not in caplog.text


def test_run_codeagent_synthesis_missing_linkup_key_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LINKUP_API_KEY", raising=False)
    _install_fake_smolagents(monkeypatch)

    with pytest.raises(runner.CodeAgentRunnerError, match="LINKUP_API_KEY"):
        runner.run_codeagent_synthesis(
            system_instructions="system",
            user_directive="user",
            model=MagicMock(),
        )


def test_run_codeagent_synthesis_agent_error_returns_result(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")

    class _FailingAgent(_MockCodeAgent):
        def run(self, user_directive, return_full_result=True):  # noqa: ARG002
            raise RuntimeError("model timeout")

    fake = types.ModuleType("smolagents")
    fake.CodeAgent = _FailingAgent
    fake.OpenAIServerModel = MagicMock()
    fake.tool = lambda f: f
    monkeypatch.setitem(sys.modules, "smolagents", fake)
    importlib.reload(runner)

    result = runner.run_codeagent_synthesis(
        system_instructions="system",
        user_directive="user",
        model=MagicMock(),
    )
    _assert_safe_error(
        result.error,
        category="execution",
        exception_class="RuntimeError",
    )
    assert result.output_markdown == ""


def test_run_codeagent_synthesis_error_drops_hostile_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Provider URLs and response bodies cannot enter runner logs or telemetry."""

    signed_url = "https://user:pass@example.test/private?signature=runner-secret"
    body_marker = "HOSTILE_RUNNER_RESPONSE_BODY"
    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")

    class _FailingAgent(_MockCodeAgent):
        def run(self, user_directive, return_full_result=True):  # noqa: ARG002
            raise RuntimeError(f"{signed_url} {body_marker}")

    fake = types.ModuleType("smolagents")
    fake.CodeAgent = _FailingAgent
    fake.OpenAIServerModel = MagicMock()
    fake.tool = lambda f: f
    monkeypatch.setitem(sys.modules, "smolagents", fake)
    importlib.reload(runner)

    with caplog.at_level("WARNING"):
        result = runner.run_codeagent_synthesis(
            system_instructions="system",
            user_directive="user",
            model=MagicMock(),
        )

    _assert_safe_error(
        result.error,
        category="execution",
        exception_class="RuntimeError",
    )
    emitted = f"{result.error}\n{caplog.text}"
    assert signed_url not in emitted
    assert body_marker not in emitted


def test_run_codeagent_synthesis_callback_stops_later_paid_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lease cancellation after one action must escape instead of becoming a soft error."""

    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    cancel_event = threading.Event()

    class _CancellingAgent(_MockCodeAgent):
        def run(self, user_directive, return_full_result=True):  # noqa: ARG002
            cancel_event.set()
            self.step_callbacks[0](ActionStep())
            raise AssertionError("a later paid step must be unreachable")

    fake = types.ModuleType("smolagents")
    fake.CodeAgent = _CancellingAgent
    fake.OpenAIServerModel = MagicMock()
    fake.tool = lambda f: f
    monkeypatch.setitem(sys.modules, "smolagents", fake)
    importlib.reload(runner)

    with pytest.raises(ExecutionCancelledError, match="lease loss"):
        runner.run_codeagent_synthesis(
            system_instructions="system",
            user_directive="user",
            model=MagicMock(),
            cancel_event=cancel_event,
        )


def test_run_codeagent_synthesis_validates_strings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    _install_fake_smolagents(monkeypatch)

    with pytest.raises(runner.CodeAgentRunnerError, match="system_instructions"):
        runner.run_codeagent_synthesis(
            system_instructions="",
            user_directive="user",
            model=MagicMock(),
        )
    with pytest.raises(runner.CodeAgentRunnerError, match="user_directive"):
        runner.run_codeagent_synthesis(
            system_instructions="system",
            user_directive="   ",
            model=MagicMock(),
        )


def test_build_default_model_requires_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("deepsearch.config.load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_BACKEND", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    fake_smolagents = types.ModuleType("smolagents")
    fake_smolagents.OpenAIServerModel = MagicMock()
    monkeypatch.setitem(sys.modules, "smolagents", fake_smolagents)
    importlib.reload(runner)
    with pytest.raises(runner.CodeAgentRunnerError, match="DEEPSEEK_API_KEY"):
        runner.build_default_model()


# ---------------------------------------------------------------------------
# Request timeout on the shared DeepSeek bind.
#
# build_default_model is the single chokepoint for BOTH synthesis
# (run_codeagent_synthesis) and critique, so a timeout set here bounds every
# model call on the mission path. smolagents forwards ``client_kwargs``
# verbatim into ``openai.OpenAI(**client_kwargs)``, which is why the timeout
# goes there rather than into the completion kwargs.
#
# WHY it matters: openai's default is a 600s timeout with automatic retries.
# On 2026-08-13 an unresponsive endpoint held a mission lease for 18 minutes
# and the mission landed terminally in 'blocked'. A bounded timeout turns that
# into a fast failure that the S89 retry loop can actually absorb.
# ---------------------------------------------------------------------------


def test_build_default_model_sets_a_bounded_client_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_smolagents = _install_fake_smolagents(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    runner.build_default_model()

    client_kwargs = fake_smolagents.OpenAIServerModel.call_args.kwargs["client_kwargs"]
    assert client_kwargs["timeout"] == runner.DEFAULT_REQUEST_TIMEOUT_S
    assert client_kwargs["max_retries"] == runner.OPENAI_CLIENT_MAX_RETRIES == 0
    assert (
        fake_smolagents.OpenAIServerModel.call_args.kwargs["retry"]
        is runner.SMOLAGENTS_RATE_LIMIT_RETRY
        is False
    )
    assert client_kwargs["timeout"] > 0


def test_build_default_model_request_timeout_is_overridable(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_smolagents = _install_fake_smolagents(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    runner.build_default_model(request_timeout_s=17.5)

    client_kwargs = fake_smolagents.OpenAIServerModel.call_args.kwargs["client_kwargs"]
    assert client_kwargs["timeout"] == 17.5


def test_build_default_model_keeps_timeout_through_the_max_tokens_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    """The PD124 max_tokens fallback must not drop the timeout on the retry.

    Without this, a construction that falls back to _FALLBACK_MAX_TOKENS would
    silently rebuild an unbounded client.
    """
    fake_smolagents = _install_fake_smolagents(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    calls: list[dict] = []

    def _flaky(**kwargs):
        calls.append(kwargs)
        if kwargs["max_tokens"] > runner._FALLBACK_MAX_TOKENS:
            raise ValueError("max_tokens too large")
        return MagicMock()

    fake_smolagents.OpenAIServerModel = MagicMock(side_effect=_flaky)

    runner.build_default_model(request_timeout_s=9.0)

    assert len(calls) == 2, "expected one rejected construction then the fallback"
    assert calls[-1]["max_tokens"] == runner._FALLBACK_MAX_TOKENS
    assert calls[-1]["client_kwargs"]["timeout"] == 9.0
    assert calls[-1]["client_kwargs"]["max_retries"] == 0
    assert calls[-1]["retry"] is False


# ---------------------------------------------------------------------------
# S66/S87 sources_collected grounding union — populated by discovery and
# successful readable fetches, then surfaced on SynthesisResult. Failed fetches,
# probes, and scratchpad state are excluded.
# ---------------------------------------------------------------------------


def test_sources_collected_populated_from_web_search(monkeypatch: pytest.MonkeyPatch):
    _install_fake_smolagents(monkeypatch)

    import deepsearch.agent.tools.web_search as ws_mod

    def fake_web_search(query, max_results=20):
        return [
            {
                "title": "AWS Lambda doc",
                "url": "https://docs.aws.amazon.com/lambda/page-1",
                "snippet": "Lambda concurrency snippet 1.",
                "score": 0.92,
                "authority_tier": "high",
            },
            {
                "title": "AWS Lambda blog",
                "url": "https://aws.amazon.com/blogs/lambda-2",
                "snippet": "Lambda blog snippet 2.",
                "score": 0.71,
                "authority_tier": "medium",
            },
            {
                "title": "Stack overflow",
                "url": "https://stackoverflow.com/q/lambda-3",
                "snippet": "SO snippet 3.",
                "score": 0.40,
                "authority_tier": "low",
            },
        ]

    monkeypatch.setattr(ws_mod, "web_search", fake_web_search)

    bundle = runner.build_codeagent_tools()
    web_search_tool = bundle.tools[0]
    web_search_tool("aws lambda concurrency", 10)

    assert len(bundle.sources_collected) == 3
    by_url = {e["url"]: e for e in bundle.sources_collected}
    assert "https://docs.aws.amazon.com/lambda/page-1" in by_url
    page_1 = by_url["https://docs.aws.amazon.com/lambda/page-1"]
    assert page_1["title"] == "AWS Lambda doc"
    assert page_1["snippet"] == "Lambda concurrency snippet 1."
    assert page_1["score"] == 0.92
    assert page_1["authority_tier"] == "high"


def test_search_tool_payload_has_deterministic_per_result_and_total_budgets(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_smolagents(monkeypatch)
    import deepsearch.agent.tools.web_search as ws_mod

    monkeypatch.setattr(
        ws_mod,
        "web_search",
        lambda query, max_results=20: [
            {
                "title": f"Result {index}",
                "url": f"https://example.com/{index}",
                "snippet": "x" * 5_000,
                "score": 0.5,
                "authority_tier": "low",
            }
            for index in range(30)
        ],
    )
    bundle = runner.build_codeagent_tools()

    public = bundle.tools[0]("bounded payload", 30)

    assert all(
        len(item["snippet"]) <= runner.SEARCH_RESULT_SNIPPET_MAX_CHARS
        for item in public
    )
    assert len(json.dumps(public, ensure_ascii=False, sort_keys=True)) <= (
        runner.SEARCH_TOOL_PAYLOAD_MAX_CHARS
    )
    assert bundle.call_log[-1]["payload_fields_truncated"] > 0
    assert bundle.call_log[-1]["payload_records_dropped"] > 0


def test_sources_collected_populated_from_source_fetch(monkeypatch: pytest.MonkeyPatch):
    _install_fake_smolagents(monkeypatch)

    import deepsearch.agent.tools.source_fetch as sf_mod

    def fake_source_fetch(url, timeout_s=15.0):
        return {
            "text": "Lambda body text " * 50,
            "status": "ok",
            "status_code": 200,
            "fetched_at": "now",
            "final_url": "https://docs.aws.amazon.com/lambda/dg/scaling.html",
        }

    monkeypatch.setattr(sf_mod, "source_fetch", fake_source_fetch)

    bundle = runner.build_codeagent_tools()
    source_fetch_tool = bundle.tools[1]
    source_fetch_tool("https://docs.aws.amazon.com/lambda/dg/scaling.html", 10.0)

    assert len(bundle.sources_collected) == 1
    entry = bundle.sources_collected[0]
    assert entry["url"] == "https://docs.aws.amazon.com/lambda/dg/scaling.html"
    # The readable body is retained; the display snippet is independently capped.
    assert entry["snippet"].startswith("Lambda body text ")
    assert len(entry["snippet"]) <= 300
    assert entry["body"] == "Lambda body text " * 50
    # Authority tier is derived from the final_url.
    assert entry["authority_tier"] == "high"


def test_source_fetch_keeps_full_internal_body_but_bounds_public_tool_payload(
    monkeypatch: pytest.MonkeyPatch,
):
    """Internal grounding fidelity must not enlarge the model-facing result."""

    _install_fake_smolagents(monkeypatch)
    import deepsearch.agent.tools.source_fetch as sf_mod

    full_body = "readable evidence " * 2_000

    def fake_source_fetch(url, timeout_s=15.0):
        return {
            "text": full_body,
            "status": "ok",
            "status_code": 200,
            "fetched_at": "now",
            "final_url": url,
            "content_type": "text/html",
        }

    monkeypatch.setattr(sf_mod, "source_fetch", fake_source_fetch)
    bundle = runner.build_codeagent_tools()

    public = bundle.tools[1]("https://docs.example.com/evidence")

    assert bundle.sources_collected[0]["body"] == full_body
    assert bundle.sources_fetched[0]["body"] == full_body
    assert len(public["text"]) == runner.SOURCE_FETCH_TOOL_TEXT_MAX_CHARS
    assert public["text_chars_total"] == len(full_body)
    assert public["text_chars_returned"] == runner.SOURCE_FETCH_TOOL_TEXT_MAX_CHARS
    assert public["text_truncated_for_tool"] is True
    assert len(json.dumps(public, ensure_ascii=False, sort_keys=True)) <= (
        runner.SOURCE_FETCH_TOOL_PAYLOAD_MAX_CHARS
    )


def test_readable_body_quality_ratchet_rejects_later_search_snippet(
    monkeypatch: pytest.MonkeyPatch,
):
    """Later discovery metadata may enrich a fetch but cannot replace its body."""

    _install_fake_smolagents(monkeypatch)
    import deepsearch.agent.tools.source_fetch as sf_mod
    import deepsearch.agent.tools.web_search as ws_mod

    url = "https://docs.example.com/evidence"
    full_body = "Full primary evidence body that must survive."
    monkeypatch.setattr(
        sf_mod,
        "source_fetch",
        lambda target, timeout_s=15.0: {
            "text": full_body,
            "status": "ok",
            "status_code": 200,
            "fetched_at": "now",
            "final_url": target,
        },
    )
    monkeypatch.setattr(
        ws_mod,
        "web_search",
        lambda query, max_results=20: [
            {
                "title": "Primary title",
                "url": url,
                "snippet": "Later thin snippet.",
                "score": 0.9,
                "authority_tier": "high",
            }
        ],
    )
    bundle = runner.build_codeagent_tools()

    bundle.tools[1](url)
    bundle.tools[0]("evidence")

    assert len(bundle.sources_collected) == 1
    assert bundle.sources_collected[0]["body"] == full_body
    assert bundle.sources_collected[0]["content_quality"] == "readable_body"
    assert bundle.sources_collected[0]["title"] == "Primary title"
    assert bundle.retrieval_state_summary() == {
        "discovered": 1,
        "fetched_readable": 1,
        "fetch_attempts": 1,
        "probes": 0,
        "grounding_sources": 1,
    }


def test_failed_fetch_is_diagnostic_only_then_success_adds_readable_evidence(
    monkeypatch: pytest.MonkeyPatch,
):
    """A failed fetch cannot create grounding; a readable success can."""

    _install_fake_smolagents(monkeypatch)

    import deepsearch.agent.tools.source_fetch as sf_mod

    responses = [
        {"text": "", "status": "dead", "status_code": 404, "fetched_at": "now",
         "final_url": "https://example.com/page"},
        {"text": "Recovered body text " * 30, "status": "ok", "status_code": 200,
         "fetched_at": "now", "final_url": "https://example.com/page"},
    ]
    calls = {"n": 0}

    def fake_source_fetch(url, timeout_s=15.0):
        out = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return out

    monkeypatch.setattr(sf_mod, "source_fetch", fake_source_fetch)

    bundle = runner.build_codeagent_tools()
    source_fetch_tool = bundle.tools[1]

    source_fetch_tool("https://example.com/page", 10.0)
    assert bundle.sources_collected == []
    assert bundle.sources_fetched == []
    assert bundle.fetch_attempts[0]["status"] == "dead"

    source_fetch_tool("https://example.com/page", 10.0)
    entry = bundle.sources_collected[0]
    assert entry["body"].startswith("Recovered body text ")
    assert entry["content_quality"] == "readable_body"
    assert len(bundle.sources_fetched) == 1
    assert len(bundle.fetch_attempts) == 2


def test_repeated_failed_fetches_never_create_grounding_sources(
    monkeypatch: pytest.MonkeyPatch,
):
    """Dead, auth-wall, and timeout outcomes remain attempt telemetry only."""

    _install_fake_smolagents(monkeypatch)

    import deepsearch.agent.tools.source_fetch as sf_mod

    responses = [
        {"text": "", "status": "dead", "status_code": 404, "fetched_at": "now",
         "final_url": "https://example.com/page"},
        {"text": "", "status": "blocked", "status_code": 403, "fetched_at": "now",
         "final_url": "https://example.com/page"},
        {"text": "", "status": "timeout", "status_code": 0, "fetched_at": "now",
         "final_url": "https://example.com/page"},
    ]
    calls = {"n": 0}

    def fake_source_fetch(url, timeout_s=15.0):
        out = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return out

    monkeypatch.setattr(sf_mod, "source_fetch", fake_source_fetch)

    bundle = runner.build_codeagent_tools()
    source_fetch_tool = bundle.tools[1]

    source_fetch_tool("https://example.com/page", 10.0)
    source_fetch_tool("https://example.com/page", 10.0)  # 403 answer
    source_fetch_tool("https://example.com/page", 10.0)  # timeout answer
    assert bundle.sources_collected == []
    assert bundle.sources_fetched == []
    assert [item["status"] for item in bundle.fetch_attempts] == [
        "dead",
        "blocked",
        "timeout",
    ]


def test_failed_redirected_fetch_records_attempt_without_evidence(
    monkeypatch: pytest.MonkeyPatch,
):
    """The request and final URL remain diagnostic provenance, not sources."""

    _install_fake_smolagents(monkeypatch)

    import deepsearch.agent.tools.source_fetch as sf_mod

    def fake_source_fetch(url, timeout_s=15.0):
        return {"text": "", "status": "dead", "status_code": 404,
                "fetched_at": "now", "final_url": "https://example.com/moved-target"}

    monkeypatch.setattr(sf_mod, "source_fetch", fake_source_fetch)

    bundle = runner.build_codeagent_tools()
    source_fetch_tool = bundle.tools[1]
    source_fetch_tool("https://example.com/serp-url", 10.0)

    assert bundle.sources_collected == []
    assert bundle.fetch_attempts == [
        {
            "url": "https://example.com/serp-url",
            "final_url": "https://example.com/moved-target",
            "status": "dead",
            "status_code": 404,
            "error": None,
            "readable_chars": 0,
        }
    ]


def test_url_liveness_is_probe_only(monkeypatch: pytest.MonkeyPatch):
    _install_fake_smolagents(monkeypatch)

    import deepsearch.agent.tools.url_liveness as ul_mod

    def fake_url_liveness(url, timeout_s=5.0):
        return {
            "alive": True,
            "status_code": 200,
            "final_url": "https://docs.aws.amazon.com/lambda/dg/foo.html",
        }

    monkeypatch.setattr(ul_mod, "url_liveness", fake_url_liveness)

    bundle = runner.build_codeagent_tools()
    url_liveness_tool = bundle.tools[2]
    url_liveness_tool("https://docs.aws.amazon.com/lambda/dg/foo.html")

    assert bundle.sources_collected == []
    assert bundle.probes == [
        {
            "url": "https://docs.aws.amazon.com/lambda/dg/foo.html",
            "final_url": "https://docs.aws.amazon.com/lambda/dg/foo.html",
            "alive": True,
            "status_code": 200,
            "error": None,
        }
    ]


def test_sources_collected_dedupe_on_url(monkeypatch: pytest.MonkeyPatch):
    _install_fake_smolagents(monkeypatch)

    import deepsearch.agent.tools.web_search as ws_mod
    import deepsearch.agent.tools.source_fetch as sf_mod

    target_url = "https://docs.aws.amazon.com/lambda/dg/scaling.html"

    def fake_web_search(query, max_results=20):
        return [
            {
                "title": "AWS Lambda doc",
                "url": target_url,
                "snippet": "Search-time snippet.",
                "score": 0.85,
                "authority_tier": "high",
            }
        ]

    def fake_source_fetch(url, timeout_s=15.0):
        return {
            "text": "Body excerpt from fetch.",
            "status": "ok",
            "status_code": 200,
            "fetched_at": "now",
            "final_url": target_url,
        }

    monkeypatch.setattr(ws_mod, "web_search", fake_web_search)
    monkeypatch.setattr(sf_mod, "source_fetch", fake_source_fetch)

    bundle = runner.build_codeagent_tools()
    web_search_tool = bundle.tools[0]
    source_fetch_tool = bundle.tools[1]

    web_search_tool("aws lambda", 5)
    source_fetch_tool(target_url, 10.0)

    # Single entry — last-wins merge semantics.
    assert len(bundle.sources_collected) == 1
    entry = bundle.sources_collected[0]
    assert entry["url"] == target_url
    # Title from web_search is preserved (source_fetch supplies no title).
    assert entry["title"] == "AWS Lambda doc"
    # Snippet is overwritten by the body excerpt from source_fetch.
    assert entry["snippet"] == "Body excerpt from fetch."
    # Score from web_search survives — source_fetch carries no score signal.
    assert entry["score"] == 0.85
    # Authority tier is consistent (both tools resolve "high" for AWS docs).
    assert entry["authority_tier"] == "high"


def test_fetch_does_not_overwrite_the_tier_used_for_scholar_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Telemetry must retain the tier that produced the persisted score."""

    _install_fake_smolagents(monkeypatch)

    import deepsearch.agent.tools.scholar_search as ss_mod
    import deepsearch.agent.tools.source_fetch as sf_mod

    target_url = "https://publisher.example/paper"

    monkeypatch.setattr(
        ss_mod,
        "scholar_search",
        lambda query, max_results=20: [
            {
                "title": "Primary paper",
                "url": target_url,
                "snippet": "Academic evidence.",
                "score": 0.04,
                "authority_tier": "high",
            }
        ],
    )
    monkeypatch.setattr(
        sf_mod,
        "source_fetch",
        lambda url, timeout_s=15.0: {
            "text": "Fetched paper body.",
            "status": "ok",
            "status_code": 200,
            "fetched_at": "now",
            "final_url": target_url,
        },
    )

    bundle = runner.build_codeagent_tools()
    bundle.tools[5]("paper", 5)
    bundle.tools[1](target_url, 10.0)

    assert bundle.sources_collected[0]["score"] == 0.04
    assert bundle.sources_collected[0]["authority_tier"] == "high"
    assert bundle.authority_tier_summary() == {"high": 1, "medium": 0, "low": 0}


def test_sources_collected_scratchpad_excluded(monkeypatch: pytest.MonkeyPatch):
    _install_fake_smolagents(monkeypatch)

    bundle = runner.build_codeagent_tools()
    _, _, _, scratchpad_write, scratchpad_read, _, _ = bundle.tools

    scratchpad_write("notes", "agent-internal state, not a source")
    scratchpad_read("notes")

    assert bundle.sources_collected == []


def test_sources_collected_default_empty(monkeypatch: pytest.MonkeyPatch):
    _install_fake_smolagents(monkeypatch)

    bundle = runner.build_codeagent_tools()

    assert bundle.sources_collected == []


def test_sources_collected_skips_error_sentinels(monkeypatch: pytest.MonkeyPatch):
    """Linkup error sentinels (``error`` key set) must not enter the accumulator."""

    _install_fake_smolagents(monkeypatch)

    import deepsearch.agent.tools.web_search as ws_mod

    def fake_web_search(query, max_results=20):
        return [
            {
                "error": "rate_limit",
                "message": "rate limited",
                "title": "",
                "url": "",
                "snippet": "",
            }
        ]

    monkeypatch.setattr(ws_mod, "web_search", fake_web_search)

    bundle = runner.build_codeagent_tools()
    web_search_tool = bundle.tools[0]
    out = web_search_tool("anything", 5)

    assert isinstance(out, list) and out[0].get("error") == "rate_limit"
    assert bundle.sources_collected == []


def test_malformed_scholar_sentinel_is_failure_telemetry_not_empty_success(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_smolagents(monkeypatch)
    import deepsearch.agent.tools.scholar_search as ss_mod

    monkeypatch.setattr(
        ss_mod,
        "scholar_search",
        lambda query, max_results=20: [
            {
                "error": "malformed_payload",
                "message": "expected list",
                "title": "",
                "url": "",
                "snippet": "",
            }
        ],
    )
    bundle = runner.build_codeagent_tools()

    output = bundle.tools[5]("papers", 5)
    outcomes = bundle.tool_outcomes()["by_tool"]["scholar_search"]

    assert output[0]["error"] == "malformed_payload"
    assert outcomes["error"] == 1
    assert outcomes["by_category"] == {"malformed_payload": 1}
    assert bundle.sources_collected == []


def test_run_codeagent_synthesis_surfaces_sources_collected(
    monkeypatch: pytest.MonkeyPatch,
):
    """``SynthesisResult.sources_collected`` mirrors the bundle accumulator."""

    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    _install_fake_smolagents(monkeypatch)

    import deepsearch.agent.tools.web_search as ws_mod

    def fake_web_search(query, max_results=20):
        return [
            {
                "title": "Doc",
                "url": "https://docs.aws.amazon.com/x",
                "snippet": "snip",
                "score": 0.9,
                "authority_tier": "high",
            }
        ]

    monkeypatch.setattr(ws_mod, "web_search", fake_web_search)

    # Drive web_search via the wrapped tool inside _MockCodeAgent before run().
    class _PrimingCodeAgent(_MockCodeAgent):
        def run(self, user_directive, return_full_result=True):  # noqa: ARG002
            for t in self.tools:
                if t.__name__ == "web_search":
                    t("aws lambda", 5)
                    break
            return _MockResult(
                output="# out\n",
                token_usage=_MockTokenUsage(),
            )

    fake = types.ModuleType("smolagents")
    fake.CodeAgent = _PrimingCodeAgent
    fake.OpenAIServerModel = MagicMock()
    fake.tool = lambda f: f
    monkeypatch.setitem(sys.modules, "smolagents", fake)
    importlib.reload(runner)
    monkeypatch.setattr(ws_mod, "web_search", fake_web_search)

    result = runner.run_codeagent_synthesis(
        system_instructions="system",
        user_directive="user",
        model=MagicMock(),
        max_steps=5,
    )

    assert result.sources_collected, "expected sources_collected to be populated"
    assert result.sources_collected[0]["url"] == "https://docs.aws.amazon.com/x"
    assert result.sources_collected[0]["score"] == 0.9
    assert result.source_collection_projection == {
        "input_records": 1,
        "projected_records": 1,
        "dropped_records": 0,
        "merged_records": 0,
        "reason_counts": {},
    }
    assert result.source_collection_error is None


def test_tolerant_source_projection_selects_richer_conflict_deterministically() -> None:
    """Conflict resolution must not depend on retrieval arrival order."""

    thin = {
        "url": "HTTPS://Example.COM:443/a/../canonical",
        "title": "Thin title",
        "snippet": "Search snippet",
        "content_quality": "snippet",
    }
    rich = {
        "url": "https://example.com/canonical",
        "title": "Fetched primary source",
        "snippet": "Search snippet",
        "body": "Readable primary evidence body",
        "body_source": "source_fetch",
        "content_quality": "readable_body",
    }

    forward_sources, forward_telemetry = runner._project_sources_collected(
        [thin, rich]
    )
    reverse_sources, reverse_telemetry = runner._project_sources_collected(
        [rich, thin]
    )

    assert forward_sources == reverse_sources
    assert forward_sources[0]["title"] == "Fetched primary source"
    assert forward_sources[0]["body"] == "Readable primary evidence body"
    assert forward_telemetry == reverse_telemetry == {
        "input_records": 2,
        "projected_records": 1,
        "dropped_records": 1,
        "merged_records": 0,
        "reason_counts": {"canonical_metadata_conflict": 1},
    }


@pytest.mark.parametrize(
    (
        "search_results",
        "expected_urls",
        "expected_reason",
        "expected_conflict_title",
    ),
    (
        (
            [
                {
                    "title": "Thin",
                    "url": "HTTPS://Example.COM:443/a/../canonical",
                    "snippet": "Evidence",
                },
                {
                    "title": "Richer canonical title",
                    "url": "https://example.com/canonical",
                    "snippet": "Evidence",
                },
                {
                    "title": "Valid sibling",
                    "url": "https://example.org/valid-sibling",
                    "snippet": "Sibling evidence",
                },
            ],
            [
                "https://example.com/canonical",
                "https://example.org/valid-sibling",
            ],
            "canonical_metadata_conflict",
            "Richer canonical title",
        ),
        (
            [
                {
                    "title": "Unsafe source",
                    "url": "https://user:password@example.com/private",
                    "snippet": "Evidence",
                },
                {
                    "title": "Valid sibling",
                    "url": "https://example.org/valid-sibling",
                    "snippet": "Sibling evidence",
                },
            ],
            ["https://example.org/valid-sibling"],
            "invalid_url",
            None,
        ),
    ),
    ids=("canonical_conflict", "userinfo"),
)
def test_run_codeagent_synthesis_drops_only_invalid_source_records(
    monkeypatch: pytest.MonkeyPatch,
    search_results: list[dict[str, str]],
    expected_urls: list[str],
    expected_reason: str,
    expected_conflict_title: str | None,
) -> None:
    """One unsafe source must not erase valid evidence from a paid run."""

    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    _install_fake_smolagents(monkeypatch)

    import deepsearch.agent.tools.web_search as ws_mod

    class _PrimingCodeAgent(_MockCodeAgent):
        def run(self, user_directive, return_full_result=True):  # noqa: ARG002
            next(tool for tool in self.tools if tool.__name__ == "web_search")(
                "canonical source",
                5,
            )
            return _MockResult(output="# out\n", token_usage=_MockTokenUsage())

    fake = types.ModuleType("smolagents")
    fake.CodeAgent = _PrimingCodeAgent
    fake.OpenAIServerModel = MagicMock()
    fake.tool = lambda function: function
    monkeypatch.setitem(sys.modules, "smolagents", fake)
    importlib.reload(runner)
    monkeypatch.setattr(
        ws_mod,
        "web_search",
        lambda query, max_results=20: search_results,
    )

    result = runner.run_codeagent_synthesis(
        system_instructions="system",
        user_directive="user",
        model=MagicMock(),
        max_steps=5,
    )

    assert result.error is None, "the paid synthesis itself completed"
    assert [source["url"] for source in result.sources_collected] == expected_urls
    if expected_conflict_title is not None:
        assert result.sources_collected[0]["title"] == expected_conflict_title
    if expected_reason == "invalid_url":
        assert result.scope_telemetry["rejected"] == [{
            "url": "", "surface": "web_search", "reason": "scope_source_not_allowed",
        }], "Unsafe URLs are now rejected before actor admission, with no credential leakage"
        return
    assert result.source_collection_error
    assert result.source_collection_projection == {
        "input_records": len(search_results),
        "projected_records": len(expected_urls),
        "dropped_records": 1,
        "merged_records": 0,
        "reason_counts": {expected_reason: 1},
    }
    assert result.tool_call_summary["by_tool"] == {"web_search": 1}
    assert len(result.tool_outcomes["records"]) == 1
    assert "password" not in result.source_collection_error
    assert "password" not in repr(result.source_collection_projection)


# ---------------------------------------------------------------------------
# S88 — max_tokens output-truncation detection. finish_reason == "length" on the
# self-terminated final-answer ActionStep means the shipped report was cut
# mid-content. The getattr chain fails safe to None (streaming raw=None,
# max-steps fallback step with no model_output_message, fake models).
# ---------------------------------------------------------------------------

from types import SimpleNamespace as _NS


def _fake_step(finish_reason):
    return _NS(
        model_output_message=_NS(raw=_NS(choices=[_NS(finish_reason=finish_reason)]))
    )


def test_step_finish_reason_detects_length_cut():
    assert runner._step_finish_reason(_fake_step("length")) == "length"


def test_step_finish_reason_stop_is_not_a_cut():
    assert runner._step_finish_reason(_fake_step("stop")) == "stop"


def test_step_finish_reason_fails_safe_to_none():
    # streaming: raw is None
    assert runner._step_finish_reason(_NS(model_output_message=_NS(raw=None))) is None
    # max-steps fallback step: no model_output_message at all
    assert runner._step_finish_reason(_NS()) is None
    # empty choices
    assert (
        runner._step_finish_reason(_NS(model_output_message=_NS(raw=_NS(choices=[]))))
        is None
    )


# ---------------------------------------------------------------------------
# S89 — transient-generation retry.
#
# Why this exists: production mission pedr-retrieval-evaluation-R001 died at
# step 1 on AgentGenerationError('...Request timed out.') having consumed 0
# tokens and made 0 tool calls. `blocked` is a terminal status — no worker path
# ever returns a mission to 'queued' — so a 16-second connect outage destroyed
# the run and required a human to re-dispatch. These tests pin the three
# conditions that decide whether re-running is safe, not merely that a retry
# loop exists.
# ---------------------------------------------------------------------------


class _AgentGenerationError(Exception):
    """Stands in for smolagents.AgentGenerationError (matched by class name)."""


_AgentGenerationError.__name__ = "AgentGenerationError"


def _zero_token_step():
    """An empty failed ActionStep with neither billing nor retained output."""
    step = ActionStep()
    step.token_usage = _MockTokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)
    step.model_output = None
    step.tool_calls = None
    step.observations = None
    step.action_output = None
    return step


def _unmetered_paid_step():
    """A completed model action whose provider usage metadata is missing."""

    step = _zero_token_step()
    step.token_usage = None
    step.model_output = "retained paid model output"
    return step


def _install_scripted_agent(monkeypatch: pytest.MonkeyPatch, *, outcomes, billed=False):
    """Install a CodeAgent whose successive runs follow ``outcomes``.

    Each outcome is either an exception instance to raise or a string to return
    as report output. ``billed`` controls whether the agent's recorded step
    reports token usage — i.e. whether the failed attempt cost money.
    """
    constructed: list[object] = []

    class _ScriptedAgent(_MockCodeAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.memory = _MockMemory([ActionStep() if billed else _zero_token_step()])
            constructed.append(self)

        def run(self, user_directive, return_full_result=True):  # noqa: ARG002
            outcome = outcomes[min(len(constructed) - 1, len(outcomes) - 1)]
            if isinstance(outcome, Exception):
                raise outcome
            return _MockResult(
                output=outcome,
                token_usage=_MockTokenUsage(
                    input_tokens=100, output_tokens=200, total_tokens=300
                ),
            )

    fake = types.ModuleType("smolagents")
    fake.CodeAgent = _ScriptedAgent
    fake.OpenAIServerModel = MagicMock()
    fake.tool = lambda f: f
    monkeypatch.setitem(sys.modules, "smolagents", fake)
    importlib.reload(runner)

    slept: list[float] = []
    monkeypatch.setattr(runner.time, "sleep", slept.append)
    monkeypatch.setattr(runner, "SYNTHESIS_RETRY_BASE_DELAY_S", 30.0)
    return constructed, slept


def _install_paid_recovery_agent(
    monkeypatch: pytest.MonkeyPatch,
    *,
    recovery_text: str,
    finish_reason: str | None,
    scratchpad_draft: str = "",
    recovery_raises: bool = False,
    unmetered_paid: bool = False,
    recovery_usage: object | None = None,
):
    """Install a billed failing agent with one observable recovery call."""

    constructed: list[object] = []

    class _PaidRecoveryAgent(_MockCodeAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            retained_step = (
                _unmetered_paid_step()
                if unmetered_paid
                else ActionStep(is_final=False)
            )
            self.memory = _MockMemory([retained_step])
            self.recovery_calls = 0
            constructed.append(self)

        def run(self, user_directive, return_full_result=True):  # noqa: ARG002
            if scratchpad_draft:
                scratchpad_write = next(
                    tool for tool in self.tools if tool.__name__ == "scratchpad_write"
                )
                scratchpad_write("report_draft", scratchpad_draft)
            raise _AgentGenerationError(
                "Connection reset at https://signed.example/?token=DO_NOT_PERSIST"
            )

        def provide_final_answer(self, task):  # noqa: ARG002
            self.recovery_calls += 1
            if recovery_raises:
                raise RuntimeError("fallback leaked token=DO_NOT_PERSIST")
            raw = (
                types.SimpleNamespace(
                    choices=[types.SimpleNamespace(finish_reason=finish_reason)]
                )
                if finish_reason is not None
                else None
            )
            return types.SimpleNamespace(
                content=recovery_text,
                raw=raw,
                token_usage=(
                    recovery_usage
                    if recovery_usage is not None
                    else _MockTokenUsage(
                        input_tokens=3,
                        output_tokens=2,
                        total_tokens=5,
                    )
                ),
            )

    fake = types.ModuleType("smolagents")
    fake.CodeAgent = _PaidRecoveryAgent
    fake.OpenAIServerModel = MagicMock()
    fake.tool = lambda f: f
    monkeypatch.setitem(sys.modules, "smolagents", fake)
    importlib.reload(runner)
    return constructed


def _run(**kwargs):
    return runner.run_codeagent_synthesis(
        system_instructions="system",
        user_directive="user",
        model=MagicMock(),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("finish_reason", "recovery_text", "expected_output", "expected_result", "truncated"),
    (
        (
            "stop",
            "# Markerless partial",
            "# Markerless partial",
            "uncited",
            False,
        ),
        (
            "length",
            "# Truncated\n\nClaim [cite:https://example.test/a]",
            "# Truncated\n\nClaim [cite:https://example.test/a]",
            "truncated",
            True,
        ),
        (
            None,
            "# Unproven\n\nClaim [cite:https://example.test/a]",
            "",
            "unknown_finish",
            False,
        ),
        (
            "stop",
            "<code>report = '# Code, not a report [cite:https://example.test/a]'\n"
            "final_answer(report)</code>",
            "",
            "invalid_wrapper",
            False,
        ),
        (
            "stop",
            "<code># Wrapped, not raw [cite:https://example.test/a]</code>",
            "",
            "invalid_wrapper",
            False,
        ),
        (
            "stop",
            "```markdown\n# Fenced, not raw [cite:https://example.test/a]\n```",
            "",
            "invalid_wrapper",
            False,
        ),
        (
            "stop",
            "final_answer ('# Spaced call [cite:https://example.test/a]')",
            "",
            "invalid_wrapper",
            False,
        ),
        (
            "stop",
            "report = '# Multi-statement [cite:https://example.test/a]'\n"
            "final_answer(report)",
            "",
            "invalid_wrapper",
            False,
        ),
        (
            "stop",
            "```markdown\n# Unclosed fence [cite:https://example.test/a]",
            "",
            "invalid_wrapper",
            False,
        ),
        (
            "stop",
            "<code># Unclosed code wrapper [cite:https://example.test/a]",
            "",
            "invalid_wrapper",
            False,
        ),
        (
            "stop",
            "report = '# Executable string [cite:https://example.test/a]'",
            "",
            "invalid_wrapper",
            False,
        ),
    ),
    ids=(
        "stop_uncited",
        "length_cited",
        "unknown_finish_cited",
        "stop_cited_nonliteral_wrapper",
        "stop_cited_code_wrapper",
        "stop_cited_markdown_fence",
        "stop_cited_spaced_call",
        "stop_cited_multistatement_call",
        "stop_cited_unclosed_fence",
        "stop_cited_unclosed_code_wrapper",
        "stop_cited_raw_python",
    ),
)
def test_paid_generation_recovery_requires_stop_and_citations(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    finish_reason: str | None,
    recovery_text: str,
    expected_output: str,
    expected_result: str,
    truncated: bool,
) -> None:
    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    constructed = _install_paid_recovery_agent(
        monkeypatch,
        recovery_text=recovery_text,
        finish_reason=finish_reason,
    )

    result = _run()

    assert len(constructed) == 1, "paid work must never construct a fresh agent"
    assert constructed[0].recovery_calls == 1
    assert result.output_markdown == expected_output
    _assert_safe_error(
        result.error,
        category="execution",
        exception_class="AgentGenerationError",
    )
    assert result.generation_error_recovered is False
    assert result.generation_recovery_result == expected_result
    assert result.output_truncated is truncated
    assert "DO_NOT_PERSIST" not in repr(result)
    assert "DO_NOT_PERSIST" not in caplog.text


def test_failed_memory_finalization_salvages_cited_scratchpad_without_clearing_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    draft = "# Draft\n\nSupported [cite:https://example.test/draft]"
    constructed = _install_paid_recovery_agent(
        monkeypatch,
        recovery_text="",
        finish_reason=None,
        scratchpad_draft=draft,
        recovery_raises=True,
    )

    result = _run()

    assert len(constructed) == 1
    assert constructed[0].recovery_calls == 1
    assert result.output_markdown == draft
    assert result.recovered_from_scratchpad is True
    assert result.generation_recovery_result == "failed"
    assert result.generation_error_recovered is False
    _assert_safe_error(
        result.error,
        category="execution",
        exception_class="AgentGenerationError",
    )
    assert "DO_NOT_PERSIST" not in repr(result)
    assert "DO_NOT_PERSIST" not in caplog.text


def test_total_only_mapping_recovery_usage_is_added_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manual recovery call is distinct from retained ActionStep billing."""

    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    _install_paid_recovery_agent(
        monkeypatch,
        recovery_text=(
            "# Recovered\n\nSupported [cite:https://example.test/source]"
        ),
        finish_reason="stop",
        recovery_usage={"total_tokens": 5},
    )

    result = _run()

    assert result.token_usage == {"input": 100, "output": 200, "total": 305}
    assert result.token_usage_complete is False


def test_transient_generation_failure_is_retried_and_can_succeed(
    monkeypatch: pytest.MonkeyPatch,
):
    """The exact production failure must not end the mission.

    A timed-out model call that billed nothing is an outage, not a verdict on
    the mission — the run has to survive it and produce the report.
    """
    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    constructed, slept = _install_scripted_agent(
        monkeypatch,
        outcomes=[
            _AgentGenerationError(
                "Error in generating model output:\nRequest timed out."
            ),
            "# Recovered Report\n\nBody with [cite:https://example.com] marker.",
        ],
    )

    result = _run()

    assert result.error is None
    assert "Recovered Report" in result.output_markdown
    assert len(constructed) == 2, "a fresh agent per attempt, so memory never carries over"
    assert slept == [30.0], "backoff must actually elapse before the retry"


def test_paid_work_is_never_re_run(monkeypatch: pytest.MonkeyPatch):
    """A transient error AFTER the model billed tokens must not restart the run.

    Retrying would charge the mission twice and could ship a different report
    than the partial state already recorded. Cost safety outranks recovery.
    """
    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    constructed, slept = _install_scripted_agent(
        monkeypatch,
        outcomes=[_AgentGenerationError("Request timed out.")],
        billed=True,
    )

    result = _run()

    _assert_safe_error(
        result.error,
        category="execution",
        exception_class="AgentGenerationError",
    )
    assert len(constructed) == 1, "billed work must never be re-run"
    assert slept == []
    assert result.generation_failure_retry_eligible is False
    assert result.generation_failure_retry_suppressed_reason == "paid_progress"


def test_unmetered_retained_action_is_finalized_without_fresh_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing usage metadata must not make retained paid memory disposable."""

    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    constructed = _install_paid_recovery_agent(
        monkeypatch,
        recovery_text=(
            "# Recovered\n\nSupported [cite:https://example.test/source]"
        ),
        finish_reason="stop",
        unmetered_paid=True,
    )

    result = _run()

    assert len(constructed) == 1
    assert constructed[0].recovery_calls == 1
    assert result.error is None
    assert result.generation_error_recovered is True
    assert result.generation_failure_consumed_tokens == 0
    assert result.generation_failure_retry_eligible is False
    assert result.generation_failure_retry_suppressed_reason == "paid_progress"


def test_paid_recovery_wrapper_fact_survives_max_step_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Max-step post-processing cannot erase prior wrapper normalization."""

    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    report = "# Recovered\n\nSupported [cite:https://example.test/source]"
    _install_paid_recovery_agent(
        monkeypatch,
        recovery_text=f"<code>final_answer({report!r})</code>",
        finish_reason="stop",
    )

    result = _run(max_steps=1)

    assert result.output_markdown == report
    assert result.hit_max_steps is True
    assert result.output_wrapper_removed is True
    assert result.generation_error_recovered is True


def test_deterministic_generation_failure_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
):
    """An auth/config failure fails identically every time.

    Retrying it only burns lease time before the mission fails anyway, so the
    failure must surface immediately.
    """
    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    constructed, slept = _install_scripted_agent(
        monkeypatch,
        outcomes=[_AgentGenerationError("Authentication failed: invalid api key")],
    )

    result = _run()

    _assert_safe_error(
        result.error,
        category="execution",
        exception_class="AgentGenerationError",
    )
    assert len(constructed) == 1
    assert slept == []
    assert result.generation_failure_retry_eligible is False
    assert result.generation_failure_retry_suppressed_reason == "deterministic"


def test_non_generation_errors_keep_the_original_failure_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    """Errors raised by the agent's own generated code are not model outages.

    Only AgentGenerationError means the model call itself failed; anything else
    keeps the pre-existing behaviour of surfacing on the result unchanged.
    """
    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    constructed, slept = _install_scripted_agent(
        monkeypatch,
        outcomes=[RuntimeError("model timeout")],
    )

    result = _run()

    _assert_safe_error(
        result.error,
        category="execution",
        exception_class="RuntimeError",
    )
    assert result.output_markdown == ""
    assert len(constructed) == 1
    assert slept == []


def test_retry_exhaustion_surfaces_the_failure_with_bounded_backoff(
    monkeypatch: pytest.MonkeyPatch,
):
    """A sustained outage still fails, but only after a bounded wait.

    Pins both halves: the run does not loop forever, and it does not give up
    after a single 16-second window the way the production failure did.
    """
    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    constructed, slept = _install_scripted_agent(
        monkeypatch,
        outcomes=[_AgentGenerationError("Connection error.")],
    )

    result = _run()

    _assert_safe_error(
        result.error,
        category="execution",
        exception_class="AgentGenerationError",
    )
    assert result.output_markdown == ""
    assert len(constructed) == runner.SYNTHESIS_RETRY_ATTEMPTS
    assert slept == [30.0, 60.0], "exponential backoff, and no sleep after the last try"
    assert result.generation_failure_retry_eligible is True
    assert result.generation_failure_retry_suppressed_reason == "attempts_exhausted"


def test_retry_guard_requires_all_three_conditions():
    """Unit-level pin on the guard itself, independent of the run loop."""
    zero = _MockMemory([_zero_token_step()])
    billed = _MockMemory([ActionStep()])
    unmetered_paid = _MockMemory([_unmetered_paid_step()])
    message_only_step = _zero_token_step()
    message_only_step.model_output_message = object()
    message_only_paid = _MockMemory([message_only_step])
    transient = _AgentGenerationError("Request timed out.")

    assert runner._is_retryable_generation_failure(transient, _NS(memory=zero))
    assert not runner._is_retryable_generation_failure(transient, _NS(memory=billed))
    assert not runner._is_retryable_generation_failure(
        transient, _NS(memory=unmetered_paid)
    )
    assert not runner._is_retryable_generation_failure(
        transient, _NS(memory=message_only_paid)
    )
    assert not runner._is_retryable_generation_failure(
        _AgentGenerationError("context_length_exceeded"), _NS(memory=zero)
    )
    assert not runner._is_retryable_generation_failure(
        RuntimeError("Request timed out."), _NS(memory=zero)
    )


@pytest.mark.parametrize(
    ("status_code", "expected_reason", "transient"),
    (
        (400, "invalid_request", False),
        (401, "authentication", False),
        (402, "balance", False),
        (403, "authentication", False),
        (408, "timeout", True),
        (422, "invalid_request", False),
        (429, "rate_limit", True),
        (500, "server", True),
        (503, "server", True),
    ),
)
def test_generation_failure_classification_is_typed_and_content_free(
    status_code: int,
    expected_reason: str,
    transient: bool,
) -> None:
    class APIStatusError(Exception):
        def __init__(self) -> None:
            super().__init__("secret=DO_NOT_PERSIST")
            self.status_code = status_code

    wrapped = _AgentGenerationError("wrapper secret=DO_NOT_PERSIST")
    wrapped.__cause__ = APIStatusError()

    classification = runner._generation_failure_classification(wrapped)
    reason = classification.reason

    assert reason == expected_reason
    assert reason in runner._GENERATION_FAILURE_REASONS
    assert (reason in runner._TRANSIENT_GENERATION_REASONS) is transient
    assert classification.source == "status"
    assert classification.root_class == "APIStatusError"
    assert classification.http_status == status_code
    assert "DO_NOT_PERSIST" not in reason


def test_bad_request_context_overflow_keeps_specific_reason_without_raw_text() -> None:
    """A generic HTTP 400 must not erase a specific context-length cause."""

    class BadRequestError(Exception):
        status_code = 400

    wrapped = _AgentGenerationError("wrapper secret=DO_NOT_PERSIST")
    wrapped.__cause__ = BadRequestError(
        "maximum context length exceeded secret=DO_NOT_PERSIST"
    )

    classification = runner._generation_failure_classification(wrapped)

    assert classification.reason == "context_length"
    assert classification.source == "message_marker"
    assert classification.root_class == "BadRequestError"
    assert classification.http_status == 400
    assert "DO_NOT_PERSIST" not in repr(classification)


def test_bad_request_timeout_word_never_becomes_retryable() -> None:
    """A recognized 4xx status outranks transient wording in its body."""

    class BadRequestError(Exception):
        status_code = 400

    wrapped = _AgentGenerationError("wrapper")
    wrapped.__cause__ = BadRequestError("upstream timeout parameter is invalid")

    classification = runner._generation_failure_classification(wrapped)

    assert classification.reason == "invalid_request"
    assert classification.source == "status"
    assert classification.http_status == 400


# ---------------------------------------------------------------------------
# S90 (#959) — error_transient carries the retry verdict past the run.
#
# In-process retry absorbs a ~2min outage. A longer one still exhausts the
# attempts, and the mission then lands terminally in 'blocked' with the lease's
# remaining attempts unreachable. error_transient is what lets the worker tell
# "the provider was down" apart from "this mission is broken", so it can return
# the mission to 'queued' instead. It must stay exactly as conservative as the
# in-process guard — same predicate, including the zero-token condition.
# ---------------------------------------------------------------------------


def test_exhausted_transient_outage_is_flagged_transient(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    _install_scripted_agent(
        monkeypatch,
        outcomes=[_AgentGenerationError("Connection error.")],
    )

    result = _run()

    assert result.error is not None
    assert result.error_transient is True


def test_deterministic_failure_is_not_flagged_transient(
    monkeypatch: pytest.MonkeyPatch,
):
    """A bad request fails identically on every attempt — requeuing it loops."""
    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    _install_scripted_agent(
        monkeypatch,
        outcomes=[_AgentGenerationError("context_length_exceeded")],
    )

    result = _run()

    assert result.error is not None
    assert result.error_transient is False


def test_billed_transient_failure_is_not_flagged_transient(
    monkeypatch: pytest.MonkeyPatch,
):
    """Tokens were spent — a requeue would re-run from step 1 and double-charge."""
    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    _install_scripted_agent(
        monkeypatch,
        outcomes=[_AgentGenerationError("Request timed out.")],
        billed=True,
    )

    result = _run()

    assert result.error is not None
    assert result.error_transient is False


def test_successful_run_is_never_flagged_transient(monkeypatch: pytest.MonkeyPatch):
    """A retry that then succeeds must not leave the flag set from the failure."""
    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    _install_scripted_agent(
        monkeypatch,
        outcomes=[
            _AgentGenerationError("Request timed out."),
            "# Report\n\nBody with [cite:https://example.com] marker.",
        ],
    )

    result = _run()

    assert result.error is None
    assert result.error_transient is False


def test_raw_paid_response_prevents_restart_when_sdk_retained_no_usage(monkeypatch) -> None:
    """An SDK parsing failure cannot turn an observed paid response into a free retry."""
    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    constructed, slept = _install_scripted_agent(
        monkeypatch, outcomes=[_AgentGenerationError("Request timed out.")], billed=False,
    )
    agent_class = sys.modules["smolagents"].CodeAgent
    original_run = agent_class.run

    def run_after_observed_response(self, *args, **kwargs):
        self.model.provider_attempts.append({
            "event_id": "paid-before-parse", "status": "returned",
            "usage": {"input": 100, "output": 20, "total": 120, "complete": True},
        })
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(agent_class, "run", run_after_observed_response)
    model = types.SimpleNamespace(provider_attempts=[], model_id="deepseek-flash")
    result = runner.run_codeagent_synthesis(
        system_instructions="system", user_directive="user", model=model,
    )
    assert len(constructed) == 1 and slept == []
    assert result.generation_failure_retry_suppressed_reason == "paid_progress"
    assert result.error_transient is False
    assert result.token_usage == {"input": 100, "output": 20, "total": 120}
    assert result.token_usage_complete is False
