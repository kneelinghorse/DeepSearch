"""Integration test for the CodeAgent runner with the live retrieval tools.

Stitches together: the runner's tool factory, the live tool modules, and
a stubbed smolagents CodeAgent. All HTTP, Linkup, OpenAlex, and Semantic
Scholar calls are mocked at the respective module boundaries so this test
makes no live API calls.

Verifies end-to-end:
- ``run_codeagent_synthesis`` wires the bundle from ``build_codeagent_tools()``
  into the CodeAgent without an admitted_pool argument.
- The agent (a stub here) can invoke each of the seven tools and observe their
  return shapes — confirming the smolagents @tool wrapping in
  ``build_codeagent_tools`` is well-formed. (S79 m01.a appended
  ``scholar_search`` as the 6th tool; S79 m02.a appended ``completeness_check``
  as the 7th.)
- Telemetry (token usage, tool_call_summary, step_trace) is captured on the
  returned ``SynthesisResult``.
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest


from deepsearch.agent import codeagent_runner as runner


@dataclass
class _MockTokenUsage:
    input_tokens: int = 250
    output_tokens: int = 750
    total_tokens: int = 1000


@dataclass
class _MockResult:
    output: str
    token_usage: _MockTokenUsage


class ActionStep:
    """Named to match smolagents — runner counts steps via type().__name__."""

    def __init__(self, *, is_final: bool = True):
        self.is_final_answer = is_final
        self.token_usage = _MockTokenUsage()
        self.timing = None
        self.tool_calls = None
        self.model_output = "model output"


class _MockMemory:
    def __init__(self, steps):
        self.steps = steps


class _ToolExercisingCodeAgent:
    """Stub CodeAgent that, on .run(), exercises each of its 7 tools once.

    This is the load-bearing assertion of the integration test: the tools
    accept arguments, return the documented shapes, and the call_log captures
    each invocation. ``scholar_search`` was appended at S79 m01.a (PD93);
    ``completeness_check`` was appended at S79 m02.a (PD97).
    """

    last_tool_outputs: dict = {}

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
        self.memory = _MockMemory([ActionStep()])

    def run(self, user_directive, return_full_result=True):  # noqa: ARG002
        # Tools are in the canonical order: web_search, source_fetch,
        # url_liveness, scratchpad_write, scratchpad_read, scholar_search,
        # completeness_check.
        (
            web_search,
            source_fetch,
            url_liveness,
            scratchpad_write,
            scratchpad_read,
            scholar_search,
            completeness_check,
        ) = self.tools

        outputs: dict = {}
        outputs["web_search"] = web_search("aws lambda concurrency", 3)
        outputs["source_fetch"] = source_fetch("https://docs.aws.amazon.com/lambda/x")
        outputs["url_liveness"] = url_liveness("https://docs.aws.amazon.com/lambda/x")
        outputs["scratchpad_write"] = scratchpad_write("notes", "lambda timeout = 15min")
        outputs["scratchpad_read"] = scratchpad_read("notes")
        outputs["scholar_search"] = scholar_search("OAuth 2.0 PKCE", 3)
        outputs["completeness_check"] = completeness_check(
            "# Synthesis Output\n\n## Comparison Matrix\nbody."
        )
        _ToolExercisingCodeAgent.last_tool_outputs = outputs

        return _MockResult(
            output=(
                "# Synthesis Output\n\n"
                "Lambda timeout is 15 minutes [1].\n\n"
                "## Sources\n[1] https://docs.aws.amazon.com/lambda/x"
            ),
            token_usage=_MockTokenUsage(),
        )


@pytest.fixture(autouse=True)
def _stub_environment(monkeypatch: pytest.MonkeyPatch):
    """Set credentials so guards pass; install fake smolagents; patch tool impls."""

    monkeypatch.setenv("LINKUP_API_KEY", "test-linkup-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")

    # Fake smolagents — passthrough @tool, our exercising CodeAgent.
    fake = types.ModuleType("smolagents")
    fake.CodeAgent = _ToolExercisingCodeAgent
    fake.OpenAIServerModel = MagicMock()
    fake.tool = lambda f: f
    monkeypatch.setitem(sys.modules, "smolagents", fake)
    importlib.reload(runner)

    # Stub the live tool implementations at their module boundaries so
    # build_codeagent_tools' lazy imports pick them up. ``scholar_search``
    # was appended at S79 m01.a — stubbed here alongside the others.
    import deepsearch.agent.tools.web_search as ws_mod
    import deepsearch.agent.tools.source_fetch as sf_mod
    import deepsearch.agent.tools.url_liveness as ul_mod
    import deepsearch.agent.tools.scholar_search as ss_mod

    def fake_web_search(query, max_results=20):
        return [
            {"title": "Lambda scaling", "url": "https://docs.aws.amazon.com/lambda/x", "snippet": "..."},
            {"title": "Cloud Run", "url": "https://cloud.google.com/run/docs", "snippet": "..."},
        ]

    def fake_source_fetch(url, timeout_s=15.0):
        return {
            "text": "<html>Lambda timeout is 15 minutes</html>",
            "status": "ok",
            "status_code": 200,
            "fetched_at": "2026-04-28T00:00:00+00:00",
            "final_url": url,
        }

    def fake_url_liveness(url, timeout_s=5.0):
        return {"alive": True, "status_code": 200, "final_url": url, "error": None}

    def fake_scholar_search(query, max_results=20):
        return [
            {
                "title": "RFC 7636 PKCE",
                "url": "https://www.rfc-editor.org/rfc/rfc7636",
                "snippet": "Proof Key for Code Exchange",
                "score": 0.93,
                "authority_tier": "high",
            },
        ]

    monkeypatch.setattr(ws_mod, "web_search", fake_web_search)
    monkeypatch.setattr(sf_mod, "source_fetch", fake_source_fetch)
    monkeypatch.setattr(ul_mod, "url_liveness", fake_url_liveness)
    monkeypatch.setattr(ss_mod, "scholar_search", fake_scholar_search)


def test_runner_end_to_end_with_live_tools_mocked():
    """Run the full runner path with all four live tools mocked + a stub agent
    that invokes each tool once. Confirms wiring is correct."""

    fake_model = MagicMock(name="fake-model")
    result = runner.run_codeagent_synthesis(
        system_instructions="You are a research synthesis agent.",
        user_directive="Synthesise platform tradeoffs for serverless deployment.",
        model=fake_model,
        max_steps=10,
    )

    assert result.error is None
    assert "Lambda timeout is 15 minutes" in result.output_markdown
    assert result.step_count == 1
    assert result.self_terminated is True
    assert result.token_usage["total"] == 1000
    # No upstream pool; authority distribution is derived from the three
    # distinct production sources the mocked retrieval tools collected.
    assert result.pool_size == 0
    assert result.authority_tier_distribution == {"high": 3, "medium": 0, "low": 0}

    # Each tool was invoked exactly once during the stub agent.run().
    summary = result.tool_call_summary
    assert summary["total_calls"] == 7
    assert summary["by_tool"]["web_search"] == 1
    assert summary["by_tool"]["source_fetch"] == 1
    assert summary["by_tool"]["url_liveness"] == 1
    assert summary["by_tool"]["scratchpad_write"] == 1
    assert summary["by_tool"]["scratchpad_read"] == 1
    assert summary["by_tool"]["scholar_search"] == 1
    assert summary["by_tool"]["completeness_check"] == 1
    assert summary["scratchpad_keys"] == ["notes"]


def test_runner_tool_outputs_match_documented_shape():
    """Each tool returns the shape documented in its module docstring."""

    fake_model = MagicMock(name="fake-model")
    runner.run_codeagent_synthesis(
        system_instructions="x",
        user_directive="y",
        model=fake_model,
        max_steps=10,
    )

    outputs = _ToolExercisingCodeAgent.last_tool_outputs

    # web_search → list of dicts with title/url/snippet
    assert isinstance(outputs["web_search"], list) and len(outputs["web_search"]) == 2
    assert set(outputs["web_search"][0].keys()) >= {"title", "url", "snippet"}

    # source_fetch → dict with status taxonomy
    sf = outputs["source_fetch"]
    assert sf["status"] == "ok"
    assert set(sf.keys()) >= {
        "text",
        "status",
        "status_code",
        "fetched_at",
        "final_url",
        "text_chars_total",
        "text_chars_returned",
        "text_truncated_for_tool",
    }
    assert sf["text_chars_returned"] == len(sf["text"])

    # url_liveness → {alive, status_code, final_url, error} (S84 m01 adds the
    # transport-category field; keep in sync with the tool docstring the
    # agent reads and with contract_to_prompt's tool-contract line)
    ul = outputs["url_liveness"]
    assert ul["alive"] is True
    assert set(ul.keys()) == {"alive", "status_code", "final_url", "error"}

    # scratchpad write/read roundtrip
    assert outputs["scratchpad_read"] == "lambda timeout = 15min"
    assert "scratchpad['notes']" in outputs["scratchpad_write"]

    # scholar_search → list of dicts with title/url/snippet/score/authority_tier
    sch = outputs["scholar_search"]
    assert isinstance(sch, list) and len(sch) == 1
    assert set(sch[0].keys()) >= {
        "title",
        "url",
        "snippet",
        "score",
        "authority_tier",
    }
    assert sch[0]["url"] == "https://www.rfc-editor.org/rfc/rfc7636"

    # completeness_check → defensive empty shape when build_codeagent_tools is
    # called without a deliverable_spec (the integration runner doesn't pass one).
    cc = outputs["completeness_check"]
    assert {"satisfied", "missing", "shortfall_score"} <= set(cc.keys())
    assert cc["assessment_status"] == "unassessed"
    assert cc["satisfied"] == []
    assert cc["missing"] == []
    assert cc["shortfall_score"] is None


def test_runner_does_not_take_admitted_pool_argument():
    """Regression: ensure run_codeagent_synthesis no longer accepts admitted_pool."""

    fake_model = MagicMock()
    with pytest.raises(TypeError, match="admitted_pool"):
        runner.run_codeagent_synthesis(  # type: ignore[call-arg]
            system_instructions="x",
            user_directive="y",
            admitted_pool=[{"url": "https://x"}],
            model=fake_model,
        )
