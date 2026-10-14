"""Unit tests for the google-genai Gemini chat adapter."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
pytest.importorskip("langchain_core")
from langchain_core.messages import HumanMessage, SystemMessage

from deepsearch.llm.gemini_genai import GeminiGenAIChatModel, _safe_thinking_budget


def test_safe_thinking_budget_does_not_cap_against_output_tokens() -> None:
    """B11.4: Thinking budget is separate from max_output_tokens, no capping.

    Previously _safe_thinking_budget incorrectly capped thinking_budget to
    max_output_tokens - 512. This was wrong because Gemini 2.5 Pro allocates
    thinking and output tokens separately.
    """
    # Budget within output tokens - unchanged
    assert _safe_thinking_budget(2048, 8000) == 2048
    # Budget exceeding output tokens - should NOT be capped (B11.4 fix)
    assert _safe_thinking_budget(16384, 8000) == 16384  # Was incorrectly capped to 7488
    # Budget much larger than output - still not capped
    assert _safe_thinking_budget(32768, 8000) == 32768
    # Unlimited budget
    assert _safe_thinking_budget(-1, 8000) == -1
    # None returns None
    assert _safe_thinking_budget(None, 8000) is None
    # Minimum clamp to 1
    assert _safe_thinking_budget(0, 8000) == 1


@dataclass(slots=True)
class _FakeUsage:
    prompt_token_count: int = 10
    total_token_count: int = 42
    thoughts_token_count: int = 20
    candidates_token_count: int = 12


@dataclass(slots=True)
class _FakePart:
    text: str


@dataclass(slots=True)
class _FakeContent:
    parts: list[_FakePart]


@dataclass(slots=True)
class _FakeCandidate:
    content: _FakeContent
    finish_reason: str = "STOP"


@dataclass(slots=True)
class _FakeResponse:
    candidates: list[_FakeCandidate]
    usage_metadata: _FakeUsage
    model_version: str = "gemini-2.5-pro"


class _FakeModels:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def generate_content(self, *args, **kwargs):  # noqa: ANN001
        return self._response


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.models = _FakeModels(response)


@pytest.mark.asyncio
async def test_gemini_genai_chat_model_returns_text_and_usage(monkeypatch) -> None:
    response = _FakeResponse(
        candidates=[
            _FakeCandidate(content=_FakeContent(parts=[_FakePart(text="Hello world")]))
        ],
        usage_metadata=_FakeUsage(),
    )
    model = GeminiGenAIChatModel(
        api_key="test-key",
        model="gemini-2.5-pro",
        temperature=0.2,
        max_output_tokens=8000,
        thinking_budget=256,
    )
    monkeypatch.setattr(model, "_client", _FakeClient(response))

    msg = await model.ainvoke(
        [
            SystemMessage(content="system"),
            HumanMessage(content="hi"),
        ]
    )

    assert msg.content == "Hello world"
    assert msg.usage_metadata["input_tokens"] == 10
    assert msg.usage_metadata["total_tokens"] == 42
    assert msg.usage_metadata["output_tokens"] == 32  # total - prompt


@pytest.mark.asyncio
async def test_gemini_genai_chat_model_logs_empty_response(monkeypatch, caplog) -> None:
    """B11.4: Empty responses are detected and logged for debugging."""
    import logging

    # Response with empty content
    response = _FakeResponse(
        candidates=[
            _FakeCandidate(content=_FakeContent(parts=[_FakePart(text="")]))
        ],
        usage_metadata=_FakeUsage(
            prompt_token_count=100,
            total_token_count=16500,
            thoughts_token_count=16384,  # All tokens spent on thinking
            candidates_token_count=0,
        ),
    )
    model = GeminiGenAIChatModel(
        api_key="test-key",
        model="gemini-2.5-pro",
        temperature=0.2,
        max_output_tokens=8000,
        thinking_budget=16384,
    )
    monkeypatch.setattr(model, "_client", _FakeClient(response))

    with caplog.at_level(logging.WARNING, logger="deepsearch.llm.gemini_genai"):
        msg = await model.ainvoke([HumanMessage(content="test")])

    # Response still returned (caller decides what to do)
    assert msg.content == ""
    # Warning logged about empty response
    assert len(caplog.records) >= 1
    assert "Empty or minimal response" in caplog.text
    assert "gemini-2.5-pro" in caplog.text
    assert "thinking_budget=16384" in caplog.text


@pytest.mark.asyncio
async def test_gemini_genai_chat_model_accepts_high_thinking_budget(monkeypatch) -> None:
    """B11.4: High thinking budgets (16384+) are not incorrectly capped."""
    response = _FakeResponse(
        candidates=[
            _FakeCandidate(content=_FakeContent(parts=[_FakePart(text="Result")]))
        ],
        usage_metadata=_FakeUsage(),
    )
    # This would have failed before B11.4 fix because 16384 > 8000 - 512
    model = GeminiGenAIChatModel(
        api_key="test-key",
        model="gemini-2.5-pro",
        temperature=0.2,
        max_output_tokens=8000,
        thinking_budget=16384,  # Higher than max_output_tokens
    )
    monkeypatch.setattr(model, "_client", _FakeClient(response))

    msg = await model.ainvoke([HumanMessage(content="test")])
    assert msg.content == "Result"

