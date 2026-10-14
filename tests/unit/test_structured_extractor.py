"""Unit tests for the provider-portable structured-output extractor (Sprint 54 L1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("pydantic")

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from deepsearch.llm.structured_extractor import (
    GeminiStructuredExtractor,
    JSONInPromptStructuredExtractor,
    _strip_json_wrapping,
)


class _Tool(BaseModel):
    name: str
    is_acronym: bool = False


class _Result(BaseModel):
    tools: list[_Tool]


# --------------------------------------------------------------------------- #
# GeminiStructuredExtractor
# --------------------------------------------------------------------------- #


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
class _FakeUsage:
    prompt_token_count: int = 12
    total_token_count: int = 24
    thoughts_token_count: int = 0
    candidates_token_count: int = 12


@dataclass(slots=True)
class _FakeResponse:
    candidates: list[_FakeCandidate]
    usage_metadata: _FakeUsage = field(default_factory=_FakeUsage)
    parsed: Any = None
    model_version: str = "gemini-2.5-pro"


class _FakeModels:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def generate_content(self, *args: Any, **kwargs: Any) -> _FakeResponse:
        return self._response


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.models = _FakeModels(response)


@pytest.mark.asyncio
async def test_gemini_structured_extractor_happy_path(monkeypatch) -> None:
    """SDK exposes a parsed Pydantic instance; serialized JSON re-validates."""

    parsed = _Result(tools=[_Tool(name="axe-core", is_acronym=False)])
    response = _FakeResponse(
        candidates=[_FakeCandidate(content=_FakeContent(parts=[_FakePart(text="ignored")]))],
        parsed=parsed,
    )

    extractor = GeminiStructuredExtractor(
        api_key="test-key",
        model="gemini-2.5-pro",
        temperature=0.0,
        max_output_tokens=1024,
    )

    from deepsearch.llm import gemini_genai

    real_init = gemini_genai.GeminiGenAIChatModel.__post_init__

    def patched_init(self: Any) -> None:
        real_init(self)
        self._client = _FakeClient(response)

    monkeypatch.setattr(gemini_genai.GeminiGenAIChatModel, "__post_init__", patched_init)

    out = await extractor.extract(
        system_instruction="extract entities",
        prompt="objective text here",
        schema=_Result,
    )
    assert isinstance(out, _Result)
    assert [tool.name for tool in out.tools] == ["axe-core"]


@pytest.mark.asyncio
async def test_gemini_structured_extractor_falls_back_to_text_parts(monkeypatch) -> None:
    """When ``parsed`` is absent, _extract_text reads JSON from candidate parts."""

    json_text = '{"tools":[{"name":"Pa11y","is_acronym":false}]}'
    response = _FakeResponse(
        candidates=[_FakeCandidate(content=_FakeContent(parts=[_FakePart(text=json_text)]))],
        parsed=None,
    )

    extractor = GeminiStructuredExtractor(
        api_key="test-key",
        model="gemini-2.5-pro",
        temperature=0.0,
        max_output_tokens=1024,
    )

    from deepsearch.llm import gemini_genai

    real_init = gemini_genai.GeminiGenAIChatModel.__post_init__

    def patched_init(self: Any) -> None:
        real_init(self)
        self._client = _FakeClient(response)

    monkeypatch.setattr(gemini_genai.GeminiGenAIChatModel, "__post_init__", patched_init)

    out = await extractor.extract(
        system_instruction="extract entities",
        prompt="objective text here",
        schema=_Result,
    )
    assert isinstance(out, _Result)
    assert [tool.name for tool in out.tools] == ["Pa11y"]


# --------------------------------------------------------------------------- #
# JSONInPromptStructuredExtractor
# --------------------------------------------------------------------------- #


class _FakeChatModel:
    """Minimal async chat-model stub. Returns a sequence of canned responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[BaseMessage]] = []
        self.temperature = 0.7

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        if not self._responses:
            return AIMessage(content="")
        return AIMessage(content=self._responses.pop(0))


@pytest.mark.asyncio
async def test_json_in_prompt_happy_path() -> None:
    chat = _FakeChatModel(['{"tools":[{"name":"axe-core","is_acronym":false}]}'])
    extractor = JSONInPromptStructuredExtractor(chat_model=chat)

    out = await extractor.extract(
        system_instruction="extract entities",
        prompt="objective text",
        schema=_Result,
    )
    assert isinstance(out, _Result)
    assert chat.temperature == 0.0  # forced to deterministic decoding
    assert len(chat.calls) == 1


@pytest.mark.asyncio
async def test_json_in_prompt_retries_then_succeeds() -> None:
    chat = _FakeChatModel(
        [
            "this is not json",
            '{"tools":[{"name":"Pa11y","is_acronym":false}]}',
        ]
    )
    extractor = JSONInPromptStructuredExtractor(chat_model=chat)

    out = await extractor.extract(
        system_instruction="extract entities",
        prompt="objective text",
        schema=_Result,
    )
    assert isinstance(out, _Result)
    assert [tool.name for tool in out.tools] == ["Pa11y"]
    assert len(chat.calls) == 2
    # Retry user message must include corrective prefix.
    retry_user = chat.calls[1][1]
    assert "previous response was malformed" in str(retry_user.content)


@pytest.mark.asyncio
async def test_json_in_prompt_retry_exhaustion_returns_none() -> None:
    chat = _FakeChatModel(["bad", "still bad", "still still bad"])
    extractor = JSONInPromptStructuredExtractor(chat_model=chat)

    out = await extractor.extract(
        system_instruction="extract entities",
        prompt="objective text",
        schema=_Result,
    )
    assert out is None
    assert len(chat.calls) == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_json_in_prompt_strips_markdown_fences() -> None:
    fenced = '```json\n{"tools":[{"name":"OODS","is_acronym":true}]}\n```'
    chat = _FakeChatModel([fenced])
    extractor = JSONInPromptStructuredExtractor(chat_model=chat)

    out = await extractor.extract(
        system_instruction="extract entities",
        prompt="objective text",
        schema=_Result,
    )
    assert isinstance(out, _Result)
    assert out.tools[0].is_acronym is True


def test_strip_json_wrapping_handles_leading_prose_and_fences() -> None:
    assert _strip_json_wrapping("```json\n{\"a\":1}\n```") == '{"a":1}'
    assert _strip_json_wrapping("Here is the JSON: {\"a\":1}") == '{"a":1}'
    assert _strip_json_wrapping("{\"a\":1}") == '{"a":1}'
    assert _strip_json_wrapping("") == ""


# --------------------------------------------------------------------------- #
# Dispatch helper
# --------------------------------------------------------------------------- #


def test_build_structured_extractor_dispatches_to_gemini(monkeypatch) -> None:
    from deepsearch.config import DeepSearchSettings
    from deepsearch.llm.structured_extractor import GeminiStructuredExtractor

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BACKEND", "gemini")
    settings = DeepSearchSettings.load()
    extractor = settings.build_structured_extractor()

    assert isinstance(extractor, GeminiStructuredExtractor)
    assert extractor.api_key == "test-key"


def test_build_structured_extractor_dispatches_universal_for_non_gemini(monkeypatch) -> None:
    """For deepseek/minimax/ollama, return the JSON-in-prompt extractor.

    We patch ``create_model`` so the test does not need real backend keys.
    """

    from deepsearch.config import DeepSearchSettings
    from deepsearch.llm.structured_extractor import JSONInPromptStructuredExtractor

    monkeypatch.setenv("LLM_BACKEND", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    sentinel = object()

    import deepsearch.agent.llm_client as llm_client

    monkeypatch.setattr(llm_client, "create_model", lambda config: sentinel)

    settings = DeepSearchSettings.load()
    extractor = settings.build_structured_extractor()

    assert isinstance(extractor, JSONInPromptStructuredExtractor)
    assert extractor.chat_model is sentinel
