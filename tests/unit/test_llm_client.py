"""Unit tests for llm_client module.

Tests cover:
- create_model dispatch (Ollama, Gemini, Gemini 2.5 Pro)
- call_llm with proper SystemMessage/HumanMessage formatting
- Empty response recovery with prompt rephrasing
- coerce_content normalization for various response formats
- Message type verification (B16.2 — never raw strings)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepsearch.agent.llm_client import (
    call_llm,
    coerce_content,
    create_model,
)
from deepsearch.config import LLMBackendConfig

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ImportError:
    AIMessage = MagicMock
    HumanMessage = MagicMock
    SystemMessage = MagicMock


def _make_settings(**overrides):
    defaults = {
        "llm_max_retries": 1,
        "llm_backoff_base": 0.01,
        "llm_backoff_max": 0.1,
        "llm_backoff_jitter_ms": 0,
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


class TestCreateModel:
    """create_model backend dispatch."""

    def test_unsupported_backend_raises(self) -> None:
        config = LLMBackendConfig(backend="unknown", model="test", temperature=0.7, max_output_tokens=8000)
        with pytest.raises(RuntimeError, match="Unsupported backend"):
            create_model(config)

    def test_gemini_no_api_key_raises(self) -> None:
        config = LLMBackendConfig(
            backend="gemini", model="gemini-2.5-flash", temperature=0.7,
            max_output_tokens=8000, api_key="",
        )
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            create_model(config)

    def test_gemini_25_pro_no_api_key_raises(self) -> None:
        config = LLMBackendConfig(
            backend="gemini", model="gemini-2.5-pro", temperature=0.7,
            max_output_tokens=8000, api_key="",
        )
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            create_model(config)

    def test_gemini_25_pro_uses_genai_model(self) -> None:
        with patch(
            "deepsearch.agent.llm_client.GeminiGenAIChatModel",
            create=True,
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            config = LLMBackendConfig(
                backend="gemini",
                model="gemini-2.5-pro",
                temperature=0.7,
                max_output_tokens=8000,
                api_key="test-key",
            )
            with patch.dict(
                "sys.modules",
                {"deepsearch.llm.gemini_genai": MagicMock(GeminiGenAIChatModel=mock_cls)},
            ):
                model = create_model(config)
                mock_cls.assert_called_once()
                call_kwargs = mock_cls.call_args.kwargs
                assert call_kwargs["api_key"] == "test-key"
                assert call_kwargs["thinking_budget"] == 16384

    def test_deepseek_model_gets_an_explicit_request_timeout(self) -> None:
        """The DeepSeek chat model must never be built without a timeout.

        WHY: the openai client defaults to a 600s timeout with retries, so an
        unresponsive endpoint can hold a mission lease for tens of minutes
        (observed: 18 minutes on 2026-08-13). An explicit timeout converts that
        hang into a fast, retryable failure.
        """
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            config = LLMBackendConfig(
                backend="deepseek",
                model="deepseek-reasoner",
                temperature=0.0,
                max_output_tokens=8000,
                api_key="test-key",
                base_url="https://api.deepseek.com",
                request_timeout=42.0,
            )
            create_model(config)

        assert mock_cls.call_args.kwargs["timeout"] == 42.0
        assert mock_cls.call_args.kwargs["max_retries"] == 0

    def test_deepseek_timeout_falls_back_to_the_default_when_unset(self) -> None:
        """A config built without an explicit timeout still gets a bounded one."""
        from deepsearch.config import DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS

        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            config = LLMBackendConfig(
                backend="deepseek",
                model="deepseek-reasoner",
                temperature=0.0,
                max_output_tokens=8000,
                api_key="test-key",
            )
            create_model(config)

        assert (
            mock_cls.call_args.kwargs["timeout"] == DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS
        )
        assert mock_cls.call_args.kwargs["max_retries"] == 0


class TestCoerceContent:
    """coerce_content normalizes various response types to plain text."""

    def test_none_returns_empty(self) -> None:
        assert coerce_content(None) == ""

    def test_string_passthrough(self) -> None:
        assert coerce_content("hello") == "hello"

    def test_dict_with_content(self) -> None:
        assert coerce_content({"content": "hello"}) == "hello"

    def test_dict_with_text(self) -> None:
        assert coerce_content({"text": "hello"}) == "hello"

    def test_list_of_strings(self) -> None:
        assert coerce_content(["hello", "world"]) == "hello\nworld"

    def test_list_of_dicts_with_text(self) -> None:
        result = coerce_content([
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ])
        assert "hello" in result
        assert "world" in result

    def test_list_skips_none(self) -> None:
        assert coerce_content([None, "hello", None]) == "hello"

    def test_list_skips_empty_strings(self) -> None:
        assert coerce_content(["", "  ", "hello"]) == "hello"

    def test_aimessage_with_content(self) -> None:
        msg = MagicMock()
        msg.content = "response text"
        assert coerce_content(msg) == "response text"

    def test_fallback_to_str(self) -> None:
        assert coerce_content(42) == "42"


class TestCallLlm:
    """call_llm with mocked LLM backend."""

    def test_formats_as_system_and_human_messages(self) -> None:
        """B16.2: Prompts must be SystemMessage + HumanMessage, never raw strings."""
        captured_messages = []

        async def mock_invoke(messages):
            captured_messages.extend(messages)
            return AIMessage(content="Research finding about topic with sufficient detail to pass the minimum length check that requires at least one hundred characters of content")

        mock_llm = MagicMock()
        mock_llm.ainvoke = mock_invoke

        result = asyncio.run(
            call_llm(mock_llm, "You are a researcher", "Find info about X")
        )

        assert len(captured_messages) == 2
        assert type(captured_messages[0]).__name__ == "SystemMessage"
        assert type(captured_messages[1]).__name__ == "HumanMessage"
        assert captured_messages[0].content == "You are a researcher"
        assert captured_messages[1].content == "Find info about X"

    def test_returns_response_text(self) -> None:
        async def mock_invoke(messages):
            return AIMessage(content="A detailed research finding with sufficient length to pass the check")

        mock_llm = MagicMock()
        mock_llm.ainvoke = mock_invoke

        result = asyncio.run(
            call_llm(mock_llm, "system", "user " * 50)
        )
        assert "detailed research finding" in result

    def test_empty_response_retries_with_rephrase(self) -> None:
        """Empty responses trigger a retry with rephrased prompt."""
        call_count = 0

        async def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(content="short")  # < 100 chars
            return AIMessage(content="A much longer and more detailed research response " * 5)

        mock_llm = MagicMock()
        mock_llm.ainvoke = mock_invoke

        result = asyncio.run(
            call_llm(mock_llm, "system", "user prompt")
        )
        assert call_count == 2
        assert "detailed research response" in result

    def test_empty_response_keeps_original_if_retry_worse(self) -> None:
        """If retry is worse than original, keeps original."""
        call_count = 0

        async def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(content="original short")
            return AIMessage(content="")  # retry is worse

        mock_llm = MagicMock()
        mock_llm.ainvoke = mock_invoke

        result = asyncio.run(
            call_llm(mock_llm, "system", "user")
        )
        assert result == "original short"

    def test_no_ainvoke_raises(self) -> None:
        mock_llm = object()  # no ainvoke attribute

        with pytest.raises(RuntimeError, match="async invocation"):
            asyncio.run(
                call_llm(mock_llm, "system", "user")
            )

    def test_with_settings_uses_retry(self) -> None:
        settings = _make_settings()

        async def mock_invoke(messages):
            return AIMessage(content="A sufficiently long response for the length check to pass " * 3)

        mock_llm = MagicMock()
        mock_llm.ainvoke = mock_invoke

        result = asyncio.run(
            call_llm(mock_llm, "system", "user", settings=settings)
        )
        assert "sufficiently long" in result
