"""LLM client — model creation, invocation, and message formatting.

Stateless module: creates LLM instances, calls them, handles empty responses.
No AgentState dependency — only depends on config and resilience.

B16.2 CRITICAL: All prompts are formatted as explicit SystemMessage + HumanMessage
objects, never raw strings. Raw strings get coerced into consecutive HumanMessages
by LangChain, causing hallucinations.
"""

from __future__ import annotations

import logging
from typing import Any, List, Sequence

from deepsearch.agent.resilience import retry_with_backoff
from deepsearch.config import DeepSearchSettings, LLMBackendConfig

try:  # pragma: no cover
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ImportError:  # pragma: no cover

    class _BaseMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    class AIMessage(_BaseMessage):  # type: ignore[no-redef]
        pass

    class HumanMessage(_BaseMessage):  # type: ignore[no-redef]
        pass

    class SystemMessage(_BaseMessage):  # type: ignore[no-redef]
        pass


logger = logging.getLogger(__name__)


def create_model(config: LLMBackendConfig) -> Any:
    """Instantiate the configured chat model backend.

    Args:
        config: LLM backend configuration (backend, model, api_key, etc.)

    Returns:
        A chat model instance with ainvoke() support.

    Raises:
        RuntimeError: If backend is unsupported or dependencies are missing.
    """
    if config.backend == "ollama":
        try:  # pragma: no cover
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("langchain-ollama is required for the Ollama backend") from exc
        return ChatOllama(
            model=config.model,
            temperature=config.temperature,
            base_url=config.base_url,
            num_predict=config.max_output_tokens,
        )

    if config.backend in ("deepseek", "minimax", "openai"):
        try:  # pragma: no cover
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "langchain-openai is required for the DeepSeek/MiniMax backend"
            ) from exc
        from deepsearch.llm.runtime import provider_async_http_client, provider_http_client

        parameters: dict[str, Any] = {}
        extra_body = dict(config.extra_body or {})
        if config.backend == "deepseek":
            # langchain-openai 0.3.35 renames max_tokens for every model.
            # extra_body is merged by the SDK after that translation.
            extra_body["max_tokens"] = config.max_output_tokens
        else:
            parameters["max_completion_tokens"] = config.max_output_tokens
        return ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            **parameters,
            reasoning_effort=config.reasoning_effort,
            extra_body=extra_body or None,
            use_responses_api=False,
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout,
            max_retries=0,
            http_client=provider_http_client(config.request_timeout),
            http_async_client=provider_async_http_client(config.request_timeout),
        )


    if config.backend == "gemini":
        # Gemini 2.5 Pro is a thinking model — use google-genai directly
        # so we can cap thinking budget and guarantee extractable output.
        if str(config.model).strip().lower() == "gemini-2.5-pro":
            from deepsearch.llm.gemini_genai import GeminiGenAIChatModel

            if not config.api_key:
                raise RuntimeError("GEMINI_API_KEY must be configured for the Gemini backend")

            thinking_budget = config.thinking_budget
            if thinking_budget is None:
                thinking_budget = 16384

            return GeminiGenAIChatModel(
                api_key=config.api_key,
                model=config.model,
                temperature=config.temperature if config.temperature is not None else 0.0,
                max_output_tokens=config.max_output_tokens,
                thinking_budget=thinking_budget,
            )

        if not config.api_key:
            raise RuntimeError("GEMINI_API_KEY must be configured for the Gemini backend")
        try:  # pragma: no cover
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("langchain-google-genai is required for the Gemini backend") from exc
        return ChatGoogleGenerativeAI(
            model=config.model,
            temperature=config.temperature if config.temperature is not None else 0.0,
            max_output_tokens=config.max_output_tokens,
            api_key=config.api_key,
        )

    raise RuntimeError(f"Unsupported backend: {config.backend!r}")


async def call_llm(
    model: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    settings: DeepSearchSettings | None = None,
) -> str:
    """Call the LLM with properly formatted messages.

    B16.2 CRITICAL: Prompts are always formatted as SystemMessage + HumanMessage
    objects, never raw strings.

    If the response is empty or minimal (< 100 chars), retries with a rephrased
    prompt to recover rather than blindly retrying the same input.

    Args:
        model: Chat model instance with ainvoke() support.
        system_prompt: System prompt text.
        user_prompt: User/human prompt text.
        settings: Optional settings for retry configuration.

    Returns:
        Response text as a string.
    """
    response = await call_llm_message(
        model,
        system_prompt,
        user_prompt,
        settings=settings,
    )
    return coerce_content(response)


async def call_llm_message(
    model: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    settings: DeepSearchSettings | None = None,
) -> Any:
    """Call the LLM and return the raw response message after recovery logic."""

    messages: List[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    async def _invoke(prompt_messages: Sequence[Any]) -> Any:
        async def _raw_invoke() -> Any:
            return await _raw_call(model, prompt_messages)

        if settings:
            wrapped = retry_with_backoff(
                max_retries=settings.llm_max_retries,
                base_delay=settings.llm_backoff_base,
                max_delay=settings.llm_backoff_max,
                jitter_ms=settings.llm_backoff_jitter_ms,
            )(_raw_invoke)
            return await wrapped()
        return await _raw_invoke()

    response = await _invoke(messages)
    text = coerce_content(response)

    # Handle empty/minimal response with prompt rephrasing
    if len(text.strip()) < 100:
        logger.warning(
            "LLM returned minimal response (%d chars) — retrying with rephrased prompt",
            len(text.strip()),
        )
        rephrased_messages: List[Any] = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    "Please provide detailed research findings with specific evidence "
                    "and citations. " + user_prompt
                )
            ),
        ]
        retry_response = await _invoke(rephrased_messages)
        retry_text = coerce_content(retry_response)
        if len(retry_text.strip()) > len(text.strip()):
            return retry_response

    return response


async def invoke_llm_raw(
    model: Any,
    system_prompt: str,
    user_prompt: str,
) -> Any:
    """Call LLM and return the raw response (AIMessage or similar).

    Unlike call_llm(), does NOT coerce to string and does NOT retry on
    empty responses.  Use this when the caller needs to inspect the raw
    message object (e.g. for JSON parsing or usage metadata).
    """
    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    return await _raw_call(model, messages)


async def _raw_call(model: Any, messages: Sequence[Any]) -> Any:
    """Raw LLM invocation with async support check."""
    if not hasattr(model, "ainvoke"):
        raise RuntimeError("Configured LLM backend does not support async invocation")
    return await model.ainvoke(messages)


def coerce_content(response: Any) -> str:
    """Normalize an LLM response payload to plain text.

    Gemini + LangChain integrations may return message content as structured
    blocks (e.g., ``[{"type": "text", "text": "..."}]``). Downstream parsers
    expect plain text, so we normalize here.
    """
    if response is None:
        return ""

    # AIMessage or similar with .content
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        # content might be a list of blocks
        return coerce_content(content)

    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        for key in ("content", "text"):
            val = response.get(key)
            if isinstance(val, str):
                return val

    if isinstance(response, (list, tuple)):
        chunks: list[str] = []
        for item in response:
            if item is None:
                continue
            if isinstance(item, str):
                if item.strip():
                    chunks.append(item)
                continue
            if isinstance(item, dict):
                for key in ("text", "content"):
                    val = item.get(key)
                    if isinstance(val, str) and val.strip():
                        chunks.append(val)
                        break
                else:
                    rendered = str(item).strip()
                    if rendered:
                        chunks.append(rendered)
                continue
            rendered = str(item).strip()
            if rendered:
                chunks.append(rendered)
        return "\n".join(chunks)

    return str(response)
