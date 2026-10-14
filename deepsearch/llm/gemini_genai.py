"""Gemini chat adapter using the `google-genai` SDK.

Why this exists:
Gemini 2.5 Pro is a "thinking" model where the thinking-token budget can consume
the entire `max_output_tokens` allocation. When that happens, some client
wrappers can return an empty text payload, which breaks downstream parsing
(e.g. research findings extraction) and can lead to non-terminating graphs.

This adapter uses `google-genai` directly so we can:
1) Set an explicit thinking budget that leaves room for actual output text.
2) Extract response text reliably from candidate parts.
3) Attach token usage metadata for DeepSearch telemetry.
4) Detect and log empty responses for debugging synthesis issues (B11.4).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised in runtime environments with google-genai installed
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

try:  # pragma: no cover - optional in some unit-test environments
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
except ImportError:  # pragma: no cover
    AIMessage = None  # type: ignore[assignment]
    BaseMessage = object  # type: ignore[assignment]
    HumanMessage = object  # type: ignore[assignment]
    SystemMessage = object  # type: ignore[assignment]


class GeminiGenAIError(RuntimeError):
    """Raised when the google-genai Gemini adapter cannot be used."""


@dataclass(slots=True)
class GeminiGenAIChatModel:
    """Async chat model wrapper for Gemini via `google-genai`.

    This implements the minimal subset DeepSearch needs: an `ainvoke(messages)`
    method that returns a LangChain `AIMessage`.

    When ``response_schema`` is set, the adapter switches to
    grammar-constrained JSON output (``response_mime_type="application/json"``).
    The Pydantic schema is passed through to ``GenerateContentConfig`` and the
    parsed instance is read from ``response.parsed`` when available; the
    JSON-text fallback at ``_extract_text`` covers SDK builds where
    ``parsed`` is not exposed.
    """

    api_key: str
    model: str
    temperature: float
    max_output_tokens: int
    thinking_budget: int | None = None
    response_schema: Any = None
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if genai is None or genai_types is None:
            raise GeminiGenAIError(
                "google-genai is not installed; install `google-genai` to use GeminiGenAIChatModel."
            )
        if AIMessage is None:
            raise GeminiGenAIError(
                "langchain-core is not installed; install `langchain-core` to use "
                "GeminiGenAIChatModel."
            )
        if not self.api_key:
            raise GeminiGenAIError("GEMINI_API_KEY must be configured for google-genai calls.")

        # Clamp basic numeric settings defensively.
        try:
            self.max_output_tokens = max(int(self.max_output_tokens), 1)
        except (TypeError, ValueError):
            self.max_output_tokens = 1
        try:
            self.temperature = float(self.temperature)
        except (TypeError, ValueError):
            self.temperature = 0.7

        if self.thinking_budget is not None:
            try:
                self.thinking_budget = int(self.thinking_budget)
            except (TypeError, ValueError):
                self.thinking_budget = None

        self._client = genai.Client(api_key=self.api_key)

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Invoke Gemini with the provided LangChain messages.

        B11.4: Added empty response detection with detailed logging to help
        diagnose synthesis issues.
        """
        system_instruction, contents = _convert_messages(messages)
        effective_budget = _safe_thinking_budget(self.thinking_budget, self.max_output_tokens)
        config = _build_config(
            system_instruction=system_instruction,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            thinking_budget=effective_budget,
            response_schema=self.response_schema,
        )

        loop = asyncio.get_event_loop()

        response = await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            ),
        )

        content = _extract_text(response)
        usage_metadata = _extract_usage_metadata(response)
        response_metadata = _extract_response_metadata(response)

        # B11.4: Detect and log empty responses for debugging
        if not content or len(content.strip()) < 10:
            logger.warning(
                "Empty or minimal response from %s (thinking_budget=%s, max_output=%d): "
                "content_length=%d, thoughts_tokens=%d, candidates_tokens=%d, finish_reason=%s",
                self.model,
                effective_budget,
                self.max_output_tokens,
                len(content) if content else 0,
                usage_metadata.get("thoughts_tokens", 0),
                usage_metadata.get("candidates_tokens", 0),
                response_metadata.get("finish_reason"),
            )

        return AIMessage(
            content=content,
            usage_metadata=usage_metadata,
            response_metadata=response_metadata,
        )


def _safe_thinking_budget(thinking_budget: int | None, max_output_tokens: int) -> int | None:
    """Return a sanitized thinking budget.

    NOTE: Gemini 2.5 Pro uses thinking_budget as a separate allocation from
    max_output_tokens. The thinking budget should NOT be capped against
    max_output_tokens - they are independent allocations.

    B11.4: Removed incorrect capping that was limiting thinking_budget to
    max_output_tokens - 512. This was causing synthesis issues when
    thinking_budget (e.g., 16384) exceeded max_output_tokens (e.g., 8000).

    Args:
        thinking_budget: Requested thinking budget, or None to disable.
        max_output_tokens: Maximum output tokens (unused, kept for API compatibility).

    Returns:
        Sanitized thinking budget, or None if disabled.
    """
    if thinking_budget is None:
        return None
    if thinking_budget == -1:
        return -1  # Unlimited thinking budget
    return max(1, int(thinking_budget))


def _build_config(
    *,
    system_instruction: str,
    temperature: float,
    max_output_tokens: int,
    thinking_budget: int | None,
    response_schema: Any = None,
) -> Any:
    if genai_types is None:  # pragma: no cover
        raise GeminiGenAIError("google-genai types are unavailable.")

    thinking_config = None
    if thinking_budget is not None:
        thinking_config = genai_types.ThinkingConfig(thinkingBudget=thinking_budget)

    kwargs: dict[str, Any] = {
        "system_instruction": system_instruction or None,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "thinking_config": thinking_config,
    }
    if response_schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = response_schema

    return genai_types.GenerateContentConfig(**kwargs)


def _convert_messages(messages: Sequence[BaseMessage]) -> tuple[str, list[Any]]:
    """Convert LangChain messages into google-genai `contents` + `system_instruction`."""

    if genai_types is None:  # pragma: no cover
        raise GeminiGenAIError("google-genai types are unavailable.")

    system_chunks: list[str] = []
    contents: list[Any] = []

    for message in messages:
        content = str(getattr(message, "content", "") or "")
        if isinstance(message, SystemMessage):
            if content.strip():
                system_chunks.append(content.strip())
            continue

        role = "user"
        if isinstance(message, HumanMessage):
            role = "user"
        elif message.__class__.__name__.lower().endswith("aimessage"):
            role = "model"

        if not content:
            content = ""

        contents.append(
            genai_types.Content(
                role=role,
                parts=[genai_types.Part(text=content)],
            )
        )

    return ("\n\n".join(system_chunks).strip(), contents)


def _extract_text(response: Any) -> str:
    """Extract concatenated text from a google-genai response.

    When the request was made with ``response_schema``, the SDK exposes a
    parsed Pydantic instance at ``response.parsed``. We serialize it back to
    JSON so callers (e.g. ``GeminiStructuredExtractor``) can re-validate
    against their own schema and obtain a typed instance. If ``parsed`` is
    absent, fall back to concatenating candidate text parts (which contain
    the raw JSON in schema mode).
    """

    parsed = getattr(response, "parsed", None)
    if parsed is not None and hasattr(parsed, "model_dump_json"):
        try:
            return str(parsed.model_dump_json()).strip()
        except Exception:  # pragma: no cover - defensive
            pass

    # Prefer candidate parts; response.text can be None for thinking models.
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts:
            text = "".join(getattr(part, "text", "") or "" for part in parts)
            return str(text or "").strip()

    text = getattr(response, "text", None)
    return str(text or "").strip()


def _extract_usage_metadata(response: Any) -> Mapping[str, int]:
    """Return usage metadata compatible with DeepSearch telemetry parsing."""

    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
    total_tokens = int(getattr(usage, "total_token_count", 0) or 0) if usage else 0
    thoughts_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0) if usage else 0
    candidates_tokens_raw = getattr(usage, "candidates_token_count", 0) if usage else 0
    candidates_tokens = int(candidates_tokens_raw or 0)

    completion_tokens = (
        max(total_tokens - prompt_tokens, 0)
        if total_tokens
        else candidates_tokens + thoughts_tokens
    )

    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "thoughts_tokens": thoughts_tokens,
        "candidates_tokens": candidates_tokens,
    }


def _extract_response_metadata(response: Any) -> Mapping[str, Any]:
    candidates = getattr(response, "candidates", None) or []
    finish_reason = None
    if candidates:
        finish_reason = getattr(candidates[0], "finish_reason", None)
    return {
        "finish_reason": str(finish_reason) if finish_reason is not None else None,
        "model": getattr(response, "model_version", None),
    }


__all__ = ["GeminiGenAIChatModel", "GeminiGenAIError"]
