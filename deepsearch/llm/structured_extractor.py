"""Provider-portable structured-output extractor.

Sprint 54 L1. This module defines a tight abstraction for "give me a
Pydantic-validated object back from an LLM call." Two strategies ship:

1. ``GeminiStructuredExtractor`` — uses Gemini's grammar-constrained mode
   (``response_mime_type="application/json"`` + ``response_schema=<Pydantic>``).
   The model literally cannot emit malformed JSON; the happy path needs no
   retry. Schema-mode caveat: Gemini's ``response_schema`` does NOT accept
   full JSON Schema. ``$ref`` is unsupported, ``oneOf`` is unsupported, and
   ``anyOf`` is only loosely supported. Stick to flat schemas with required
   primitive fields and lists of objects whose nested schemas are also flat.

2. ``JSONInPromptStructuredExtractor`` — universal fallback that wraps any
   LangChain ``BaseChatModel`` (DeepSeek via ``langchain_openai``, MiniMax,
   Anthropic). It appends the Pydantic JSON schema to the system prompt,
   forces ``temperature=0``, parses the response, validates against the
   schema, and retries up to 2× on malformed JSON or schema-validation
   failure. After three attempts it returns ``None``.

The dispatch helper ``build_structured_extractor`` lives in
``deepsearch.config`` and selects the right strategy from
``LLMBackendConfig.backend``.

Architecture guardrail: this module MUST NOT import from
``deepsearch.mission`` or any contract/entity module. The port knows about
LLM calls and Pydantic schemas, nothing about entities or contracts. Higher
layers (e.g. ``deepsearch.mission.entity_extractor_llm``) consume this
abstraction.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=BaseModel)


class StructuredExtractor(Protocol):
    """Async port for "extract a Pydantic instance from an LLM call."""

    async def extract(
        self,
        *,
        system_instruction: str,
        prompt: str,
        schema: type[T],
    ) -> T | None:
        """Run the extraction.

        Returns ``None`` only after retry exhaustion or unrecoverable
        backend error. Callers decide whether to fall back, raise, or
        proceed without structured data.
        """


@dataclass(slots=True)
class GeminiStructuredExtractor:
    """Grammar-constrained Gemini extractor.

    Wraps ``GeminiGenAIChatModel`` with a per-call schema. Because the model
    cannot emit malformed JSON in this mode, no retry is necessary on the
    happy path. We still defensively re-validate the output against the
    caller's schema so that downstream code is type-safe even if a future
    SDK regression returns lax output.
    """

    api_key: str
    model: str
    temperature: float = 0.0
    max_output_tokens: int = 8000
    thinking_budget: int | None = None

    async def extract(
        self,
        *,
        system_instruction: str,
        prompt: str,
        schema: type[T],
    ) -> T | None:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:  # pragma: no cover
            logger.error("langchain-core not installed; cannot run GeminiStructuredExtractor")
            return None

        from deepsearch.llm.gemini_genai import GeminiGenAIChatModel, GeminiGenAIError

        try:
            model = GeminiGenAIChatModel(
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                thinking_budget=self.thinking_budget,
                response_schema=schema,
            )
        except GeminiGenAIError:
            logger.exception("Failed to construct GeminiGenAIChatModel for structured extraction")
            return None

        messages: list[Any] = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction))
        messages.append(HumanMessage(content=prompt))

        try:
            response = await model.ainvoke(messages)
        except Exception:
            logger.exception("Gemini structured extraction call failed")
            return None

        text = _coerce_to_text(response)
        if not text:
            logger.warning("Gemini structured extractor returned empty content")
            return None

        try:
            return schema.model_validate_json(text)
        except ValidationError:
            logger.warning(
                "Gemini structured extractor output failed schema validation: %s",
                _truncate_for_log(text),
            )
            return None
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "Unexpected error parsing Gemini structured output: %s",
                _truncate_for_log(text),
            )
            return None


@dataclass(slots=True)
class JSONInPromptStructuredExtractor:
    """Universal JSON-in-prompt extractor for any LangChain ChatModel.

    Strategy: append the Pydantic JSON schema to the system prompt with a
    strict "respond with raw JSON, no markdown, no prose" instruction; force
    ``temperature=0`` for determinism; run, strip common wrapping (markdown
    code fences, leading prose), validate. On ``ValidationError`` or JSON
    decode failure, retry up to 2× (3 attempts total) with a corrective
    prefix on the user message.
    """

    chat_model: Any
    max_retries: int = 2
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    effective_config: dict[str, Any] = field(default_factory=dict)

    async def extract(
        self,
        *,
        system_instruction: str,
        prompt: str,
        schema: type[T],
    ) -> T | None:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError:  # pragma: no cover
            logger.error("langchain-core not installed; cannot run JSONInPromptStructuredExtractor")
            return None

        schema_text = _render_schema(schema)
        full_system = (
            (system_instruction.strip() + "\n\n" if system_instruction.strip() else "")
            + "Respond with a JSON object matching this schema "
            + "(do not include markdown fences, do not include any prose):\n"
            + schema_text
        )

        # Force deterministic decoding for structured output.
        try:
            self._set_temperature_zero()
        except Exception:  # pragma: no cover - defensive
            logger.debug("Could not force temperature=0 on chat_model; continuing.")

        last_error_hint = ""
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            from deepsearch.agent.execution_control import (
                ExecutionCancelledError,
                raise_if_execution_cancelled,
            )
            from deepsearch.llm.runtime import request_failure

            raise_if_execution_cancelled()
            user_content = prompt
            if last_error_hint:
                user_content = (
                    "The previous response was malformed JSON or did not match "
                    "the schema. Try again.\n\n" + prompt
                )

            messages: list[Any] = [
                SystemMessage(content=full_system),
                HumanMessage(content=user_content),
            ]

            from deepsearch.llm.usage import usage_event

            started = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()
            response = None
            call_error: BaseException | None = None
            try:
                response = await self.chat_model.ainvoke(messages)
            except ExecutionCancelledError as exc:
                call_error = exc
                raise
            except Exception as exc:
                call_error = exc
                failure = request_failure(
                    exc, model=str(getattr(self.chat_model, "model_name", "")),
                    stage="compiler",
                )
                self.diagnostics.append({**failure, "attempt": attempt + 1})
                logger.warning("Compiler provider request failed: %s", failure)
                if not failure["retryable"]:
                    return None
                last_error_hint = "call_failed"
                continue
            finally:
                self.diagnostics.append(usage_event(
                    response, config=self.effective_config, stage="compiler",
                    started_at=started, elapsed=time.monotonic() - t0,
                    error=call_error, attempt=attempt + 1,
                ))

            metadata = getattr(response, "response_metadata", {})
            finish = metadata.get("finish_reason") if isinstance(metadata, dict) else None
            if finish in {"length", "content_filter", "tool_calls"}:
                self.diagnostics.append({
                    "stage": "compiler", "category": "incomplete_response",
                    "requested_model": str(getattr(self.chat_model, "model_name", "")),
                    "finish_reason": finish, "attempt": attempt + 1,
                })
                # Repeating an unchanged output budget cannot repair truncation.
                return None

            text = _coerce_to_text(response)
            if not text:
                last_error_hint = "empty_response"
                continue

            cleaned = _strip_json_wrapping(text)
            try:
                return schema.model_validate_json(cleaned)
            except (ValidationError, json.JSONDecodeError):
                logger.info(
                    "JSONInPrompt attempt %d/%d failed schema validation",
                    attempt + 1,
                    attempts,
                )
                last_error_hint = "validation_failed"
                continue
            except Exception:  # pragma: no cover - defensive
                logger.warning("Unexpected error parsing JSONInPrompt output")
                last_error_hint = "parse_error"
                continue

        return None

    def _set_temperature_zero(self) -> None:
        for attr in ("temperature", "model_kwargs"):
            if not hasattr(self.chat_model, attr):
                continue
            current = getattr(self.chat_model, attr)
            if attr == "temperature":
                if current is None:
                    continue  # Effective provider contract deliberately omits temperature.
                try:
                    setattr(self.chat_model, attr, 0.0)
                except Exception:
                    pass
            elif isinstance(current, dict) and "temperature" in current:
                current["temperature"] = 0.0


_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)
_LEADING_PROSE_PATTERN = re.compile(r"^[^\{\[]*?(?P<json>[\{\[].*[\}\]])\s*$", re.DOTALL)


def _strip_json_wrapping(text: str) -> str:
    """Defensively strip markdown fences and leading prose from JSON output."""

    stripped = (text or "").strip()
    if not stripped:
        return ""
    fence = _FENCE_PATTERN.match(stripped)
    if fence:
        stripped = fence.group(1).strip()
    if not stripped:
        return ""
    if stripped[0] not in "[{":
        match = _LEADING_PROSE_PATTERN.match(stripped)
        if match:
            stripped = match.group("json").strip()
    return stripped


def _coerce_to_text(response: Any) -> str:
    """Best-effort string coercion for chat-model responses."""

    if response is None:
        return ""
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    chunks.append(value)
        return "".join(chunks).strip()
    if isinstance(response, str):
        return response.strip()
    return str(response).strip()


def _render_schema(schema: type[BaseModel]) -> str:
    """Render a Pydantic model's JSON schema for inclusion in the prompt."""

    try:
        spec = schema.model_json_schema()
    except Exception:  # pragma: no cover - defensive
        return ""
    return json.dumps(spec, indent=2, sort_keys=True)


def _truncate_for_log(text: str, *, limit: int = 240) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "..."


__all__ = [
    "GeminiStructuredExtractor",
    "JSONInPromptStructuredExtractor",
    "StructuredExtractor",
]
