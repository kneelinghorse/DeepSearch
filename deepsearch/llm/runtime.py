"""One explicit request contract for the active research pipeline.

Limits verified against official provider documentation on 2026-09-19.
Legacy development adapters remain in config.py; they are not worker targets.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterator, Literal

import httpx

if TYPE_CHECKING:
    from deepsearch.config import DeepSearchSettings


class ModelConfigurationError(ValueError):
    """Invalid provider configuration, diagnosed before paid research."""


class ProviderResponseError(RuntimeError):
    """HTTP 200 whose body is a provider error object instead of a completion."""


_TOTAL_DEADLINE = "deepsearch_total_deadline"


class _TotalDeadlineStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """Fail a response body still arriving after the request's total budget.

    httpx timeouts bound each read, and provider keep-alive lines reset them:
    on 2026-10-15 overloaded DeepSeek held calls for 15 minutes under a 300s
    timeout before answering with an error object.
    """

    def __init__(self, stream: Any, deadline: float, request: httpx.Request) -> None:
        self._stream = stream
        self._deadline = deadline
        self._request = request

    def _check(self) -> None:
        if time.monotonic() > self._deadline:
            raise httpx.ReadTimeout(
                "provider request exceeded its total time budget", request=self._request
            )

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._stream:
            self._check()
            yield chunk

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            self._check()
            yield chunk

    def close(self) -> None:
        self._stream.close()

    async def aclose(self) -> None:
        await self._stream.aclose()


def _start_budget(total_seconds: float) -> Callable[[httpx.Request], None]:
    def start(request: httpx.Request) -> None:
        request.extensions[_TOTAL_DEADLINE] = time.monotonic() + total_seconds

    return start


def _bound_body(response: httpx.Response) -> bool:
    """Bound the body's total time; return whether it is a completion to inspect."""
    response.stream = _TotalDeadlineStream(
        response.stream, response.request.extensions[_TOTAL_DEADLINE], response.request
    )
    return (
        response.status_code == 200
        and "text/event-stream" not in response.headers.get("content-type", "")
    )


def _reject_error_object(response: httpx.Response) -> None:
    try:
        body = json.loads(response.content)
    except ValueError:
        return  # Not JSON: the SDK reports it.
    if isinstance(body, dict) and body.get("error") and not body.get("choices"):
        raise ProviderResponseError("provider returned an error object with HTTP 200")


def provider_http_client(total_seconds: float, **kwargs: Any) -> httpx.Client:
    """Return the SDK's default client, bounded in total time and rejecting error bodies."""
    from openai import DefaultHttpxClient

    def bound(response: httpx.Response) -> None:
        if _bound_body(response):
            response.read()
            _reject_error_object(response)

    return DefaultHttpxClient(
        event_hooks={"request": [_start_budget(total_seconds)], "response": [bound]},
        **kwargs,
    )


def provider_async_http_client(total_seconds: float, **kwargs: Any) -> httpx.AsyncClient:
    """Async twin of :func:`provider_http_client`."""
    from openai import DefaultAsyncHttpxClient

    start = _start_budget(total_seconds)

    async def start_async(request: httpx.Request) -> None:
        start(request)

    async def bound(response: httpx.Response) -> None:
        if _bound_body(response):
            await response.aread()
            _reject_error_object(response)

    return DefaultAsyncHttpxClient(
        event_hooks={"request": [start_async], "response": [bound]}, **kwargs
    )


@dataclass(frozen=True, slots=True)
class EffectiveModelConfig:
    """Validated provider identity, credentials and request parameters."""

    provider: Literal["deepseek", "openai"]
    model: str
    base_url: str
    credential_source: str
    api_key: str = field(repr=False)
    max_output_tokens: int
    context_window: int
    reasoning_effort: str
    request_timeout: float

    def request_parameters(self) -> dict[str, Any]:
        """Return parameters accepted by the selected Chat Completions API."""
        if self.provider == "openai":
            return {
                "max_completion_tokens": self.max_output_tokens,
                "reasoning_effort": self.reasoning_effort,
            }
        return {
            "max_tokens": self.max_output_tokens,
            "reasoning_effort": self.reasoning_effort,
            "extra_body": {
                "thinking": {
                    "type": "disabled" if self.reasoning_effort == "none" else "enabled"
                }
            },
        }

    def public_metadata(self) -> dict[str, Any]:
        """Return content-free configuration safe for health and artifacts."""
        return {
            "provider": self.provider,
            "requested_model": self.model,
            "base_url": self.base_url,
            "credential_source": self.credential_source,
            "max_output_tokens": self.max_output_tokens,
            "context_window": self.context_window,
            "reasoning_effort": self.reasoning_effort,
            "temperature": "omitted",
            "request_timeout_seconds": self.request_timeout,
            "stage_overrides": {},
        }


def resolve_effective_model(settings: DeepSearchSettings) -> EffectiveModelConfig:
    """Resolve only exact supported identities; never redirect credentials."""
    raw_provider = settings.llm_backend
    provider: Literal["deepseek", "openai"]
    if raw_provider == "deepseek":
        provider = "deepseek"
        model = settings.deepseek_model
        endpoint = settings.deepseek_base_url.rstrip("/")
        credential_source = "DEEPSEEK_API_KEY"
        key = settings.deepseek_api_key
        cap = settings.deepseek_max_output_tokens
        # DeepSeek documents 1M; use the conservative decimal interpretation.
        context, output_limit = 1_000_000, 393_216
        effort = settings.llm_reasoning_effort or "high"
        allowed_efforts = {"none", "low", "high", "max"}
        if model not in {"deepseek-flash", "deepseek-v4-pro"}:
            raise ModelConfigurationError(
                "DEEPSEEK_MODEL must be deepseek-flash or deepseek-v4-pro"
            )
        if endpoint not in {"https://api.deepseek.com", "https://api.deepseek.com/v1"}:
            raise ModelConfigurationError(
                "DEEPSEEK_BASE_URL must be the official DeepSeek endpoint"
            )
    elif raw_provider == "openai":
        provider = "openai"
        model = settings.openai_model
        endpoint = settings.openai_base_url.rstrip("/")
        credential_source = "OPENAI_API_KEY"
        key = settings.openai_api_key
        cap = settings.openai_max_output_tokens
        context, output_limit = 1_050_000, 128_000
        effort = settings.llm_reasoning_effort or "medium"
        allowed_efforts = {"none", "low", "medium", "high", "xhigh", "max"}
        if model != "gpt-5.6-luna":
            raise ModelConfigurationError("OPENAI_MODEL must be gpt-5.6-luna")
        if endpoint != "https://api.openai.com/v1":
            raise ModelConfigurationError("OPENAI_BASE_URL must be the official OpenAI endpoint")
    else:
        raise ModelConfigurationError("Production LLM_BACKEND must be deepseek or openai")
    if not isinstance(cap, int) or isinstance(cap, bool) or not 1 <= cap <= output_limit:
        raise ModelConfigurationError(
            "Selected provider MAX_TOKENS is invalid or exceeds its limit"
        )
    if effort not in allowed_efforts:
        raise ModelConfigurationError(
            "LLM_REASONING_EFFORT is unsupported by the selected provider"
        )
    timeout = settings.llm_request_timeout_seconds
    if not math.isfinite(timeout) or timeout <= 0:
        raise ModelConfigurationError("LLM_REQUEST_TIMEOUT_SECONDS must be finite and positive")
    if not key or not key.strip() or any(char.isspace() for char in key):
        raise ModelConfigurationError(f"{credential_source} missing or structurally invalid")
    return EffectiveModelConfig(
        provider=provider, model=model, base_url=endpoint, credential_source=credential_source,
        api_key=key, max_output_tokens=cap, context_window=context,
        reasoning_effort=effort, request_timeout=timeout,
    )


def _rejection_details(exc: BaseException) -> dict[str, Any]:
    """Classify untrusted SDK fields into fixed labels; never retain their text."""
    code = getattr(exc, "code", None)
    error_type = getattr(exc, "type", None)
    by_code = {
        "context_length_exceeded": "context_limit",
        "unsupported_parameter": "unsupported_parameter",
        "unsupported_value": "unsupported_value",
        "content_filter": "content_rejection",
        "content_policy_violation": "content_rejection",
        "invalid_json": "invalid_json",
        "invalid_message": "invalid_message_structure",
    }
    subtype, source = "unknown", "unknown"
    for label, value in (("code", code), ("type", error_type)):
        if isinstance(value, str) and value in by_code:
            subtype, source = by_code[value], label
            break
    if subtype == "unknown":
        # Only explicit provider descriptions count. Conflicting descriptions
        # stay unknown; matching a phrase is diagnostic evidence, not proof.
        message = getattr(exc, "message", None)
        if isinstance(message, str):
            message = message[:8192].lower()
            patterns = {
                "context_limit": (
                    r"context (?:length|window).{0,40}(?:exceed|limit)|maximum context length"
                ),
                "unsupported_parameter": (
                    r"unsupported parameter|parameter.{0,60}(?:not supported|unsupported)"
                ),
                "invalid_message_structure": (
                    r"invalid message (?:format|structure)|messages must|invalid role|roles must"
                ),
                "invalid_json": r"invalid json|json (?:decode|parse|parsing) error",
                "content_rejection": r"content policy|content filter|content safety",
            }
            matches = [name for name, pattern in patterns.items() if re.search(pattern, message)]
            if len(matches) == 1:
                subtype, source = matches[0], "message_pattern"
    raw_param = getattr(exc, "param", None)
    parameter = "unknown"
    if isinstance(raw_param, str):
        if raw_param in {
            "model", "messages", "tools", "tool_choice", "response_format",
            "max_tokens", "max_completion_tokens", "reasoning_effort", "temperature",
        }:
            parameter = raw_param
        elif re.fullmatch(
            r"messages\[\d{1,6}\]\.(content|role|tool_calls|tool_call_id)", raw_param
        ):
            parameter = "messages." + raw_param.rsplit(".", 1)[1]
    return {
        "version": 1, "subtype": subtype, "subtype_source": source,
        "parameter": parameter,
        "provider_type": error_type if isinstance(error_type, str) and error_type in {
            "invalid_request_error", "authentication_error", "rate_limit_error", "server_error",
        } else "unknown",
    }


def request_failure(exc: BaseException, *, model: str, stage: str) -> dict[str, Any]:
    """Project a provider exception without its message, body, URL or headers."""
    from deepsearch.error_safety import safe_exception_details

    # The SDK reports an exception raised by the HTTP client as a connection error.
    if isinstance(exc.__cause__, ProviderResponseError):
        exc = exc.__cause__
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if not isinstance(status, int) or isinstance(status, bool) or not 100 <= status <= 599:
        status = None
    category = safe_exception_details(exc).category
    if status in {401, 403}:
        category = "authentication"
    elif status == 429:
        category = "rate_limit"
    elif status is not None and 400 <= status < 500 and status not in {408, 409}:
        category = "configuration"
    elif status is not None and status >= 500:
        category = "server"
    elif isinstance(exc, ModelConfigurationError):
        category = "configuration"
    elif isinstance(exc, ProviderResponseError):
        category = "server"
    # Unknown provider strings may include request content: only fixed codes survive.
    code = getattr(exc, "code", None)
    allowed_codes = {
        "invalid_api_key", "model_not_found", "unsupported_parameter",
        "unsupported_value", "context_length_exceeded", "insufficient_quota",
        "rate_limit_exceeded", "invalid_request_error",
    }
    retryable = status in {408, 409, 429} or (status is not None and status >= 500)
    retryable = retryable or (
        status is None and category in {"timeout", "transport", "rate_limit", "server"}
    )
    return {
        "stage": stage,
        "requested_model": (
            model if model in {"deepseek-flash", "deepseek-v4-pro", "gpt-5.6-luna"}
            else "unresolved"
        ),
        "category": category,
        "http_status": status,
        "code": code if isinstance(code, str) and code in allowed_codes else None,
        "exception_class": safe_exception_details(exc).exception_class,
        "retryable": retryable,
        "rejection": _rejection_details(exc),
    }
