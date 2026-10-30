"""A provider outage fails fast, retryably and with its real cause.

On 2026-10-15 an overloaded DeepSeek held each JSPACE-1 request open for 15
minutes with keep-alive lines, defeating the 300s per-read timeout, then
answered HTTP 200 with an error object instead of a completion. The compiler
called that invalid output and stopped retrying; synthesis recorded it as a
returned (paid) response and suppressed its zero-token retry. The mission
blocked after 45 minutes.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import openai
import pytest
from pydantic import BaseModel

from deepsearch.llm.runtime import (
    ProviderResponseError,
    provider_async_http_client,
    provider_http_client,
    request_failure,
)

COMPLETION = {
    "id": "c1",
    "object": "chat.completion",
    "created": 0,
    "model": "deepseek-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": '{"answer": "ok"}'},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}
ERROR_OBJECT = {"error": {"message": "SECRET overload detail", "type": "server_error"}}
JSON = {"content-type": "application/json"}


class KeepAlive(httpx.SyncByteStream, httpx.AsyncByteStream):
    """DeepSeek's non-streaming overload behaviour: blank lines, then a body."""

    def __init__(self, lines: int, body: dict) -> None:
        self.lines = lines
        self.body = json.dumps(body).encode()

    def __iter__(self):
        for _ in range(self.lines):
            time.sleep(0.02)
            yield b"\n"
        yield self.body

    async def __aiter__(self):
        for _ in range(self.lines):
            await asyncio.sleep(0.02)
            yield b"\n"
        yield self.body


def sdk(budget: float, lines: int, body: dict) -> openai.OpenAI:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers=JSON, stream=KeepAlive(lines, body))
    )
    return openai.OpenAI(
        api_key="test", base_url="http://provider.test", max_retries=0,
        http_client=provider_http_client(budget, transport=transport),
    )


def create(client: openai.OpenAI):
    return client.chat.completions.create(
        model="deepseek-flash", messages=[{"role": "user", "content": "SECRET"}]
    )


def test_keep_alive_lines_cannot_hold_a_call_past_its_total_budget() -> None:
    started = time.monotonic()
    with pytest.raises(openai.APITimeoutError) as caught:
        create(sdk(0.1, lines=500, body=COMPLETION))  # ~10s if unbounded

    assert time.monotonic() - started < 2
    failure = request_failure(caught.value, model="deepseek-flash", stage="compiler")
    assert failure["category"] == "timeout" and failure["retryable"]


def test_keep_alive_lines_before_a_timely_answer_still_return_it() -> None:
    # Keep-alives also precede ordinary answers; the paid answer must survive.
    response = create(sdk(5, lines=3, body=COMPLETION))

    assert response.choices[0].message.content == '{"answer": "ok"}'


def test_async_client_enforces_the_same_total_budget() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=JSON, stream=KeepAlive(500, COMPLETION))

    client = openai.AsyncOpenAI(
        api_key="test", base_url="http://provider.test", max_retries=0,
        http_client=provider_async_http_client(0.1, transport=httpx.MockTransport(handler)),
    )

    async def call():
        return await client.chat.completions.create(model="deepseek-flash", messages=[])

    with pytest.raises(openai.APITimeoutError):
        asyncio.run(call())


def test_synthesis_records_an_error_object_as_a_failed_retryable_server_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deepsearch.llm.runtime as runtime
    from deepsearch.agent import codeagent_runner as runner

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers=JSON, stream=KeepAlive(2, ERROR_OBJECT))
    )
    bounded = runtime.provider_http_client
    monkeypatch.setattr(
        runtime, "provider_http_client", lambda seconds: bounded(seconds, transport=transport)
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    model = runner.build_default_model()

    with pytest.raises(openai.APIConnectionError) as caught:
        model.generate(messages=[{"role": "user", "content": "SECRET"}])

    assert isinstance(caught.value.__cause__, ProviderResponseError)
    # A "returned" record counts as paid progress and suppresses the retry.
    [attempt] = model.provider_attempts
    assert attempt["status"] == "failed" and not attempt["usage"]["total"]
    failure = attempt["failure"]
    assert failure["category"] == "server" and failure["retryable"]
    assert failure["exception_class"] == "ProviderResponseError"
    assert failure["http_status"] is None
    assert "SECRET" not in json.dumps(model.provider_attempts)


def test_compiler_retries_an_error_object_instead_of_calling_it_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deepsearch.llm.runtime as runtime
    from deepsearch.agent.llm_client import create_model
    from deepsearch.config import LLMBackendConfig
    from deepsearch.llm.structured_extractor import JSONInPromptStructuredExtractor

    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = ERROR_OBJECT if len(requests) == 1 else COMPLETION
        return httpx.Response(200, headers=JSON, json=body)

    bounded = runtime.provider_async_http_client
    monkeypatch.setattr(
        runtime, "provider_async_http_client",
        lambda seconds: bounded(seconds, transport=httpx.MockTransport(handler)),
    )
    model = create_model(LLMBackendConfig(
        backend="deepseek", model="deepseek-flash", temperature=None,
        max_output_tokens=100, api_key="test", base_url="http://provider.test",
    ))

    class Answer(BaseModel):
        answer: str

    extractor = JSONInPromptStructuredExtractor(chat_model=model)
    result = asyncio.run(extractor.extract(system_instruction="", prompt="q", schema=Answer))

    assert result == Answer(answer="ok") and len(requests) == 2
    failure = next(d for d in extractor.diagnostics if d.get("attempt") == 1 and "retryable" in d)
    assert failure["category"] == "server" and failure["retryable"]
