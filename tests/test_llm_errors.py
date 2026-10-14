"""LLM retry and degradation regression tests."""

from __future__ import annotations

import pytest

from deepsearch.agent.resilience import retry_with_backoff


@pytest.mark.asyncio
async def test_retry_with_backoff_retries_async_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decorator should retry async callables before succeeding."""

    attempts = {"count": 0}

    async def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("not yet")
        return "ok"

    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("deepsearch.agent.resilience.asyncio.sleep", fast_sleep)

    wrapped = retry_with_backoff(max_retries=3)(flaky)
    result = await wrapped()

    assert result == "ok"
    assert attempts["count"] == 3


@pytest.mark.asyncio
async def test_retry_with_backoff_raises_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decorator should propagate the last error when retries exhausted."""

    attempts = {"count": 0}

    async def always_fail() -> None:
        attempts["count"] += 1
        raise RuntimeError("boom")

    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("deepsearch.agent.resilience.asyncio.sleep", fast_sleep)

    wrapped = retry_with_backoff(max_retries=2)(always_fail)
    with pytest.raises(RuntimeError):
        await wrapped()

    assert attempts["count"] == 3
