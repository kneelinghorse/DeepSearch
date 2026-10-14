"""Unit tests for DeepSearchSettings environment normalization."""

from __future__ import annotations

import pytest

import deepsearch.config as config_mod
from deepsearch.config import DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS, DeepSearchSettings


def _isolate_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop ``load()`` from repopulating deleted vars out of the repo ``.env``.

    ``load_dotenv`` does not override vars that are already set, but it *does*
    fill in ones that are absent — so a ``delenv`` test would otherwise assert
    against whatever the developer's ``.env`` happens to hold and pass for the
    wrong reason on a machine that has one.
    """
    monkeypatch.setattr(config_mod, "load_dotenv", lambda *a, **k: False)


def test_load_normalizes_gemini_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed-case Gemini model names should be normalized for the Google API."""
    monkeypatch.setenv("LLM_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-Pro")

    settings = DeepSearchSettings.load()

    assert settings.gemini_model == "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# LLM_BACKEND resolution.
#
# WHY this matters: ``llm_backend`` selects the contract-compile structured
# extractor (``build_structured_extractor``). On 2026-08-13 the Gemini branch
# hung 18 minutes mid-compile and cost a production mission its lease. The env
# var was corrected on Railway afterwards, but an unset-or-typo'd value still
# resolved back to Gemini in code — so the outage was one lost variable away
# from returning. These tests pin that it cannot.
# ---------------------------------------------------------------------------


def test_load_defaults_to_deepseek_when_backend_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset LLM_BACKEND must land on the production backend, not Gemini."""
    _isolate_dotenv(monkeypatch)
    monkeypatch.delenv("LLM_BACKEND", raising=False)

    assert DeepSearchSettings.load().llm_backend == "deepseek"


def test_load_preserves_unknown_backend_for_preflight(monkeypatch) -> None:
    """A typo cannot select another provider or credential; preflight rejects it."""
    from deepsearch.llm.runtime import ModelConfigurationError, resolve_effective_model

    _isolate_dotenv(monkeypatch)
    monkeypatch.setenv("LLM_BACKEND", "gemni")
    settings = DeepSearchSettings.load()
    assert settings.llm_backend == "gemni"
    with pytest.raises(ModelConfigurationError):
        resolve_effective_model(settings)


def test_load_still_honors_an_explicit_gemini_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking for Gemini on purpose still works — only the silent path is closed."""
    _isolate_dotenv(monkeypatch)
    monkeypatch.setenv("LLM_BACKEND", "GEMINI")

    assert DeepSearchSettings.load().llm_backend == "gemini"


# ---------------------------------------------------------------------------
# LLM request timeout.
# ---------------------------------------------------------------------------


def test_llm_request_timeout_defaults_and_reads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The timeout is configurable but must never be absent."""
    _isolate_dotenv(monkeypatch)
    monkeypatch.delenv("LLM_REQUEST_TIMEOUT_SECONDS", raising=False)
    assert (
        DeepSearchSettings.load().llm_request_timeout_seconds
        == DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS
    )

    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "90.5")
    assert DeepSearchSettings.load().llm_request_timeout_seconds == 90.5


def test_llm_request_timeout_reaches_the_backend_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_llm_config must carry the timeout to whichever backend is selected."""
    _isolate_dotenv(monkeypatch)
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "123")
    # Building the selected provider config validates its credential boundary.
    # Use an inert test value so this unit test is independent of local .env.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline")

    settings = DeepSearchSettings.load()

    assert settings.build_llm_config().request_timeout == 123.0
    assert settings.build_reflection_llm_config().request_timeout == 123.0
