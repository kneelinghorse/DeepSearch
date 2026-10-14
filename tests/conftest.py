"""Repo-wide pytest fixtures.

Sprint 54: with the LLM-first entity extractor as the default contract-compile
path, tests that compile a contract would otherwise make a real Gemini call.
The autouse fixture below stubs every LLM-touching symbol that
``compile_contract_from_state`` reaches into so contract compilation
deterministically falls back to the regex extractor in the test environment.
Tests that need LLM behavior override the stubs explicitly via
``monkeypatch.setattr`` (see ``tests/unit/test_entity_extractor_llm.py`` and
``tests/integration/test_sprint_54_llm_entity_extraction.py``).

Sprint 76: extended to cover ``run_declared_entity_disambiguation_sync`` —
``compile_contract_from_state`` calls this whenever ``required_entities`` is
non-empty, and without the stub it hit live Gemini, producing 30+ second
non-deterministic test runs (``test_compile_contract_empty_authored_fields_preserve_legacy_output``
was the visible tip; tests/ root spent ~14 min mostly on this path).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _stub_llm_entity_extractor_default(monkeypatch):
    """Default-stub the LLM entity extractor + disambiguator to None.

    Tests that want a specific LLM outcome re-patch the symbol after this
    fixture runs (monkeypatch is per-function, so the override wins).
    """

    try:
        from deepsearch.mission import entity_extractor_llm
    except Exception:  # pragma: no cover - module always importable in normal envs
        return

    monkeypatch.setattr(
        entity_extractor_llm,
        "run_llm_entity_extraction_sync",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        entity_extractor_llm,
        "run_declared_entity_disambiguation_sync",
        lambda *args, **kwargs: None,
    )
    yield


@pytest.fixture(autouse=True)
def _offline_provider_defaults(monkeypatch):
    """Offline tests must not depend on private dotenv files or leaked settings.

    Existing extractor/transport fixtures still control calls. These are only
    configuration defaults; individual tests override or remove them when they
    exercise provider validation. Opt-in Gemini tests use their explicit client.
    """
    monkeypatch.setattr("deepsearch.config.load_dotenv", lambda: None)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("LLM_BACKEND", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unpaid-offline-key")
