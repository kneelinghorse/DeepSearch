"""Unit tests for the S71 Path B DeepEvalBaseLLM adapter.

Pure-Python; the SDK clients (``openai``, ``google.genai``) are mocked
at the module-import seam inside :meth:`load_model`. Closes substrate
blocker (3) from S70 m02 calibration: ``FaithfulnessMetric(model=<str>)``
cannot resolve a plain string. The factory + adapters give it a
runnable ``DeepEvalBaseLLM`` so divergence becomes measurable.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

import deepsearch.eval.deepeval_adapter as adapter_mod
from deepsearch.eval.deepeval_adapter import (
    DEEPSEEK_API_BASE,
    DEEPSEEK_API_KEY_ENV,
    DEEPSEEK_MODEL_ID,
    GEMINI_MODEL_ID,
    GOOGLE_API_KEY_ENV,
    MissingCredentialsError,
    resolve_deepeval_llm,
)


# ---------------------------------------------------------------------------
# Stub openai SDK shape — enough for DeepSeekAdapter.load_model + generate.
# ---------------------------------------------------------------------------


class _StubChatChoiceMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubChatChoice:
    def __init__(self, content: str) -> None:
        self.message = _StubChatChoiceMessage(content)


class _StubChatCompletionsResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_StubChatChoice(content)]


class _StubChatCompletions:
    def __init__(self, response_content: str = "stub-completion") -> None:
        self._response_content = response_content
        self.last_call: dict | None = None

    def create(self, **kwargs):
        self.last_call = dict(kwargs)
        return _StubChatCompletionsResponse(self._response_content)


class _StubOpenAIClient:
    def __init__(self, *, api_key: str = "", base_url: str = "") -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.chat = MagicMock()
        self.chat.completions = _StubChatCompletions("ds-response")


def _install_stub_openai(monkeypatch: pytest.MonkeyPatch) -> type:
    """Inject a stub ``openai`` module so DeepSeekAdapter.load_model resolves."""

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _StubOpenAIClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return _StubOpenAIClient


# ---------------------------------------------------------------------------
# Stub google.genai SDK shape — enough for GeminiAdapter.load_model + generate.
# ---------------------------------------------------------------------------


class _StubGenerateContentResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _StubModels:
    def __init__(self, response_text: str = "gemini-response") -> None:
        self._response_text = response_text
        self.last_call: dict | None = None

    def generate_content(self, **kwargs):
        self.last_call = dict(kwargs)
        return _StubGenerateContentResponse(self._response_text)


class _StubGenaiClient:
    def __init__(self, *, api_key: str = "") -> None:
        self.api_key = api_key
        self.models = _StubModels("gemini-response")


class _StubGenerateContentConfig:
    def __init__(self, *, temperature: float = 0.0) -> None:
        self.temperature = temperature


def _install_stub_genai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject stub ``google`` + ``google.genai`` + ``google.genai.types``."""

    google_pkg = types.ModuleType("google")
    google_pkg.__path__ = []  # type: ignore[attr-defined]

    genai_pkg = types.ModuleType("google.genai")
    genai_pkg.__path__ = []  # type: ignore[attr-defined]
    genai_pkg.Client = _StubGenaiClient  # type: ignore[attr-defined]

    types_module = types.ModuleType("google.genai.types")
    types_module.GenerateContentConfig = _StubGenerateContentConfig  # type: ignore[attr-defined]
    genai_pkg.types = types_module  # type: ignore[attr-defined]
    google_pkg.genai = genai_pkg  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", genai_pkg)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)


# ---------------------------------------------------------------------------
# 1. Factory routing — known prefixes return adapters; unknown names pass through.
# ---------------------------------------------------------------------------


def test_factory_returns_deepseek_adapter_for_deepseek_v4_pro(monkeypatch):
    monkeypatch.setenv(DEEPSEEK_API_KEY_ENV, "fake-deepseek-key")
    result = resolve_deepeval_llm("deepseek-v4-pro")
    assert result is not None
    assert hasattr(result, "get_model_name")
    assert result.get_model_name() == "deepseek-v4-pro"


def test_factory_returns_gemini_adapter_for_gemini_2_5_pro(monkeypatch):
    monkeypatch.setenv(GOOGLE_API_KEY_ENV, "fake-google-key")
    result = resolve_deepeval_llm("gemini-2.5-pro")
    assert result is not None
    assert hasattr(result, "get_model_name")
    assert result.get_model_name() == "gemini-2.5-pro"


def test_factory_passes_through_unknown_model_name():
    """Unknown name returns the bare string — DeepEval's default
    string-resolution path stays in effect for non-FActScore surfaces.
    """
    assert resolve_deepeval_llm("gpt-4o") == "gpt-4o"
    assert resolve_deepeval_llm("claude-opus-4-7") == "claude-opus-4-7"
    assert resolve_deepeval_llm("") == ""
    assert resolve_deepeval_llm(None) == ""


def test_factory_normalises_case_and_whitespace(monkeypatch):
    """Mixed-case / whitespace-padded names still route to the adapter."""

    monkeypatch.setenv(DEEPSEEK_API_KEY_ENV, "fake")
    result = resolve_deepeval_llm("  DeepSeek-V4-Pro  ")
    assert hasattr(result, "get_model_name"), (
        "Whitespace + casing shouldn't drop the routing into pass-through; "
        "the factory normalises before prefix-matching."
    )


# ---------------------------------------------------------------------------
# 2. Missing credentials — typed MissingCredentialsError on load_model.
#    Construction itself never raises (per docstring contract).
# ---------------------------------------------------------------------------


def test_deepseek_adapter_construction_does_not_raise_when_key_absent(monkeypatch):
    """The redirect-fire surface needs to build the adapter speculatively
    (e.g. when deciding whether to swap to Gemini); construction must not
    pay for a credential probe. The credential check fires inside
    :meth:`generate`/:meth:`a_generate` instead.
    """

    monkeypatch.delenv(DEEPSEEK_API_KEY_ENV, raising=False)
    adapter = resolve_deepeval_llm("deepseek-v4-pro")
    assert adapter.get_model_name() == "deepseek-v4-pro"


def test_deepseek_adapter_generate_raises_missing_credentials(monkeypatch):
    monkeypatch.delenv(DEEPSEEK_API_KEY_ENV, raising=False)
    adapter = resolve_deepeval_llm("deepseek-v4-pro")
    with pytest.raises(MissingCredentialsError) as excinfo:
        adapter.generate("probe")
    assert DEEPSEEK_API_KEY_ENV in str(excinfo.value)


def test_gemini_adapter_generate_raises_missing_credentials(monkeypatch):
    monkeypatch.delenv(GOOGLE_API_KEY_ENV, raising=False)
    adapter = resolve_deepeval_llm("gemini-2.5-pro")
    with pytest.raises(MissingCredentialsError) as excinfo:
        adapter.generate("probe")
    assert GOOGLE_API_KEY_ENV in str(excinfo.value)
    # PD2 redirect-fire visibility — the message names the swap target context.
    assert "PD2" in str(excinfo.value) or "Gemini" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. Generate happy-path — stubbed SDK clients, sync + async flavours.
# ---------------------------------------------------------------------------


def test_deepseek_adapter_generate_happy_path(monkeypatch):
    monkeypatch.setenv(DEEPSEEK_API_KEY_ENV, "fake-deepseek-key")
    _install_stub_openai(monkeypatch)

    adapter = resolve_deepeval_llm("deepseek-v4-pro")
    out = adapter.generate("Atomise this claim.")

    assert out == "ds-response"
    # The wrapped client carries the locked endpoint + API key.
    client = adapter.load_model()
    assert isinstance(client, _StubOpenAIClient)
    assert client.api_key == "fake-deepseek-key"
    assert client.base_url == DEEPSEEK_API_BASE
    last = client.chat.completions.last_call
    assert last is not None
    assert last["model"] == DEEPSEEK_MODEL_ID
    assert last["temperature"] == 0.0
    assert last["messages"][0]["content"] == "Atomise this claim."


def test_deepseek_adapter_a_generate_runs_in_thread(monkeypatch):
    monkeypatch.setenv(DEEPSEEK_API_KEY_ENV, "fake-deepseek-key")
    _install_stub_openai(monkeypatch)

    adapter = resolve_deepeval_llm("deepseek-v4-pro")
    out = asyncio.run(adapter.a_generate("Async probe."))
    assert out == "ds-response"


def test_gemini_adapter_generate_happy_path(monkeypatch):
    monkeypatch.setenv(GOOGLE_API_KEY_ENV, "fake-google-key")
    _install_stub_genai(monkeypatch)

    adapter = resolve_deepeval_llm("gemini-2.5-pro")
    out = adapter.generate("Atomise this claim.")

    assert out == "gemini-response"
    client = adapter.load_model()
    assert isinstance(client, _StubGenaiClient)
    assert client.api_key == "fake-google-key"
    last = client.models.last_call
    assert last is not None
    assert last["model"] == GEMINI_MODEL_ID
    assert last["contents"] == "Atomise this claim."
    assert isinstance(last["config"], _StubGenerateContentConfig)
    assert last["config"].temperature == 0.0


def test_gemini_adapter_a_generate_runs_in_thread(monkeypatch):
    monkeypatch.setenv(GOOGLE_API_KEY_ENV, "fake-google-key")
    _install_stub_genai(monkeypatch)

    adapter = resolve_deepeval_llm("gemini-2.5-pro")
    out = asyncio.run(adapter.a_generate("Async probe."))
    assert out == "gemini-response"


# ---------------------------------------------------------------------------
# 4. get_model_name round-trip — adapters report the model_id verbatim.
# ---------------------------------------------------------------------------


def test_deepseek_adapter_get_model_name_round_trips_custom_id(monkeypatch):
    monkeypatch.setenv(DEEPSEEK_API_KEY_ENV, "fake")
    adapter = resolve_deepeval_llm("deepseek-chat")
    assert adapter.get_model_name() == "deepseek-chat"


def test_gemini_adapter_get_model_name_round_trips_custom_id(monkeypatch):
    monkeypatch.setenv(GOOGLE_API_KEY_ENV, "fake")
    adapter = resolve_deepeval_llm("gemini-1.5-flash")
    assert adapter.get_model_name() == "gemini-1.5-flash"


# ---------------------------------------------------------------------------
# 5. load_model is idempotent — repeated calls return the same client.
#    Guards against the ``self._client = None`` reset path getting flipped
#    to "always rebuild" which would re-pay the construction cost on every
#    measure() call.
# ---------------------------------------------------------------------------


def test_deepseek_adapter_load_model_caches_client(monkeypatch):
    monkeypatch.setenv(DEEPSEEK_API_KEY_ENV, "fake")
    _install_stub_openai(monkeypatch)
    adapter = resolve_deepeval_llm("deepseek-v4-pro")
    c1 = adapter.load_model()
    c2 = adapter.load_model()
    assert c1 is c2


def test_gemini_adapter_load_model_caches_client(monkeypatch):
    monkeypatch.setenv(GOOGLE_API_KEY_ENV, "fake")
    _install_stub_genai(monkeypatch)
    adapter = resolve_deepeval_llm("gemini-2.5-pro")
    c1 = adapter.load_model()
    c2 = adapter.load_model()
    assert c1 is c2


# ---------------------------------------------------------------------------
# 6. Wiring — factscore._resolve_verification_llm dispatches through the
#    factory module-level symbol, so the harness picks up the adapter
#    without any other changes to score_mission.
# ---------------------------------------------------------------------------


def test_factscore_resolve_verification_llm_uses_adapter_factory(monkeypatch):
    monkeypatch.setenv(DEEPSEEK_API_KEY_ENV, "fake-deepseek-key")
    from deepsearch.eval import factscore

    result = factscore._resolve_verification_llm("deepseek-v4-pro")
    assert hasattr(result, "get_model_name")
    assert result.get_model_name() == "deepseek-v4-pro"


def test_factscore_resolve_verification_llm_passes_through_unknown_name():
    from deepsearch.eval import factscore

    assert factscore._resolve_verification_llm("gpt-4o") == "gpt-4o"
    assert factscore._resolve_verification_llm("") == ""


# ---------------------------------------------------------------------------
# 7. DeepEval schema-kwarg contract: when called with `schema=...` the
#    adapter must raise TypeError so DeepEval's `except TypeError` clause
#    falls back to the string-JSON path. Without this, schema-aware metrics
#    (FaithfulnessMetric._generate_verdicts / _generate_truths /
#    _generate_claims) silently return a plain string into code that
#    expects a Pydantic Verdicts/Truths/Claims instance and fail with
#    `'str' object has no attribute 'verdicts'/'truths'/'claims'`.
# ---------------------------------------------------------------------------


def test_deepseek_adapter_generate_raises_type_error_on_schema_kwarg(monkeypatch):
    monkeypatch.setenv(DEEPSEEK_API_KEY_ENV, "fake")
    _install_stub_openai(monkeypatch)
    adapter = resolve_deepeval_llm("deepseek-v4-pro")
    with pytest.raises(TypeError):
        adapter.generate("probe", schema=object())


def test_deepseek_adapter_a_generate_raises_type_error_on_schema_kwarg(monkeypatch):
    monkeypatch.setenv(DEEPSEEK_API_KEY_ENV, "fake")
    _install_stub_openai(monkeypatch)
    adapter = resolve_deepeval_llm("deepseek-v4-pro")
    with pytest.raises(TypeError):
        asyncio.run(adapter.a_generate("probe", schema=object()))


def test_gemini_adapter_generate_raises_type_error_on_schema_kwarg(monkeypatch):
    monkeypatch.setenv(GOOGLE_API_KEY_ENV, "fake")
    _install_stub_genai(monkeypatch)
    adapter = resolve_deepeval_llm("gemini-2.5-pro")
    with pytest.raises(TypeError):
        adapter.generate("probe", schema=object())


def test_gemini_adapter_a_generate_raises_type_error_on_schema_kwarg(monkeypatch):
    monkeypatch.setenv(GOOGLE_API_KEY_ENV, "fake")
    _install_stub_genai(monkeypatch)
    adapter = resolve_deepeval_llm("gemini-2.5-pro")
    with pytest.raises(TypeError):
        asyncio.run(adapter.a_generate("probe", schema=object()))


def test_schema_none_does_not_raise(monkeypatch):
    """A literal ``schema=None`` (some DeepEval surfaces pass it explicitly)
    must not trigger the TypeError fallback — only positive schemas do.
    """

    monkeypatch.setenv(DEEPSEEK_API_KEY_ENV, "fake")
    _install_stub_openai(monkeypatch)
    adapter = resolve_deepeval_llm("deepseek-v4-pro")
    out = adapter.generate("probe", schema=None)
    assert out == "ds-response"
