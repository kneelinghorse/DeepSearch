"""S71 Path B — DeepEvalBaseLLM adapter for the FActScore harness.

Closes substrate blocker (3) from S70 m02 calibration: DeepEval's
``FaithfulnessMetric(model=…)`` cannot resolve a plain string into a
runnable LLM, so the harness needs a thin ``DeepEvalBaseLLM`` shim for
the two production-relevant verification models.

PD16 (mission set) — Path B = this adapter. Pairs with Path A
(``sources_collected`` persistence on ``MissionProtocolComplete``); both
must close together for FActScore divergence to be measurable on the
canonical fixture.

PD12 (S70 carry-forward) — locked swap target on PD2 redirect fire =
Gemini 2.5 Pro. The DeepSeek default matches the locked S62 production
binding (``deepseek-v4-pro`` per
``project_smolagents_v124_bind.md``), so the same-model bias baseline
runs without a swap.

PD10 LOCAL-ONLY contract — extends to this module. No imports from
production-runtime surfaces (``codeagent_finalize_node``,
``citation_liveness``, ``critique``, ``contract_to_prompt``,
``codeagent_runner``, ``worker.writer``, ``agent.tools.web_search``,
``agent.tools.source_fetch``). The adapter wraps third-party SDKs
(``openai`` / ``google.genai``) directly and is consumed only by the
harness module ``deepsearch.eval.factscore``.

Surface design:

* :func:`resolve_deepeval_llm` — single entry-point factory. Returns a
  :class:`DeepEvalBaseLLM` adapter for known model names, or the
  original ``model_name`` string verbatim for anything else (pass-
  through preserves DeepEval's default string-resolution path for
  non-FActScore surfaces).
* :class:`DeepSeekAdapter` — wraps the OpenAI-compatible DeepSeek API
  via the ``openai`` Python client. Reuses ``DEEPSEEK_API_KEY`` for
  parity with the locked S62 runtime bind.
* :class:`GeminiAdapter` — wraps the Google GenAI SDK
  (``google-genai``). Uses ``GOOGLE_API_KEY``.
* :class:`MissingCredentialsError` — typed error so a redirect-fire
  path emits an actionable message (PD2 swap) rather than a generic
  resolution failure.

Lazy SDK imports (mirrors :func:`deepsearch.eval.factscore._resolve_deepeval`)
keep this module importable without ``deepeval``, ``openai``, or
``google.genai`` installed. The actual SDK calls happen inside
:meth:`load_model` / :meth:`generate` / :meth:`a_generate`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


DEEPSEEK_MODEL_ID: str = "deepseek-v4-pro"
"""PD2 same-model bias baseline; matches the locked S62 production stack."""

DEEPSEEK_API_BASE: str = "https://api.deepseek.com/v1"
"""DeepSeek's OpenAI-compatible endpoint; matches codeagent_runner."""

GEMINI_MODEL_ID: str = "gemini-2.5-pro"
"""PD12-locked swap target on PD2 redirect fire."""

DEEPSEEK_API_KEY_ENV: str = "DEEPSEEK_API_KEY"
GOOGLE_API_KEY_ENV: str = "GOOGLE_API_KEY"


class MissingCredentialsError(RuntimeError):
    """Raised when an adapter cannot find the credentials it needs.

    Carries a typed name so the redirect-fire surface in
    :func:`deepsearch.eval.factscore.score_mission` can branch on
    "swap target unreachable" vs. a generic LLM error.
    """


# ---------------------------------------------------------------------------
# DeepEvalBaseLLM resolution — lazy so the module imports without deepeval.
# Tests monkeypatch this to inject a stub base class.
# ---------------------------------------------------------------------------


def _resolve_base_llm() -> Any:
    """Return DeepEval's ``DeepEvalBaseLLM`` class, lazily.

    The harness's ``_resolve_deepeval`` helper handles the
    ``FaithfulnessMetric``/``LLMTestCase`` pair; this sibling helper
    handles the abstract base class the adapters subclass dynamically
    inside :func:`resolve_deepeval_llm` so AST imports stay clean.
    """

    try:
        from deepeval.models.base_model import DeepEvalBaseLLM  # type: ignore
    except ImportError as exc:  # pragma: no cover — dependency guard
        raise RuntimeError(
            "deepeval is not installed; run `pip install deepeval` to enable "
            "the FActScore harness. The DeepEvalBaseLLM adapter cannot be "
            "constructed without it."
        ) from exc
    return DeepEvalBaseLLM


# ---------------------------------------------------------------------------
# Adapter implementations.
# ---------------------------------------------------------------------------


def _build_deepseek_adapter_class(base_cls: Any) -> Any:
    """Construct the DeepSeek adapter class against the runtime base.

    Done lazily so the module imports without deepeval. Subclassing
    ``DeepEvalBaseLLM`` at module-import time would force the import
    eagerly, breaking the AST-guard and persistence tests that don't
    install the SDK.
    """

    class DeepSeekAdapter(base_cls):
        """Wraps the OpenAI-compatible DeepSeek chat-completions endpoint.

        Same provider + endpoint + model_id the locked S62 production
        binding uses (``OpenAIServerModel(deepseek-v4-pro,
        api_base="https://api.deepseek.com/v1")``), via the ``openai``
        Python client rather than smolagents (PD10 keeps eval/ off the
        agent-side surfaces).

        Construction is non-raising — DeepEval's base ``__init__`` calls
        ``load_model`` eagerly, but the credential check is deferred to
        :meth:`generate` / :meth:`a_generate`. This lets the factory
        return an adapter on the redirect-fire path even before the
        caller has populated the env var, so the surface that needs to
        emit the actionable :class:`MissingCredentialsError` is the
        actual scoring callsite (not the construction site).
        """

        def __init__(
            self,
            *,
            model_id: str = DEEPSEEK_MODEL_ID,
            api_base: str = DEEPSEEK_API_BASE,
            api_key: Optional[str] = None,
            temperature: float = 0.0,
        ) -> None:
            self._model_id = model_id
            self._api_base = api_base
            self._api_key = api_key or os.environ.get(DEEPSEEK_API_KEY_ENV)
            self._temperature = float(temperature)
            self._client: Any = None
            super().__init__(model_name=model_id)

        def get_model_name(self) -> str:
            return self._model_id

        def load_model(self) -> Any:
            """Return the cached OpenAI client, building it lazily.

            Returns ``None`` (rather than raising) when credentials are
            absent — the base ``__init__`` calls this eagerly and we do
            not want construction to fail. The credential check fires
            inside :meth:`generate` / :meth:`a_generate` via
            :meth:`_require_client`.
            """
            if self._client is not None:
                return self._client
            if not self._api_key:
                return None
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:  # pragma: no cover — dep guard
                raise RuntimeError(
                    "`openai` Python client not installed; DeepSeek adapter "
                    "needs the OpenAI-compatible client to call DeepSeek's API."
                ) from exc
            self._client = OpenAI(api_key=self._api_key, base_url=self._api_base)
            return self._client

        def _require_client(self) -> Any:
            client = self.load_model()
            if client is None:
                raise MissingCredentialsError(
                    f"{DEEPSEEK_API_KEY_ENV} not set; DeepSeek adapter cannot "
                    "authenticate with DeepSeek's OpenAI-compatible endpoint. "
                    "Set the env var or pass api_key explicitly."
                )
            return client

        def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            # DeepEval's metric code calls ``model.generate(prompt, schema=X)``
            # for native-Pydantic-output models and falls back to a plain
            # string-JSON path inside an ``except TypeError`` clause. Raising
            # TypeError here triggers that fallback — DeepEval's templates
            # already instruct the LLM to "return in JSON format" so the
            # string path round-trips through ``trimAndLoadJson`` correctly.
            schema = kwargs.get("schema") if "schema" in kwargs else None
            if schema is not None:
                raise TypeError(
                    "DeepSeekAdapter does not support DeepEval native-schema "
                    "responses; use the string-JSON fallback path."
                )
            client = self._require_client()
            response = client.chat.completions.create(
                model=self._model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
            )
            choice = response.choices[0]
            content = getattr(choice.message, "content", None)
            return str(content or "")

        async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            schema = kwargs.get("schema") if "schema" in kwargs else None
            if schema is not None:
                raise TypeError(
                    "DeepSeekAdapter does not support DeepEval native-schema "
                    "responses; use the string-JSON fallback path."
                )
            return await asyncio.to_thread(self.generate, prompt)

    return DeepSeekAdapter


def _build_gemini_adapter_class(base_cls: Any) -> Any:
    """Construct the Gemini adapter class against the runtime base.

    PD12 — Gemini 2.5 Pro is the locked PD2 swap target. Wraps
    ``google.genai`` rather than ``langchain-google-genai`` so the
    adapter stays dependency-light (the harness already pulls
    ``google-genai>=0.4.0`` for the production runtime; no new top-
    level dep).
    """

    class GeminiAdapter(base_cls):
        """Wraps the Google GenAI SDK for Gemini 2.5 Pro.

        Construction is non-raising — DeepEval's base ``__init__`` calls
        ``load_model`` eagerly, but the credential check is deferred to
        :meth:`generate` / :meth:`a_generate`.
        """

        def __init__(
            self,
            *,
            model_id: str = GEMINI_MODEL_ID,
            api_key: Optional[str] = None,
            temperature: float = 0.0,
        ) -> None:
            self._model_id = model_id
            self._api_key = api_key or os.environ.get(GOOGLE_API_KEY_ENV)
            self._temperature = float(temperature)
            self._client: Any = None
            super().__init__(model_name=model_id)

        def get_model_name(self) -> str:
            return self._model_id

        def load_model(self) -> Any:
            if self._client is not None:
                return self._client
            if not self._api_key:
                return None
            try:
                from google import genai  # type: ignore
            except ImportError as exc:  # pragma: no cover — dep guard
                raise RuntimeError(
                    "`google-genai` SDK not installed; Gemini adapter needs "
                    "it to call Gemini 2.5 Pro."
                ) from exc
            self._client = genai.Client(api_key=self._api_key)
            return self._client

        def _require_client(self) -> Any:
            client = self.load_model()
            if client is None:
                raise MissingCredentialsError(
                    f"{GOOGLE_API_KEY_ENV} not set; Gemini adapter cannot "
                    "authenticate with the Google GenAI SDK. Set the env "
                    "var or pass api_key explicitly. PD2 redirect-fire "
                    "swap target requires this credential."
                )
            return client

        def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            # See DeepSeekAdapter.generate for the schema-kwarg rationale —
            # DeepEval falls back to the string-JSON path on TypeError.
            schema = kwargs.get("schema") if "schema" in kwargs else None
            if schema is not None:
                raise TypeError(
                    "GeminiAdapter does not support DeepEval native-schema "
                    "responses; use the string-JSON fallback path."
                )
            client = self._require_client()
            try:
                from google.genai import types as genai_types  # type: ignore
            except ImportError as exc:  # pragma: no cover — dep guard
                raise RuntimeError(
                    "`google-genai` SDK missing the `types` submodule; "
                    "cannot configure GenerationConfig."
                ) from exc
            response = client.models.generate_content(
                model=self._model_id,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=self._temperature,
                ),
            )
            text = getattr(response, "text", None)
            return str(text or "")

        async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
            schema = kwargs.get("schema") if "schema" in kwargs else None
            if schema is not None:
                raise TypeError(
                    "GeminiAdapter does not support DeepEval native-schema "
                    "responses; use the string-JSON fallback path."
                )
            return await asyncio.to_thread(self.generate, prompt)

    return GeminiAdapter


# ---------------------------------------------------------------------------
# Factory — single entry-point consumed by deepsearch.eval.factscore.
# ---------------------------------------------------------------------------


# Public alias for the adapter union returned by the factory. ``str`` falls
# through verbatim for unsupported model names (DeepEval's default
# string-resolution path stays in effect).
DeepEvalLLMOrName = Union[Any, str]


def _normalize_model_name(model_name: Optional[str]) -> str:
    if not model_name:
        return ""
    return model_name.strip().lower()


def resolve_deepeval_llm(model_name: Optional[str]) -> DeepEvalLLMOrName:
    """Resolve ``model_name`` to a DeepEvalBaseLLM adapter (or pass through).

    Known prefixes:

    * ``"deepseek"`` — returns a :class:`DeepSeekAdapter` instance bound
      to the locked S62 endpoint + API key. Same-model bias baseline.
    * ``"gemini"`` — returns a :class:`GeminiAdapter` instance for
      Gemini 2.5 Pro (PD12 PD2-redirect swap target).

    Anything else (empty string, ``None``, or an unknown name) is
    passed through verbatim so DeepEval's existing string-resolution
    behaviour is preserved on non-FActScore surfaces.

    Raises :class:`MissingCredentialsError` only when the caller
    actually invokes :meth:`load_model` / :meth:`generate` /
    :meth:`a_generate` on the returned adapter without the relevant
    env var set. Construction itself never raises.
    """

    normalised = _normalize_model_name(model_name)
    if not normalised:
        return model_name or ""

    if normalised.startswith("deepseek"):
        base_cls = _resolve_base_llm()
        adapter_cls = _build_deepseek_adapter_class(base_cls)
        return adapter_cls(model_id=model_name or DEEPSEEK_MODEL_ID)

    if normalised.startswith("gemini"):
        base_cls = _resolve_base_llm()
        adapter_cls = _build_gemini_adapter_class(base_cls)
        return adapter_cls(model_id=model_name or GEMINI_MODEL_ID)

    return model_name


__all__ = [
    "DEEPSEEK_API_BASE",
    "DEEPSEEK_API_KEY_ENV",
    "DEEPSEEK_MODEL_ID",
    "DeepEvalLLMOrName",
    "GEMINI_MODEL_ID",
    "GOOGLE_API_KEY_ENV",
    "MissingCredentialsError",
    "resolve_deepeval_llm",
]
