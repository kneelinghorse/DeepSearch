"""Configuration helpers for DeepSearch."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv

from deepsearch.data_paths import data_path

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LLAMA_MODEL = "llama3"
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
DEFAULT_REFLECTION_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_EXTRACTION_THINKING_BUDGET = 2048
DEFAULT_GEMINI_SYNTHESIS_THINKING_BUDGET = 16384
DEFAULT_GEMINI_REFLECTION_THINKING_BUDGET = 0
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
DEFAULT_REFLECTION_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_TAVILY_URL = "https://api.tavily.com"
DEFAULT_MINIMAX_BASE_URL = "https://api.minimax.io/v1"
DEFAULT_MINIMAX_MODEL = "MiniMax-M2.7"
DEFAULT_REFLECTION_MINIMAX_MODEL = "MiniMax-M2.7-highspeed"
DEFAULT_THREAD_PREFIX = "deepsearch"
DEFAULT_SEARCH_BACKEND = (
    "linkup"  # "linkup", "serp", "google", "tavily", or "exa"
)
DEFAULT_DUAL_SEARCH = False  # Enable dual search (primary + exa for more sources)
DEFAULT_TRIPLE_SEARCH = False  # Enable triple search (primary + exa + serpapi for Alpha diversity)
DEFAULT_TOOL_MAX_RETRIES = 3
DEFAULT_TOOL_BACKOFF_BASE = 1.0
DEFAULT_TOOL_BACKOFF_MAX = 60.0
DEFAULT_TOOL_BACKOFF_JITTER_MS = 200
DEFAULT_TOOL_CIRCUIT_FAILURE_THRESHOLD = 5
DEFAULT_TOOL_CIRCUIT_SUCCESS_THRESHOLD = 2
DEFAULT_TOOL_CIRCUIT_TIMEOUT_SECONDS = 60
DEFAULT_TOOL_TIMEOUT_SECONDS = 60.0
DEFAULT_LLM_MAX_RETRIES = 3
DEFAULT_LLM_BACKOFF_BASE = 1.0
DEFAULT_LLM_BACKOFF_MAX = 30.0
DEFAULT_LLM_BACKOFF_JITTER_MS = 150
# Wall-clock ceiling for a single LLM request. The openai client defaults to
# 600s with automatic retries, so an unresponsive endpoint can hold a mission
# lease for tens of minutes — 18 was observed on 2026-08-13. 300s is well clear
# of a real long-form generation (the slowest observed step is ~30s) while
# bounding a hang to something the S89 synthesis retry can absorb.
DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 300.0
SUPPORTED_LLM_BACKENDS = ("ollama", "gemini", "deepseek", "minimax", "openai")
# The backend the production worker actually runs. An unset or unrecognized
# LLM_BACKEND used to resolve to "gemini", which silently routed contract
# compilation through the Gemini structured extractor.
DEFAULT_LLM_BACKEND = "deepseek"
DEFAULT_PEDR_ENABLE_GRAPH = True
TRACELAB_DEEPSEARCH_SERVICE_SECRET_ENV = "TRACELAB_DEEPSEARCH_SERVICE_SECRET"


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _as_int_optional(value: str | None) -> int | None:
    """Parse an optional integer from environment configuration."""

    text = (value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _resolve_optional_int_env(
    name: str,
    *,
    default: int | None = None,
    legacy: int | None = None,
) -> int | None:
    """Resolve an optional integer env var with explicit-disable semantics."""

    raw_value = os.getenv(name)
    if raw_value is not None:
        return _as_int_optional(raw_value)
    if legacy is not None:
        return legacy
    return default


def _resolve_llm_backend(raw: str | None) -> str:
    """Preserve unknown selections so preflight can diagnose them without routing."""
    return (raw or DEFAULT_LLM_BACKEND).lower().strip() or DEFAULT_LLM_BACKEND


def _normalize_model_name(value: str | None, *, default: str, lowercase: bool) -> str:
    text = (value or "").strip()
    if not text:
        text = default
    if lowercase:
        text = re.sub(r"\s+", "-", text.lower())
    return text


@dataclass(slots=True)
class LLMBackendConfig:
    """Configuration payload describing the selected LLM backend."""

    backend: Literal["ollama", "gemini", "deepseek", "minimax", "openai"]
    model: str
    temperature: float | None
    max_output_tokens: int
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] | None = None
    thinking_budget: int | None = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    # Per-request wall-clock ceiling handed to the provider client. Defaulted
    # rather than optional so a hand-built config cannot produce an unbounded
    # call by omission.
    request_timeout: float = DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS


LLMCallType = Literal["extraction", "synthesis"]


@dataclass(slots=True)
class DeepSearchSettings:
    """Application configuration derived from environment variables."""

    ollama_base_url: str = DEFAULT_OLLAMA_URL
    gemini_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    tavily_base_url: str = DEFAULT_TAVILY_URL
    exa_api_key: Optional[str] = None
    serpapi_api_key: Optional[str] = None
    linkup_api_key: Optional[str] = None
    search_backend: Literal["google", "tavily", "exa", "serp", "linkup"] = "linkup"
    dual_search: bool = False  # When True, use primary + exa for more diverse sources
    triple_search: bool = False  # B13.7: When True, add SerpAPI for Alpha source diversity
    tavily_dual_search: bool = (
        False  # B23.4: add Tavily as third source for Deep/Alpha
    )
    semantic_scholar_api_key: Optional[str] = None  # B24.4: Semantic Scholar API key
    openalex_api_key: Optional[str] = None  # B24.4: OpenAlex API key (polite pool token)
    minimax_api_key: Optional[str] = None
    minimax_base_url: str = DEFAULT_MINIMAX_BASE_URL
    minimax_model: str = DEFAULT_MINIMAX_MODEL
    reflection_minimax_model: str = DEFAULT_REFLECTION_MINIMAX_MODEL
    minimax_temperature: float = 0.7
    minimax_max_output_tokens: int = 8000
    # Keep unvalidated selections intact so preflight can reject typos safely.
    llm_backend: str = "deepseek"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-5.6-luna"
    openai_max_output_tokens: int = 32768
    llm_reasoning_effort: str | None = None
    ollama_model: str = DEFAULT_LLAMA_MODEL
    ollama_temperature: float = 0.7
    ollama_max_output_tokens: int = 2000
    gemini_model: str = DEFAULT_GEMINI_MODEL
    reflection_gemini_model: str = DEFAULT_REFLECTION_GEMINI_MODEL
    gemini_temperature: float = 0.7
    gemini_max_output_tokens: int = 8000  # Increased for comprehensive research reports
    gemini_extraction_thinking_budget: int | None = DEFAULT_GEMINI_EXTRACTION_THINKING_BUDGET
    gemini_synthesis_thinking_budget: int | None = DEFAULT_GEMINI_SYNTHESIS_THINKING_BUDGET
    gemini_reflection_thinking_budget: int | None = DEFAULT_GEMINI_REFLECTION_THINKING_BUDGET
    gemini_thinking_budget: int | None = None  # Deprecated legacy fallback
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    deepseek_model: str = DEFAULT_DEEPSEEK_MODEL
    reflection_deepseek_model: str = DEFAULT_REFLECTION_DEEPSEEK_MODEL
    deepseek_temperature: float = 0.7
    deepseek_max_output_tokens: int = 32768
    log_level: str = DEFAULT_LOG_LEVEL
    checkpoint_db_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("CHECKPOINT_DB_PATH") or data_path("deepsearch.db")
        )
    )
    thread_id_prefix: str = DEFAULT_THREAD_PREFIX
    tool_max_retries: int = DEFAULT_TOOL_MAX_RETRIES
    tool_backoff_base: float = DEFAULT_TOOL_BACKOFF_BASE
    tool_backoff_max: float = DEFAULT_TOOL_BACKOFF_MAX
    tool_backoff_jitter_ms: int = DEFAULT_TOOL_BACKOFF_JITTER_MS
    tool_circuit_failure_threshold: int = DEFAULT_TOOL_CIRCUIT_FAILURE_THRESHOLD
    tool_circuit_success_threshold: int = DEFAULT_TOOL_CIRCUIT_SUCCESS_THRESHOLD
    tool_circuit_timeout_seconds: int = DEFAULT_TOOL_CIRCUIT_TIMEOUT_SECONDS
    tool_timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS
    llm_max_retries: int = DEFAULT_LLM_MAX_RETRIES
    llm_backoff_base: float = DEFAULT_LLM_BACKOFF_BASE
    llm_backoff_max: float = DEFAULT_LLM_BACKOFF_MAX
    llm_backoff_jitter_ms: int = DEFAULT_LLM_BACKOFF_JITTER_MS
    llm_request_timeout_seconds: float = DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS
    # Feature flags for parser hardening (B3.2)
    use_new_citation_engine: bool = True
    use_new_reflection_parser: bool = True
    # Feature flag for overhauled prompts (B6.3)
    use_v2_prompts: bool = True
    pedr_enable_graph: bool = DEFAULT_PEDR_ENABLE_GRAPH

    @classmethod
    def load(cls) -> "DeepSearchSettings":
        """Load environment variables and build a settings instance."""

        load_dotenv()
        llm_backend = _resolve_llm_backend(os.getenv("LLM_BACKEND"))
        search_backend = os.getenv("SEARCH_BACKEND", DEFAULT_SEARCH_BACKEND).lower().strip()
        if search_backend not in {"google", "tavily", "exa", "serp", "linkup"}:
            search_backend = "linkup"
        dual_search = os.getenv("DUAL_SEARCH", "false").lower() == "true"
        triple_search = os.getenv("TRIPLE_SEARCH", "false").lower() == "true"
        tavily_dual_search = os.getenv("TAVILY_DUAL_SEARCH", "false").lower() == "true"
        legacy_gemini_thinking_budget = _as_int_optional(os.getenv("GEMINI_THINKING_BUDGET"))
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            tavily_base_url=os.getenv("TAVILY_BASE_URL", DEFAULT_TAVILY_URL),
            exa_api_key=os.getenv("EXA_API_KEY"),
            serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
            linkup_api_key=os.getenv("LINKUP_API_KEY"),
            search_backend=search_backend,
            dual_search=dual_search,
            triple_search=triple_search,
            tavily_dual_search=tavily_dual_search,
            semantic_scholar_api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            openalex_api_key=os.getenv("OPENALEX_API_KEY"),
            llm_backend=llm_backend,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip(),
            openai_max_output_tokens=_as_int(os.getenv("OPENAI_MAX_TOKENS", "32768"), -1),
            llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT"),
            ollama_model=_normalize_model_name(
                os.getenv("OLLAMA_MODEL"),
                default=DEFAULT_LLAMA_MODEL,
                lowercase=False,
            ),
            ollama_temperature=_as_float(os.getenv("OLLAMA_TEMPERATURE"), 0.7),
            ollama_max_output_tokens=_as_int(os.getenv("OLLAMA_MAX_TOKENS"), 2000),
            gemini_model=_normalize_model_name(
                os.getenv("GEMINI_MODEL"),
                default=DEFAULT_GEMINI_MODEL,
                lowercase=True,
            ),
            reflection_gemini_model=_normalize_model_name(
                os.getenv("REFLECTION_GEMINI_MODEL"),
                default=DEFAULT_REFLECTION_GEMINI_MODEL,
                lowercase=True,
            ),
            gemini_temperature=_as_float(os.getenv("GEMINI_TEMPERATURE"), 0.7),
            gemini_max_output_tokens=_as_int(os.getenv("GEMINI_MAX_TOKENS"), 8000),
            gemini_extraction_thinking_budget=_resolve_optional_int_env(
                "GEMINI_EXTRACTION_THINKING_BUDGET",
                default=DEFAULT_GEMINI_EXTRACTION_THINKING_BUDGET,
                legacy=legacy_gemini_thinking_budget,
            ),
            gemini_synthesis_thinking_budget=_resolve_optional_int_env(
                "GEMINI_SYNTHESIS_THINKING_BUDGET",
                default=DEFAULT_GEMINI_SYNTHESIS_THINKING_BUDGET,
                legacy=legacy_gemini_thinking_budget,
            ),
            gemini_reflection_thinking_budget=_resolve_optional_int_env(
                "GEMINI_REFLECTION_THINKING_BUDGET",
                default=DEFAULT_GEMINI_REFLECTION_THINKING_BUDGET,
                legacy=legacy_gemini_thinking_budget,
            ),
            gemini_thinking_budget=legacy_gemini_thinking_budget,
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
            deepseek_model=_normalize_model_name(
                os.getenv("DEEPSEEK_MODEL"),
                default=DEFAULT_DEEPSEEK_MODEL,
                lowercase=False,
            ),
            reflection_deepseek_model=_normalize_model_name(
                os.getenv("REFLECTION_DEEPSEEK_MODEL"),
                default=DEFAULT_REFLECTION_DEEPSEEK_MODEL,
                lowercase=False,
            ),
            deepseek_temperature=_as_float(os.getenv("DEEPSEEK_TEMPERATURE"), 0.7),
            deepseek_max_output_tokens=_as_int(os.getenv("DEEPSEEK_MAX_TOKENS", "32768"), -1),
            minimax_api_key=os.getenv("MINIMAX_API_KEY"),
            minimax_base_url=os.getenv("MINIMAX_BASE_URL", DEFAULT_MINIMAX_BASE_URL),
            minimax_model=_normalize_model_name(
                os.getenv("MINIMAX_MODEL"),
                default=DEFAULT_MINIMAX_MODEL,
                lowercase=False,
            ),
            reflection_minimax_model=_normalize_model_name(
                os.getenv("REFLECTION_MINIMAX_MODEL"),
                default=DEFAULT_REFLECTION_MINIMAX_MODEL,
                lowercase=False,
            ),
            minimax_temperature=_as_float(os.getenv("MINIMAX_TEMPERATURE"), 0.7),
            minimax_max_output_tokens=_as_int(os.getenv("MINIMAX_MAX_TOKENS"), 8000),
            log_level=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL),
            thread_id_prefix=os.getenv("THREAD_ID_PREFIX", DEFAULT_THREAD_PREFIX),
            tool_max_retries=_as_int(os.getenv("TOOL_MAX_RETRIES"), DEFAULT_TOOL_MAX_RETRIES),
            tool_backoff_base=_as_float(os.getenv("TOOL_BACKOFF_BASE"), DEFAULT_TOOL_BACKOFF_BASE),
            tool_backoff_max=_as_float(os.getenv("TOOL_BACKOFF_MAX"), DEFAULT_TOOL_BACKOFF_MAX),
            tool_backoff_jitter_ms=_as_int(
                os.getenv("TOOL_BACKOFF_JITTER_MS"),
                DEFAULT_TOOL_BACKOFF_JITTER_MS,
            ),
            tool_circuit_failure_threshold=_as_int(
                os.getenv("TOOL_CIRCUIT_FAILURE_THRESHOLD"),
                DEFAULT_TOOL_CIRCUIT_FAILURE_THRESHOLD,
            ),
            tool_circuit_success_threshold=_as_int(
                os.getenv("TOOL_CIRCUIT_SUCCESS_THRESHOLD"),
                DEFAULT_TOOL_CIRCUIT_SUCCESS_THRESHOLD,
            ),
            tool_circuit_timeout_seconds=_as_int(
                os.getenv("TOOL_CIRCUIT_TIMEOUT_SECONDS"),
                DEFAULT_TOOL_CIRCUIT_TIMEOUT_SECONDS,
            ),
            tool_timeout_seconds=_as_float(
                os.getenv("TOOL_TIMEOUT_SECONDS"), DEFAULT_TOOL_TIMEOUT_SECONDS
            ),
            llm_max_retries=_as_int(os.getenv("LLM_MAX_RETRIES"), DEFAULT_LLM_MAX_RETRIES),
            llm_backoff_base=_as_float(os.getenv("LLM_BACKOFF_BASE"), DEFAULT_LLM_BACKOFF_BASE),
            llm_backoff_max=_as_float(os.getenv("LLM_BACKOFF_MAX"), DEFAULT_LLM_BACKOFF_MAX),
            llm_backoff_jitter_ms=_as_int(
                os.getenv("LLM_BACKOFF_JITTER_MS"),
                DEFAULT_LLM_BACKOFF_JITTER_MS,
            ),
            llm_request_timeout_seconds=_as_float(
                os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", str(DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS)),
                -1.0,
            ),
            use_new_citation_engine=(
                os.getenv("USE_NEW_CITATION_ENGINE", "true").lower() == "true"
            ),
            use_new_reflection_parser=(
                os.getenv("USE_NEW_REFLECTION_PARSER", "true").lower() == "true"
            ),
            use_v2_prompts=os.getenv("USE_V2_PROMPTS", "true").lower() == "true",
            pedr_enable_graph=os.getenv("PEDR_ENABLE_GRAPH", "true").lower() == "true",
        )

    def build_llm_config(
        self,
        *,
        call_type: LLMCallType = "extraction",
    ) -> LLMBackendConfig:
        """Return the configuration for the currently selected LLM backend."""

        timeout = self.llm_request_timeout_seconds
        if self.llm_backend == "gemini":
            return LLMBackendConfig(
                backend="gemini",
                model=self.gemini_model,
                temperature=self.gemini_temperature,
                max_output_tokens=self.gemini_max_output_tokens,
                thinking_budget=self._resolve_gemini_thinking_budget(call_type),
                api_key=self.gemini_api_key,
                request_timeout=timeout,
            )
        if self.llm_backend in {"deepseek", "openai"}:
            from deepsearch.llm.runtime import resolve_effective_model

            effective = resolve_effective_model(self)
            return LLMBackendConfig(
                backend=effective.provider,
                model=effective.model,
                temperature=None,
                max_output_tokens=effective.max_output_tokens,
                reasoning_effort=effective.reasoning_effort,
                extra_body=effective.request_parameters().get("extra_body"),
                api_key=effective.api_key,
                base_url=effective.base_url,
                request_timeout=effective.request_timeout,
            )
        if self.llm_backend == "minimax":
            return LLMBackendConfig(
                backend="minimax",
                model=self.minimax_model,
                temperature=self.minimax_temperature,
                max_output_tokens=self.minimax_max_output_tokens,
                api_key=self.minimax_api_key,
                base_url=self.minimax_base_url,
                request_timeout=timeout,
            )
        if self.llm_backend != "ollama":
            raise ValueError("Unsupported LLM_BACKEND")
        return LLMBackendConfig(
            backend="ollama",
            model=self.ollama_model,
            temperature=self.ollama_temperature,
            max_output_tokens=self.ollama_max_output_tokens,
            base_url=self.ollama_base_url,
            request_timeout=timeout,
        )

    def build_reflection_llm_config(self) -> LLMBackendConfig:
        """Return the LLM config for the reflection/judging model.

        Defaults to gemini-2.5-flash for cost efficiency — reflection
        only needs to score and evaluate, not generate long-form content.
        Override via REFLECTION_GEMINI_MODEL env var.
        """
        timeout = self.llm_request_timeout_seconds
        if self.llm_backend == "gemini":
            return LLMBackendConfig(
                backend="gemini",
                model=self.reflection_gemini_model,
                temperature=self.gemini_temperature,
                max_output_tokens=self.gemini_max_output_tokens,
                thinking_budget=self._resolve_gemini_reflection_budget(),
                api_key=self.gemini_api_key,
                request_timeout=timeout,
            )
        if self.llm_backend == "deepseek":
            return LLMBackendConfig(
                backend="deepseek",
                model=self.reflection_deepseek_model,
                temperature=self.deepseek_temperature,
                max_output_tokens=self.deepseek_max_output_tokens,
                api_key=self.deepseek_api_key,
                base_url=self.deepseek_base_url,
                request_timeout=timeout,
            )
        if self.llm_backend == "minimax":
            return LLMBackendConfig(
                backend="minimax",
                model=self.reflection_minimax_model,
                temperature=self.minimax_temperature,
                max_output_tokens=self.minimax_max_output_tokens,
                api_key=self.minimax_api_key,
                base_url=self.minimax_base_url,
                request_timeout=timeout,
            )
        # Ollama doesn't differentiate — use same config
        return self.build_llm_config()

    def _resolve_gemini_thinking_budget(self, call_type: LLMCallType) -> int | None:
        """Return the Gemini thinking budget for the requested call type."""

        if call_type == "synthesis":
            if self.gemini_synthesis_thinking_budget is not None:
                return self.gemini_synthesis_thinking_budget
        elif self.gemini_extraction_thinking_budget is not None:
            return self.gemini_extraction_thinking_budget
        return self.gemini_thinking_budget

    def _resolve_gemini_reflection_budget(self) -> int | None:
        """Return the reflection thinking budget with legacy fallback."""

        if self.gemini_reflection_thinking_budget is not None:
            return self.gemini_reflection_thinking_budget
        return self.gemini_thinking_budget

    def build_structured_extractor(self) -> Any:
        """Return a provider-portable structured-output extractor.

        Sprint 54 L1. Mirrors ``build_llm_config`` dispatch: Gemini gets the
        grammar-constrained adapter; everything else gets the universal
        JSON-in-prompt fallback wrapping the same chat model that
        ``deepsearch.agent.llm_client.create_model`` would build.
        """

        from deepsearch.llm.structured_extractor import (
            GeminiStructuredExtractor,
            JSONInPromptStructuredExtractor,
        )

        if self.llm_backend == "gemini":
            if not self.gemini_api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY must be configured for Gemini structured extraction"
                )
            return GeminiStructuredExtractor(
                api_key=self.gemini_api_key,
                model=self.gemini_model,
                temperature=0.0,
                max_output_tokens=self.gemini_max_output_tokens,
                thinking_budget=self._resolve_gemini_thinking_budget("extraction"),
            )

        from deepsearch.agent.llm_client import create_model

        chat_model = create_model(self.build_llm_config())
        effective_config = {}
        if self.llm_backend in {"deepseek", "openai"}:
            from deepsearch.llm.runtime import resolve_effective_model

            effective_config = resolve_effective_model(self).public_metadata()
        return JSONInPromptStructuredExtractor(
            chat_model=chat_model, effective_config=effective_config
        )

    def generate_thread_id(
        self,
        mission_id: str,
        *,
        execution_token: str | None = None,
    ) -> str:
        """Return a LangGraph thread identifier for a mission.

        By default this remains deterministic for a given mission ID so the
        local runner and resume flows keep their existing behavior. When an
        ``execution_token`` is supplied, the token is appended as a run-scoped
        suffix so independent executions do not silently resume each other's
        checkpoints.
        """

        slug = re.sub(r"[^a-z0-9]+", "-", mission_id.strip().lower())
        slug = slug.strip("-") or "mission"
        thread_id = f"{self.thread_id_prefix}-{slug}"
        if execution_token:
            token_slug = re.sub(r"[^a-z0-9]+", "-", execution_token.strip().lower())
            token_slug = token_slug.strip("-")[:12]
            if token_slug:
                thread_id = f"{thread_id}-{token_slug}"
        return thread_id


__all__ = [
    "DeepSearchSettings",
    "LLMBackendConfig",
    "TRACELAB_DEEPSEARCH_SERVICE_SECRET_ENV",
]
