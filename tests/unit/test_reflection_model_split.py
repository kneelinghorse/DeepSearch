"""Tests for S20-M03 reflection model split.

Covers:
- DeepSearchSettings.reflection_gemini_model default and env override
- build_reflection_llm_config() returns correct model
- Ollama fallback uses same config for both
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from deepsearch.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_REFLECTION_GEMINI_MODEL,
    DeepSearchSettings,
    LLMBackendConfig,
)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestReflectionModelDefaults:
    """Verify default reflection model configuration."""

    def test_default_reflection_model(self) -> None:
        """Default reflection model should be gemini-2.5-flash."""
        settings = DeepSearchSettings()
        assert settings.reflection_gemini_model == "gemini-2.5-flash"

    def test_default_research_model_unchanged(self) -> None:
        """Research model default should still be gemini-2.5-pro."""
        settings = DeepSearchSettings()
        assert settings.gemini_model == "gemini-2.5-pro"

    def test_models_are_different(self) -> None:
        """Research and reflection should use different models by default."""
        settings = DeepSearchSettings()
        assert settings.gemini_model != settings.reflection_gemini_model


# ---------------------------------------------------------------------------
# Environment variable override
# ---------------------------------------------------------------------------


class TestReflectionModelEnvOverride:
    """Verify env var overrides for reflection model."""

    def test_reflection_model_env_override(self) -> None:
        """REFLECTION_GEMINI_MODEL env var should override default."""
        with patch.dict(os.environ, {
            "REFLECTION_GEMINI_MODEL": "gemini-3-flash-preview",
            "GEMINI_API_KEY": "test-key",
        }):
            settings = DeepSearchSettings.load()
        assert settings.reflection_gemini_model == "gemini-3-flash-preview"

    def test_reflection_model_empty_env_uses_default(self) -> None:
        """Empty REFLECTION_GEMINI_MODEL should fall back to default."""
        with patch.dict(os.environ, {
            "REFLECTION_GEMINI_MODEL": "",
            "GEMINI_API_KEY": "test-key",
        }):
            settings = DeepSearchSettings.load()
        assert settings.reflection_gemini_model == DEFAULT_REFLECTION_GEMINI_MODEL

    def test_research_model_unaffected_by_reflection_override(self) -> None:
        """Overriding reflection model should not change research model."""
        with patch.dict(os.environ, {
            "REFLECTION_GEMINI_MODEL": "gemini-3-flash-preview",
            "GEMINI_API_KEY": "test-key",
        }):
            settings = DeepSearchSettings.load()
        assert settings.gemini_model == DEFAULT_GEMINI_MODEL


# ---------------------------------------------------------------------------
# build_reflection_llm_config
# ---------------------------------------------------------------------------


class TestBuildReflectionLLMConfig:
    """Verify build_reflection_llm_config returns correct config."""

    def test_returns_reflection_model(self) -> None:
        """Config should use reflection_gemini_model, not gemini_model."""
        settings = DeepSearchSettings(llm_backend="gemini", gemini_api_key="test-key")
        config = settings.build_reflection_llm_config()
        assert config.model == "gemini-2.5-flash"
        assert config.backend == "gemini"

    def test_research_config_still_uses_pro(self) -> None:
        """build_llm_config should still use gemini_model (pro)."""
        settings = DeepSearchSettings(llm_backend="gemini", gemini_api_key="test-key")
        research_config = settings.build_llm_config()
        reflection_config = settings.build_reflection_llm_config()
        assert research_config.model == "gemini-2.5-pro"
        assert reflection_config.model == "gemini-2.5-flash"

    def test_research_config_uses_extraction_budget(self) -> None:
        """Research config should default to the lower extraction thinking budget."""
        settings = DeepSearchSettings(llm_backend="gemini", gemini_api_key="test-key")
        config = settings.build_llm_config()
        assert config.thinking_budget == 2048

    def test_synthesis_config_uses_synthesis_budget(self) -> None:
        """Synthesis config should retain the higher synthesis thinking budget."""
        settings = DeepSearchSettings(llm_backend="gemini", gemini_api_key="test-key")
        config = settings.build_llm_config(call_type="synthesis")
        assert config.thinking_budget == 16384

    def test_shares_api_key(self) -> None:
        """Reflection config should use the same API key."""
        settings = DeepSearchSettings(llm_backend="gemini", gemini_api_key="shared-key")
        config = settings.build_reflection_llm_config()
        assert config.api_key == "shared-key"

    def test_shares_temperature(self) -> None:
        """Reflection config should use the same temperature."""
        settings = DeepSearchSettings(llm_backend="gemini",
            gemini_api_key="key",
            gemini_temperature=0.3,
        )
        config = settings.build_reflection_llm_config()
        assert config.temperature == 0.3

    def test_shares_max_output_tokens(self) -> None:
        """Reflection config should use the same max_output_tokens."""
        settings = DeepSearchSettings(llm_backend="gemini",
            gemini_api_key="key",
            gemini_max_output_tokens=4000,
        )
        config = settings.build_reflection_llm_config()
        assert config.max_output_tokens == 4000

    def test_reflection_config_uses_reflection_budget(self) -> None:
        """Reflection config should default to zero thinking budget."""
        settings = DeepSearchSettings(llm_backend="gemini", gemini_api_key="key")
        config = settings.build_reflection_llm_config()
        assert config.thinking_budget == 0

    def test_ollama_backend_uses_same_config(self) -> None:
        """Ollama backend should return the same config for both methods."""
        settings = DeepSearchSettings(llm_backend="ollama")
        research = settings.build_llm_config()
        reflection = settings.build_reflection_llm_config()
        assert research.model == reflection.model
        assert research.backend == reflection.backend == "ollama"

    def test_custom_reflection_model_in_config(self) -> None:
        """Custom reflection model should be returned in config."""
        settings = DeepSearchSettings(llm_backend="gemini",
            gemini_api_key="key",
            reflection_gemini_model="gemini-3.1-flash-lite-preview",
        )
        config = settings.build_reflection_llm_config()
        assert config.model == "gemini-3.1-flash-lite-preview"


