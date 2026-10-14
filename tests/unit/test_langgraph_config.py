"""Unit tests for LangGraph runtime configuration helpers."""

from __future__ import annotations

from deepsearch.agent.checkpointer import calculate_recursion_limit, thread_config
from deepsearch.config import DeepSearchSettings


def test_thread_config_includes_recursion_limit_when_provided() -> None:
    config = thread_config("deepsearch-test", recursion_limit=99)
    assert config["configurable"]["thread_id"] == "deepsearch-test"
    assert config["recursion_limit"] == 99


def test_calculate_recursion_limit_scales_with_max_loops() -> None:
    assert calculate_recursion_limit(1) >= 25
    assert calculate_recursion_limit(20) >= calculate_recursion_limit(10)


def test_generate_thread_id_accepts_execution_token() -> None:
    settings = DeepSearchSettings()

    assert settings.generate_thread_id("DRM.0.5", execution_token="job-1234") == (
        "deepsearch-drm-0-5-job-1234"
    )
    assert settings.generate_thread_id("DRM.0.5", execution_token="job-5678") == (
        "deepsearch-drm-0-5-job-5678"
    )
