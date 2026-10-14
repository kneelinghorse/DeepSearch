"""Unit tests for checkpoint utilities and configuration helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from deepsearch.agent.checkpointer import (
    clear_thread_checkpoints,
    create_checkpointer,
    get_checkpoint_history,
    list_all_threads,
    thread_config,
)
from deepsearch.config import DeepSearchSettings

_checkpoint_module = pytest.importorskip(
    "langgraph.checkpoint.base",
    reason="LangGraph is required for checkpoint tests",
)
Checkpoint = _checkpoint_module.Checkpoint


def _checkpoint_payload(identifier: str) -> Checkpoint:
    return {
        "id": identifier,
        "v": 1,
        "ts": "2025-01-01T00:00:00Z",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "updated_channels": [],
    }


def test_thread_id_generation_is_deterministic() -> None:
    """Thread IDs should be deterministic and sanitized."""

    settings = DeepSearchSettings()
    assert settings.generate_thread_id("DRM.0.5") == "deepsearch-drm-0-5"
    assert settings.generate_thread_id("DRM.0.5") == settings.generate_thread_id("DRM.0.5")


def test_thread_config_assigns_namespace() -> None:
    """Thread config should carry namespaces through to LangGraph."""

    config = thread_config("deepsearch-test", checkpoint_ns="ns-1")
    assert config["configurable"]["thread_id"] == "deepsearch-test"
    assert config["configurable"]["checkpoint_ns"] == "ns-1"


@pytest.mark.asyncio
async def test_checkpoint_history_round_trip(tmp_path: Path) -> None:
    """Checkpoints written via AsyncSqliteSaver should be discoverable."""

    db_path = tmp_path / "checkpoints" / "history.db"
    async with create_checkpointer(db_path) as saver:
        config = thread_config("deepsearch-history")

        first = _checkpoint_payload("cp-1")
        second = _checkpoint_payload("cp-2")
        await saver.aput(config, first, {"step": "research"}, {})
        await saver.aput(config, second, {"step": "reflection"}, {})

        history = await get_checkpoint_history("deepsearch-history", checkpointer=saver)

        assert len(history) == 2
        assert history[0]["checkpoint_id"] == "cp-2"
        assert history[1]["checkpoint_id"] == "cp-1"

        await clear_thread_checkpoints("deepsearch-history", checkpointer=saver)
        assert await get_checkpoint_history("deepsearch-history", checkpointer=saver) == []


@pytest.mark.asyncio
async def test_list_all_threads_returns_sorted_ids(tmp_path: Path) -> None:
    """Listing threads should include each thread exactly once."""

    db_path = tmp_path / "checkpoints" / "threads.db"
    async with create_checkpointer(db_path) as saver:
        for identifier in ("thread-b", "thread-a"):
            await saver.aput(thread_config(identifier), _checkpoint_payload(identifier), {}, {})

        threads = await list_all_threads(checkpointer=saver)
        assert threads == ["thread-a", "thread-b"]
