"""Helpers for configuring and introspecting LangGraph checkpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Mapping, Optional

from .concept_clustering import cluster_findings
from .finding import coerce_state_finding_records, project_finding_summaries

try:  # pragma: no cover - optional dependency validated at runtime
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ImportError:  # pragma: no cover - dependency missing
    AsyncSqliteSaver = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from langgraph.checkpoint.base import CheckpointTuple, RunnableConfig
else:  # pragma: no cover - fallback for runtime typing
    CheckpointTuple = Any  # type: ignore[assignment]
    RunnableConfig = Dict[str, Any]  # type: ignore[assignment]


class CheckpointerUnavailableError(RuntimeError):
    """Raised when LangGraph checkpoint dependencies are missing."""


def calculate_recursion_limit(
    max_loops: int,
    *,
    minimum: int = 40,
    buffer_steps: int = 10,
) -> int:
    """Return a safe LangGraph recursion limit for the workflow.

    LangGraph counts graph node executions (not research loops) toward
    ``recursion_limit``. The DeepSearch workflow executes roughly three nodes per
    loop (research → cluster_findings → reflection), plus finalization and bookkeeping.

    Args:
        max_loops: Maximum number of research loops configured for the mission.
        minimum: Minimum recursion limit to return (LangGraph default is 25).
        buffer_steps: Extra steps to allow for finalization/overhead.

    Returns:
        Integer recursion limit suitable for passing via ``RunnableConfig``.
    """

    try:
        loops = int(max_loops)
    except (TypeError, ValueError):
        loops = 1
    loops = max(loops, 1)
    estimated_steps = loops * 3 + max(buffer_steps, 0)
    return max(minimum, estimated_steps)


def _require_async_saver() -> type[AsyncSqliteSaver]:
    if AsyncSqliteSaver is None:
        raise CheckpointerUnavailableError(
            "LangGraph checkpoint extras are not installed. Install "
            "`langgraph-checkpoint-sqlite` to enable persistence."
        )
    return AsyncSqliteSaver


@asynccontextmanager
async def create_checkpointer(db_path: Path | str) -> AsyncIterator[AsyncSqliteSaver]:
    """Yield an AsyncSqliteSaver backed by ``db_path``."""

    saver_cls = _require_async_saver()
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    async with saver_cls.from_conn_string(str(path)) as saver:
        yield saver


def thread_config(
    thread_id: str,
    checkpoint_ns: str | None = None,
    *,
    recursion_limit: int | None = None,
) -> RunnableConfig:
    """Return a LangGraph config payload for a specific thread."""

    namespace = checkpoint_ns or ""
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": namespace,
        }
    }
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    return config


async def clear_thread_checkpoints(
    thread_id: str,
    *,
    checkpointer: AsyncSqliteSaver | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Delete all checkpoints and writes for ``thread_id``.

    Note: Some checkpointer versions may not implement adelete_thread.
    In that case, the operation is silently skipped.
    """
    import logging
    logger = logging.getLogger(__name__)

    async def _try_delete(saver: AsyncSqliteSaver) -> None:
        try:
            await saver.adelete_thread(thread_id)
        except NotImplementedError:
            logger.debug("Checkpointer does not support adelete_thread, skipping clear")

    if checkpointer is not None:
        await _try_delete(checkpointer)
        return
    async with create_checkpointer(_resolve_db_path(db_path)) as saver:
        await _try_delete(saver)


async def get_checkpoint_history(
    thread_id: str,
    *,
    checkpointer: AsyncSqliteSaver | None = None,
    db_path: Path | str | None = None,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """Return a list of checkpoint metadata dictionaries for ``thread_id``."""

    if checkpointer is not None:
        return await _history_with_saver(checkpointer, thread_id, limit=limit)
    async with create_checkpointer(_resolve_db_path(db_path)) as saver:
        return await _history_with_saver(saver, thread_id, limit=limit)


async def get_latest_checkpoint(
    thread_id: str,
    *,
    checkpointer: AsyncSqliteSaver | None = None,
    db_path: Path | str | None = None,
) -> Optional[Dict[str, Any]]:
    """Return the newest checkpoint tuple for ``thread_id`` if present."""

    if checkpointer is not None:
        checkpoint = await checkpointer.aget_tuple(thread_config(thread_id))
        return _serialize_checkpoint(checkpoint) if checkpoint else None
    async with create_checkpointer(_resolve_db_path(db_path)) as saver:
        checkpoint = await saver.aget_tuple(thread_config(thread_id))
        return _serialize_checkpoint(checkpoint) if checkpoint else None


async def list_all_threads(
    *,
    checkpointer: AsyncSqliteSaver | None = None,
    db_path: Path | str | None = None,
) -> List[str]:
    """Return the set of thread IDs observed in the checkpoint database."""

    if checkpointer is not None:
        return await _list_threads_with_saver(checkpointer)
    async with create_checkpointer(_resolve_db_path(db_path)) as saver:
        return await _list_threads_with_saver(saver)


def _resolve_db_path(db_path: Path | str | None) -> Path:
    if db_path is None:
        raise ValueError("db_path must be supplied when checkpointer is not provided.")
    return Path(db_path)


async def _history_with_saver(
    saver: AsyncSqliteSaver,
    thread_id: str,
    *,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    config = thread_config(thread_id)
    history: List[Dict[str, Any]] = []
    async for checkpoint in saver.alist(config, limit=limit):
        history.append(_serialize_checkpoint(checkpoint))
    return history


async def _list_threads_with_saver(saver: AsyncSqliteSaver) -> List[str]:
    await saver.setup()
    async with saver.lock, saver.conn.cursor() as cursor:
        await cursor.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id ASC"
        )
        rows = await cursor.fetchall()
    return [row[0] for row in rows]


def normalize_checkpoint_state(state: Any) -> Any:
    """Project legacy checkpoint state into the post-S77 shape."""

    if not isinstance(state, Mapping):
        return state
    normalized = dict(state)
    if "findings" in state or "finding_records" in state:
        finding_records = coerce_state_finding_records(normalized)
        tagged_records, concept_clusters = cluster_findings(finding_records)
        normalized["finding_records"] = [record.to_dict() for record in tagged_records]
        normalized["concept_clusters"] = [cluster.to_dict() for cluster in concept_clusters]
        normalized["findings"] = project_finding_summaries(tagged_records)
    return normalized


def normalize_checkpoint_payload(payload: Any) -> Any:
    """Normalize serialized checkpoint payloads for legacy-state inspection."""

    if not isinstance(payload, Mapping):
        return payload

    normalized = dict(payload)
    channel_values = payload.get("channel_values")
    if isinstance(channel_values, Mapping):
        normalized["channel_values"] = normalize_checkpoint_state(channel_values)
    return normalized


def _serialize_checkpoint(checkpoint: CheckpointTuple | None) -> Dict[str, Any]:
    if checkpoint is None:
        return {}
    config = checkpoint.config.get("configurable", {})
    parent_config = (checkpoint.parent_config or {}).get("configurable", {})
    return {
        "thread_id": config.get("thread_id"),
        "checkpoint_id": config.get("checkpoint_id") or (checkpoint.checkpoint or {}).get("id"),
        "parent_checkpoint_id": parent_config.get("checkpoint_id"),
        "metadata": checkpoint.metadata or {},
        "checkpoint": normalize_checkpoint_payload(checkpoint.checkpoint),
        "pending_writes": checkpoint.pending_writes or [],
    }


__all__ = [
    "CheckpointerUnavailableError",
    "calculate_recursion_limit",
    "create_checkpointer",
    "thread_config",
    "clear_thread_checkpoints",
    "get_checkpoint_history",
    "get_latest_checkpoint",
    "list_all_threads",
    "normalize_checkpoint_payload",
    "normalize_checkpoint_state",
]
