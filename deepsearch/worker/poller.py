"""Mission poller for fetching queued missions from TraceLab PostgreSQL.

This module implements polling logic to fetch queued missions with proper
row-level locking to prevent duplicate processing in multi-worker scenarios.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Optional

from .database import DatabaseClient
from .lease import LeaseAttempt, LeasePolicy

logger = logging.getLogger(__name__)


def _decode_jsonb(value: Any) -> Any:
    """Decode an asyncpg JSONB value while preserving native JSON values."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _coerce_entity_list(value: Any) -> Optional[list[str]]:
    """Normalize a JSONB entity column into list[str].

    TraceLab persists ``required_entities`` / ``excluded_entities`` as JSONB
    arrays of strings. asyncpg may surface them as a Python list or as a raw
    JSON string depending on driver configuration. Returns ``None`` for any
    shape that cannot be coerced so legacy rows do not poison the contract
    compiler's authoritative-entity path.
    """
    value = _decode_jsonb(value)
    if not isinstance(value, list):
        return None
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    return cleaned


def _coerce_reference_list(value: Any) -> Optional[list[dict[str, Any]]]:
    """Normalize the TraceLab reference JSONB column without losing ``[]``."""
    value = _decode_jsonb(value)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        return None
    return [dict(item) for item in value]


def _coerce_json_object(value: Any) -> Optional[dict[str, Any]]:
    """Normalize a JSONB object surfaced natively or as a JSON string."""
    value = _decode_jsonb(value)
    return dict(value) if isinstance(value, dict) else None


@dataclass
class MissionRow:
    """Represents a mission row from the TraceLab missions table.

    Attributes:
        id: Unique mission identifier (UUID)
        mission_id: Human-readable mission ID (e.g., "B17.1")
        title: Mission title/name
        objective: Research objective text
        context: Mission context (background, constraints, etc.)
        status: Current status (queued, in_progress, completed, blocked)
        queued_at: Timestamp when mission was queued
        success_criteria: Optional list of success criteria
        deliverables: Optional list of deliverable requirements
        research_phases: Optional phase configuration
        research_depth: Optional depth tier (baseline/deep/alpha)
        background: Optional authored background prose.
        focus: Optional authored scope focus.
        references: Optional authored seed-reference objects.
        required_entities: Optional declared entities authors anchor the
            contract on (sibling top-level column on TraceLab's missions
            table; persisted as JSONB array of strings).
        excluded_entities: Optional declared exclusions paired with the
            same column shape as required_entities.
        expected_output_schema: Optional authored deliverable schema.
        coverage_thresholds: Optional authored coverage gate overrides.
        validation_thresholds: Optional authored validation gate overrides.
        deliverable_format: Optional authored output format.
        max_loops: Optional authored maximum execution loop budget.
        min_loops: Optional authored minimum execution loop budget.
        constraints: Optional authored constraint strings.
        lease: Opaque lease-v2 ownership proof for this claim attempt.
    """
    id: str
    mission_id: str
    title: str
    objective: str
    context: dict[str, Any]
    status: str
    queued_at: datetime
    lease: LeaseAttempt
    success_criteria: Optional[list[str]] = None
    deliverables: Optional[list[str]] = None
    research_phases: Optional[dict[str, Any]] = None
    research_depth: Optional[str] = None
    background: Optional[str] = None
    focus: Optional[str] = None
    references: Optional[list[dict[str, Any]]] = None
    required_entities: Optional[list[str]] = None
    excluded_entities: Optional[list[str]] = None
    expected_output_schema: Optional[dict[str, Any]] = None
    coverage_thresholds: Optional[dict[str, Any]] = None
    validation_thresholds: Optional[dict[str, Any]] = None
    deliverable_format: Optional[str] = None
    max_loops: Optional[int] = None
    min_loops: Optional[int] = None
    constraints: Optional[list[str]] = None


class PollerError(Exception):
    """Base exception for poller operations."""
    pass


class PollerStoppedError(PollerError):
    """Raised when attempting operations on a stopped poller."""
    pass


class MissionPoller:
    """Polls for queued missions from TraceLab's PostgreSQL database.

    Claims span every TraceLab project; only rows carrying a project survive
    the orphan guard. Uses FOR UPDATE SKIP LOCKED to safely claim missions in
    multi-worker environments without race conditions.

    Usage:
        policy = LeasePolicy(120, 30, 3)
        async with MissionPoller(
            db_client,
            lease_owner=worker_id,
            lease_policy=policy,
        ) as poller:
            mission = await poller.poll_once()
            if mission:
                await process_mission(mission)

    Or with continuous polling:
        async def handle_mission(mission: MissionRow) -> None:
            # Process the mission
            pass

        poller = MissionPoller(
            db_client,
            lease_owner=worker_id,
            lease_policy=policy,
            on_mission=handle_mission,
        )
        await poller.start()
    """

    # Default poll interval in seconds
    DEFAULT_POLL_INTERVAL = 5.0

    # Maximum jitter as fraction of poll interval (0.1 = 10%)
    MAX_JITTER_RATIO = 0.1

    # Atomic lease-v2 claim. The row lock, status transition, ownership token,
    # expiry, and attempt increment occur in one statement. Expired in-progress
    # rows are eligible for bounded recovery; completed rows are never selected.
    # T42.2 lockstep with TraceLab PR #213 (alembic 028 DROP COLUMN
    # research_depth): the column is dropped end-to-end on TraceLab's side,
    # so the worker SELECT no longer requests it. ``MissionRow.research_depth``
    # is retained as ``None``-default for backwards compat with the converter
    # downstream until S76 finishes the depth.py scrub.
    #
    # Claims are NOT scoped to a single TraceLab project: missions are authored
    # across every project and visibility is RBAC's concern, not the worker's.
    # ``project_id IS NOT NULL`` is the orphan guard - the result is written
    # back onto this same row, so a row without a project would strand its
    # report outside any project.
    CLAIM_MISSION_SQL = """
        WITH candidate AS (
            SELECT id
            FROM missions
            WHERE project_id IS NOT NULL
              AND COALESCE(deepsearch_attempt_count, 0) < $4
              AND (
                  status = 'queued'
                  OR (
                      status = 'in_progress'
                      AND (
                          deepsearch_lease_token IS NULL
                          OR (
                              deepsearch_lease_expires_at IS NOT NULL
                              AND deepsearch_lease_expires_at
                                  <= NOW() - ($5 * INTERVAL '1 second')
                          )
                      )
                  )
              )
            ORDER BY queued_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        UPDATE missions AS mission
        SET
            status = 'in_progress',
            started_at = COALESCE(mission.started_at, NOW()),
            updated_at = NOW(),
            deepsearch_lease_owner = $1,
            deepsearch_lease_token = $2,
            deepsearch_leased_at = NOW(),
            deepsearch_heartbeat_at = NOW(),
            deepsearch_lease_expires_at = NOW() + ($3 * INTERVAL '1 second'),
            deepsearch_attempt_count =
                COALESCE(mission.deepsearch_attempt_count, 0) + 1
        FROM candidate
        WHERE mission.id = candidate.id
        RETURNING
            mission.id,
            mission.mission_id,
            mission.title,
            mission.objective,
            mission.context,
            mission.status,
            mission.queued_at,
            mission.success_criteria,
            mission.deliverables,
            mission.research_phases,
            mission.background,
            mission.focus,
            mission."references",
            mission.required_entities,
            mission.excluded_entities,
            mission.expected_output_schema,
            mission.coverage_thresholds,
            mission.validation_thresholds,
            mission.deliverable_format,
            mission.max_loops,
            mission.min_loops,
            mission.constraints,
            mission.deepsearch_lease_owner,
            mission.deepsearch_lease_token,
            mission.deepsearch_leased_at,
            mission.deepsearch_heartbeat_at,
            mission.deepsearch_lease_expires_at,
            mission.deepsearch_attempt_count
    """

    # S90 (#952) — the sweep must cover exhausted 'queued' rows as well as
    # 'in_progress' ones. CLAIM_MISSION_SQL caps attempts on BOTH statuses, so
    # a row sitting in 'queued' at or past max_attempts was claimed by nobody
    # and swept by nobody: an invisible zombie with every health signal green.
    # That had 0 rows in production as of 2026-07-29, but S90's transient
    # requeue (#959) returns missions to 'queued' automatically, which makes
    # the trap reachable by machine rather than only by a hand re-queue.
    #
    # The lease-expiry condition applies only to 'in_progress' — a queued row
    # holds no lease, so requiring an expired one would exclude every row this
    # clause exists to catch.
    DEAD_LETTER_EXHAUSTED_SQL = """
        UPDATE missions
        SET
            status = 'blocked',
            error_message = 'DeepSearch lease attempts exhausted',
            updated_at = NOW(),
            deepsearch_lease_token = NULL,
            deepsearch_lease_expires_at = NULL
        WHERE project_id IS NOT NULL
          AND COALESCE(deepsearch_attempt_count, 0) >= $1
          AND (
              status = 'queued'
              OR (
                  status = 'in_progress'
                  AND (
                      deepsearch_lease_token IS NULL
                      OR (
                          deepsearch_lease_expires_at IS NOT NULL
                          AND deepsearch_lease_expires_at
                              <= NOW() - ($2 * INTERVAL '1 second')
                      )
                  )
              )
          )
    """

    HEARTBEAT_SQL = """
        UPDATE missions
        SET
            deepsearch_heartbeat_at = NOW(),
            deepsearch_lease_expires_at = NOW() + ($5 * INTERVAL '1 second'),
            updated_at = NOW()
        WHERE id = $1
          AND status = 'in_progress'
          AND deepsearch_lease_owner = $2
          AND deepsearch_lease_token = $3
          AND deepsearch_attempt_count = $4
          AND deepsearch_lease_expires_at > NOW()
        RETURNING deepsearch_heartbeat_at
    """

    RELEASE_SQL = """
        UPDATE missions
        SET
            deepsearch_lease_owner = NULL,
            deepsearch_lease_token = NULL,
            deepsearch_leased_at = NULL,
            deepsearch_heartbeat_at = NULL,
            deepsearch_lease_expires_at = NULL,
            updated_at = NOW()
        WHERE id = $1
          AND status = 'in_progress'
          AND deepsearch_lease_owner = $2
          AND deepsearch_lease_token = $3
          AND deepsearch_attempt_count = $4
        RETURNING id
    """

    def __init__(
        self,
        db_client: DatabaseClient,
        *,
        lease_owner: str,
        lease_policy: LeasePolicy,
        poll_interval: Optional[float] = None,
        on_mission: Optional[Callable[[MissionRow], Awaitable[None]]] = None,
        enable_jitter: bool = True,
    ) -> None:
        """Initialize the mission poller.

        Args:
            db_client: Connected DatabaseClient instance
            lease_owner: Stable identity for this worker process
            lease_policy: Preflight-validated timing and retry bounds
            poll_interval: Seconds between poll cycles (default from env or 5s)
            on_mission: Async callback invoked when a mission is claimed
            enable_jitter: Add random jitter to prevent thundering herd
        """
        self.db_client = db_client
        if not lease_owner or not lease_policy.valid:
            raise ValueError("MissionPoller requires validated lease-v2 configuration")
        self.lease_owner = lease_owner
        self.lease_policy = lease_policy
        self.poll_interval = poll_interval or self._get_poll_interval_from_env()
        self.on_mission = on_mission
        self.enable_jitter = enable_jitter

        self._running = False
        self._stop_event = asyncio.Event()
        self._poll_task: Optional[asyncio.Task[None]] = None
        self._poll_count = 0
        self._missions_claimed = 0
        self._exhaustion_sweeps = 0
        self._heartbeat_runtime: dict[
            tuple[str, str, int],
            dict[str, Any],
        ] = {}

    @staticmethod
    def _get_poll_interval_from_env() -> float:
        """Get poll interval from WORKER_POLL_INTERVAL env var."""
        try:
            return float(os.getenv("WORKER_POLL_INTERVAL", MissionPoller.DEFAULT_POLL_INTERVAL))
        except ValueError:
            logger.warning(
                "Invalid WORKER_POLL_INTERVAL, using default: %s",
                MissionPoller.DEFAULT_POLL_INTERVAL
            )
            return MissionPoller.DEFAULT_POLL_INTERVAL

    def _calculate_sleep_duration(self) -> float:
        """Calculate sleep duration with optional jitter."""
        base = self.poll_interval

        if not self.enable_jitter:
            return base

        # Add jitter up to MAX_JITTER_RATIO of the interval
        max_jitter = base * self.MAX_JITTER_RATIO
        jitter = random.uniform(-max_jitter, max_jitter)

        return max(0.1, base + jitter)  # Minimum 100ms

    @staticmethod
    def _parse_mission_row(record: Any) -> MissionRow:
        """Parse a database record into a MissionRow.

        Args:
            record: asyncpg.Record from the database

        Returns:
            Parsed MissionRow instance
        """
        # asyncpg can surface JSONB either decoded or as raw JSON text,
        # depending on codec configuration.
        context = _decode_jsonb(record["context"])
        if not isinstance(context, dict):
            context = {}

        # Handle success_criteria - may be JSON array or None
        success_criteria = _decode_jsonb(record.get("success_criteria"))

        # Handle research_phases - may be JSON or dict
        research_phases = _decode_jsonb(record.get("research_phases"))

        deliverables = _decode_jsonb(record.get("deliverables"))

        # T42.2 lockstep: research_depth dropped from CLAIM_MISSION_SQL; the
        # parse path is gone with it. MissionRow.research_depth defaults to
        # None and converter.py falls through to ResearchDepth.BASELINE.

        required_entities = _coerce_entity_list(record.get("required_entities"))
        excluded_entities = _coerce_entity_list(record.get("excluded_entities"))
        constraints = _coerce_entity_list(record.get("constraints"))
        background = record.get("background")
        focus = record.get("focus")
        deliverable_format = record.get("deliverable_format")
        max_loops = record.get("max_loops")
        min_loops = record.get("min_loops")

        return MissionRow(
            id=str(record["id"]),
            mission_id=record["mission_id"] or str(record["id"])[:8],
            title=record["title"],
            objective=record["objective"] or "",
            context=context if isinstance(context, dict) else {},
            status=record["status"],
            queued_at=record["queued_at"],
            lease=LeaseAttempt.from_record(record),
            success_criteria=success_criteria if isinstance(success_criteria, list) else None,
            deliverables=deliverables if isinstance(deliverables, list) else None,
            research_phases=research_phases if isinstance(research_phases, dict) else None,
            background=background if isinstance(background, str) else None,
            focus=focus if isinstance(focus, str) else None,
            references=_coerce_reference_list(record.get("references")),
            required_entities=required_entities,
            excluded_entities=excluded_entities,
            expected_output_schema=_coerce_json_object(
                record.get("expected_output_schema")
            ),
            coverage_thresholds=_coerce_json_object(
                record.get("coverage_thresholds")
            ),
            validation_thresholds=_coerce_json_object(
                record.get("validation_thresholds")
            ),
            deliverable_format=(
                deliverable_format if isinstance(deliverable_format, str) else None
            ),
            max_loops=max_loops if type(max_loops) is int else None,
            min_loops=min_loops if type(min_loops) is int else None,
            constraints=constraints,
        )

    async def poll_once(self) -> Optional[MissionRow]:
        """Poll for a single queued mission.

        Atomically claims the oldest queued or expired recoverable mission.
        If a mission is found, its status and lease-v2 ownership are returned.

        Returns:
            MissionRow if a mission was claimed, None if queue is empty

        Raises:
            PollerError: If database operation fails
        """
        self._poll_count += 1

        logger.debug("Poll cycle #%d: checking for queued missions", self._poll_count)

        try:
            # The claim UPDATE is atomic; the transaction also orders the
            # idempotent retry-exhaustion sweep immediately before it.
            conn = await self.db_client.acquire()
            try:
                async with conn.transaction():
                    await conn.execute(
                        self.DEAD_LETTER_EXHAUSTED_SQL,
                        self.lease_policy.max_attempts,
                        self.lease_policy.reclaim_grace_seconds,
                    )
                    self._exhaustion_sweeps += 1
                    token = secrets.token_urlsafe(32)
                    record = await conn.fetchrow(
                        self.CLAIM_MISSION_SQL,
                        self.lease_owner,
                        token,
                        self.lease_policy.lease_seconds,
                        self.lease_policy.max_attempts,
                        self.lease_policy.reclaim_grace_seconds,
                    )

                    if record is None:
                        logger.debug("No queued missions found")
                        return None

                    mission = self._parse_mission_row(record)

                    self._missions_claimed += 1
                    logger.info(
                        "Claimed mission %s attempt %d: %s (queued at %s)",
                        mission.id,
                        mission.lease.attempt_count,
                        mission.title,
                        mission.queued_at.isoformat()
                    )

                    return mission
            finally:
                await self.db_client.release(conn)

        except Exception as e:
            logger.error("Failed to poll for missions: %s", str(e))
            raise PollerError(f"Poll failed: {e}") from e

    async def heartbeat(self, mission_id: str, lease: LeaseAttempt) -> bool:
        """Extend one active lease; stale or expired ownership returns false."""

        try:
            heartbeat_at = await self.db_client.fetchval(
                self.HEARTBEAT_SQL,
                mission_id,
                lease.owner,
                lease.token,
                lease.attempt_count,
                self.lease_policy.lease_seconds,
            )
            if isinstance(heartbeat_at, datetime):
                key = (mission_id, lease.token, lease.attempt_count)
                previous = self._heartbeat_runtime.get(key, {})
                lease_seconds = int(self.lease_policy.lease_seconds or 0)
                self._heartbeat_runtime[key] = {
                    "snapshot_phase": "pre_terminal",
                    "heartbeat_count": int(previous.get("heartbeat_count") or 0) + 1,
                    "last_confirmed_heartbeat_at": heartbeat_at.isoformat(),
                    "last_confirmed_expires_at": (
                        heartbeat_at + timedelta(seconds=lease_seconds)
                    ).isoformat(),
                }
            return heartbeat_at is not None
        except Exception as exc:
            raise PollerError(f"Heartbeat failed: {exc}") from exc

    def heartbeat_audit(
        self,
        mission_id: str,
        lease: LeaseAttempt,
    ) -> dict[str, Any]:
        """Return the last database-confirmed heartbeat for one attempt."""

        return dict(
            self._heartbeat_runtime.get(
                (mission_id, lease.token, lease.attempt_count),
                {
                    "snapshot_phase": "pre_terminal",
                    "heartbeat_count": 0,
                    "last_confirmed_heartbeat_at": None,
                    "last_confirmed_expires_at": None,
                },
            )
        )

    def clear_heartbeat_audit(self, mission_id: str, lease: LeaseAttempt) -> None:
        """Forget completed attempt telemetry after the worker persists it."""

        self._heartbeat_runtime.pop(
            (mission_id, lease.token, lease.attempt_count),
            None,
        )

    async def release(self, mission_id: str, lease: LeaseAttempt) -> bool:
        """Clear a dead attempt's ownership so recovery bypasses crash grace."""

        try:
            result = await self.db_client.fetchval(
                self.RELEASE_SQL,
                mission_id,
                lease.owner,
                lease.token,
                lease.attempt_count,
            )
            if result is not None:
                self.clear_heartbeat_audit(mission_id, lease)
            return result is not None
        except Exception as exc:
            raise PollerError(f"Lease release failed: {exc}") from exc

    async def start(self) -> None:
        """Start the continuous polling loop.

        Runs until stop() is called or an unrecoverable error occurs.
        If on_mission callback is set, it will be invoked for each claimed mission.
        """
        if self._running:
            logger.warning("Poller is already running")
            return

        self._running = True
        self._stop_event.clear()

        logger.info(
            "Starting mission poller (interval: %.1fs, jitter: %s)",
            self.poll_interval,
            "enabled" if self.enable_jitter else "disabled"
        )

        try:
            while not self._stop_event.is_set():
                try:
                    mission = await self.poll_once()

                    if mission and self.on_mission:
                        try:
                            await self.on_mission(mission)
                        except Exception as e:
                            logger.error(
                                "Mission handler failed for %s: %s",
                                mission.id,
                                str(e)
                            )

                except PollerError as e:
                    logger.warning("Poll error (will retry): %s", str(e))

                # Wait before next poll
                sleep_duration = self._calculate_sleep_duration()
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=sleep_duration
                    )
                    # If we get here, stop was called
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, continue polling
                    pass

        finally:
            self._running = False
            logger.info(
                "Poller stopped (polls: %d, missions claimed: %d)",
                self._poll_count,
                self._missions_claimed
            )

    async def stop(self) -> None:
        """Stop the polling loop gracefully."""
        if not self._running:
            return

        logger.info("Stopping mission poller...")
        self._stop_event.set()

        # Wait for the poll loop to exit
        if self._poll_task and not self._poll_task.done():
            try:
                await asyncio.wait_for(self._poll_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Poller did not stop gracefully, cancelling")
                self._poll_task.cancel()

    @property
    def is_running(self) -> bool:
        """Check if poller is currently running."""
        return self._running

    @property
    def stats(self) -> dict[str, Any]:
        """Get poller statistics."""
        return {
            "running": self._running,
            "poll_count": self._poll_count,
            "missions_claimed": self._missions_claimed,
            "exhaustion_sweeps": self._exhaustion_sweeps,
            "poll_interval": self.poll_interval,
            "jitter_enabled": self.enable_jitter,
        }

    async def __aenter__(self) -> "MissionPoller":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()
