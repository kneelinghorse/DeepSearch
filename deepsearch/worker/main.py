"""DeepSearch worker orchestration for TraceLab mission processing.

This module provides the main DeepSearchWorker class that:
- Polls TraceLab PostgreSQL for queued missions
- Converts mission rows to AgentState
- Executes research using LangGraph workflow
- Writes results back to TraceLab database
- Handles graceful shutdown on SIGTERM/SIGINT
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import threading
import time
import uuid
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, cast
from urllib.parse import urlparse

from deepsearch.agent import graph as agent_graph
from deepsearch.agent.checkpointer import (
    CheckpointerUnavailableError,
    calculate_recursion_limit,
    create_checkpointer,
    thread_config,
)
from deepsearch.agent.codeagent_runner import (
    OPENAI_CLIENT_MAX_RETRIES,
    SCHOLAR_BACKEND_TIMEOUT_S,
    SCHOLAR_MAX_BACKENDS,
    SMOLAGENTS_RATE_LIMIT_RETRY,
    SOURCE_FETCH_MAX_REDIRECTS,
    SOURCE_FETCH_MAX_TIMEOUT_S,
    SYNTHESIS_RETRY_ATTEMPTS,
    URL_LIVENESS_MAX_REQUESTS,
    URL_LIVENESS_MAX_TIMEOUT_S,
)
from deepsearch.agent.evidence import (
    TelemetryCollector,
    TelemetryWriter,
    extract_sources_from_state,
    record_finding_records_from_state,
)
from deepsearch.agent.execution_control import (
    ExecutionCancelledError,
    bind_execution_cancel_event,
    raise_if_execution_cancelled,
)
from deepsearch.agent.outcome import MissionOutcome, resolve_final_outcome_payload
from deepsearch.agent.quality_series import build_quality_record
from deepsearch.agent.state import AgentState
from deepsearch.agent.telemetry import normalize_final_telemetry
from deepsearch.config import DeepSearchSettings
from deepsearch.error_safety import safe_exception_details
from deepsearch.mission import prepare_execution_contract
from deepsearch.observability import apply_runtime_context, build_runtime_context
from deepsearch.report import generate_markdown_report
from deepsearch.report.protocol import generate_mission_protocol
from deepsearch.tracelab import (
    CompletionReceipt,
    CompletionReceiptClient,
    EvidenceLedgerClient,
    TraceLabAuth,
    TracelabLogHandler,
)

from .converter import ConversionError, convert_row_to_state
from .database import DatabaseClient, DatabaseConfig, DatabaseError
from .evidence_outbox import EvidenceOutboxDispatcher, EvidenceOutboxRepository
from .execution import (
    IsolatedExecutionError,
    MissionProcessExecutor,
    ProcessTerminationError,
)
from .health import HealthServer
from .lease import ExecutionFencePolicy, LeaseAttempt, LeasePolicy
from .poller import MissionPoller, MissionRow, PollerError
from .preflight import WorkerPreflightReport, run_worker_preflight
from .writer import (
    ResultIdentityError,
    ResultWriter,
    WriterError,
    assert_result_identity,
    build_execution_metadata,
    protocol_identity_sha256,
)

logger = logging.getLogger(__name__)

_SUPPORTED_LLM_BACKENDS = {"ollama", "gemini", "deepseek", "minimax", "openai"}
_MODEL_ENV_BY_BACKEND = {
    "ollama": "OLLAMA_MODEL",
    "gemini": "GEMINI_MODEL",
    "deepseek": "DEEPSEEK_MODEL",
    "minimax": "MINIMAX_MODEL",
    "openai": "OPENAI_MODEL",
}

MAX_PENDING_LOG_HANDLERS = 10
MAX_LOG_HANDLER_RETRIES_PER_FENCE = 2
MAX_PENDING_COMPLETION_RECEIPTS = 1_000
MAX_RECEIPT_RETRIES_PER_FENCE = 2


def _delivery_health_category(
    *,
    success: bool,
    retryable: bool,
    status_code: int | None,
) -> str:
    """Return an aggregate transport outcome without retaining error detail."""

    if success:
        return "success"
    if status_code == 429:
        return "rate_limited"
    if status_code is not None and status_code >= 500:
        return "server_error"
    if status_code is not None:
        return "client_error"
    return "transport_error" if retryable else "configuration_error"


def _public_completion_receipt_health(
    telemetry: object,
) -> dict[str, Any] | None:
    """Project internal receipt telemetry onto the identity-free health shape."""

    if not isinstance(telemetry, Mapping):
        return None
    raw_status = telemetry.get("status_code")
    status_code = (
        raw_status
        if isinstance(raw_status, int) and not isinstance(raw_status, bool)
        else None
    )
    raw_attempts = telemetry.get("attempts")
    attempts = (
        max(0, raw_attempts)
        if isinstance(raw_attempts, int) and not isinstance(raw_attempts, bool)
        else 0
    )
    success = telemetry.get("success") is True
    retryable = telemetry.get("retryable") is True
    return {
        "category": _delivery_health_category(
            success=success,
            retryable=retryable,
            status_code=status_code,
        ),
        "status_code": status_code,
        "attempts": attempts,
    }


def _public_log_delivery_health(telemetry: object) -> dict[str, Any] | None:
    """Return log-delivery counters without its retained exception summary."""

    if not isinstance(telemetry, Mapping):
        return None
    public = {
        key: telemetry.get(key)
        for key in (
            "configured",
            "accepted_entries",
            "pending_entries",
            "delivery_attempts",
            "failed_batches",
            "overflowed_entries",
            "discarded_entries",
            "last_status_code",
        )
    }
    raw_status = telemetry.get("last_status_code")
    status_code = (
        raw_status
        if isinstance(raw_status, int) and not isinstance(raw_status, bool)
        else None
    )
    has_attempt = bool(public["delivery_attempts"])
    public["last_category"] = (
        _delivery_health_category(
            success=status_code is not None and 200 <= status_code < 300,
            retryable=(status_code is None or status_code == 429 or status_code >= 500),
            status_code=status_code,
        )
        if has_attempt
        else None
    )
    return public


def _default_model_for_backend(settings: DeepSearchSettings, backend: str) -> str:
    if backend == "openai":
        return settings.openai_model
    if backend == "gemini":
        return settings.gemini_model
    if backend == "deepseek":
        return settings.deepseek_model
    if backend == "minimax":
        return settings.minimax_model
    return settings.ollama_model


def _lease_int_setting(
    env_name: str,
    default: int,
    override: int | None,
) -> int | None:
    """Preserve invalid lease configuration for fail-closed preflight."""

    if override is not None:
        return override
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return None


def _fence_float_setting(env_name: str, default: float) -> float | None:
    """Preserve invalid execution-fence values for fail-closed preflight."""

    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return None


def _provider_limit_snapshot(
    settings: DeepSearchSettings,
    backend: str,
) -> dict[str, dict[str, Any]]:
    """Return the authoritative provider/retry bind used by this worker."""

    openai_compatible_contract = backend in {"deepseek", "minimax", "openai"}
    gemini_contract = backend == "gemini"
    hard_bound_mode = "spawn-process-v1"
    return {
        "contract": {
            "backend": backend,
            "request_timeout_seconds": (
                getattr(settings, "llm_request_timeout_seconds", 300.0)
                if openai_compatible_contract
                else None
            ),
            "client_max_retries": 0 if openai_compatible_contract else None,
            "configured_max_retries": getattr(settings, "llm_max_retries", 3),
            "wrapper_max_retries": 0 if gemini_contract else 2,
            "wrapper_attempts": 1 if gemini_contract else 3,
            "hard_bound_mode": hard_bound_mode,
        },
        "codeagent": {
            "backend": f"{backend}-openai-compatible",
            "request_timeout_seconds": settings.llm_request_timeout_seconds,
            "client_max_retries": OPENAI_CLIENT_MAX_RETRIES,
            "smolagents_rate_limit_retry": SMOLAGENTS_RATE_LIMIT_RETRY,
            "synthesis_attempts": SYNTHESIS_RETRY_ATTEMPTS,
            "max_step_fallback_attempts": 1,
            "recovery_passes": 1,
            "hard_bound_mode": hard_bound_mode,
        },
        "tools": {
            "configured_request_timeout_seconds": getattr(
                settings,
                "tool_timeout_seconds",
                60.0,
            ),
            "configured_max_retries": getattr(settings, "tool_max_retries", 3),
            "linkup_request_timeout_seconds": 60.0,
            "linkup_max_retries": 3,
            "source_fetch_request_timeout_seconds": SOURCE_FETCH_MAX_TIMEOUT_S,
            "source_fetch_max_redirects": SOURCE_FETCH_MAX_REDIRECTS,
            "url_liveness_request_timeout_seconds": URL_LIVENESS_MAX_TIMEOUT_S,
            "url_liveness_max_requests": URL_LIVENESS_MAX_REQUESTS,
            "scholar_backend_request_timeout_seconds": (
                SCHOLAR_BACKEND_TIMEOUT_S
            ),
            "scholar_max_backends": SCHOLAR_MAX_BACKENDS,
            "hard_bound_mode": hard_bound_mode,
        },
    }


def _database_config_error(config: DatabaseConfig) -> str | None:
    """Return one redacted structural DATABASE_URL error, if any."""

    try:
        parsed = urlparse(config.url)
    except ValueError:
        return "DATABASE_URL is structurally invalid"
    if parsed.scheme not in {"postgres", "postgresql"}:
        return "DATABASE_URL must use the postgres or postgresql scheme"
    if not parsed.hostname:
        return "DATABASE_URL must include a database host"
    if not parsed.path.strip("/"):
        return "DATABASE_URL must include a database name"
    return None


def _read_emitted_text_artifact(path_value: Any, *, label: str) -> str | None:
    """Return a finalize-emitted artifact when it exists on disk."""

    path_text = str(path_value or "").strip()
    if not path_text:
        return None

    path = Path(path_text)
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Finalize-emitted %s artifact missing: %s", label, path)
        return None
    except (OSError, UnicodeError) as exc:
        error_details = safe_exception_details(exc, category="persistence")
        logger.warning(
            "Failed reading finalize-emitted %s artifact %s: %s",
            label,
            path,
            error_details.to_text(),
        )
        return None

    logger.info("Using finalize-emitted %s artifact from %s", label, path)
    return content


def _read_emitted_json_artifact(path_value: Any, *, label: str) -> Dict[str, Any] | None:
    """Return a finalize-emitted JSON artifact when it exists and parses cleanly."""

    content = _read_emitted_text_artifact(path_value, label=label)
    if content is None:
        return None

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        error_details = safe_exception_details(exc)
        logger.warning(
            "Failed parsing finalize-emitted %s artifact %s: %s",
            label,
            path_value,
            error_details.to_text(),
        )
        return None

    if not isinstance(payload, dict):
        logger.warning(
            "Finalize-emitted %s artifact must decode to a JSON object: %s",
            label,
            path_value,
        )
        return None
    return payload


def _read_inline_text_artifact(value: Any, *, label: str) -> str | None:
    """Return a structurally valid finalize-emitted inline text artifact."""

    if value is None:
        return None
    if not isinstance(value, str):
        reason = f"invalid_inline_{label}_artifact"
        raise ResultIdentityError(
            reason,
            diagnostic={
                "artifact_transport": {
                    "artifact": label,
                    "inline_present": True,
                    "inline_type_valid": False,
                },
                "violations": [reason],
            },
        )
    logger.info("Found finalize-emitted inline %s artifact", label)
    return value


def _read_inline_json_artifact(value: Any, *, label: str) -> Dict[str, Any] | None:
    """Return a JSON-safe finalize-emitted inline object artifact."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        reason = f"invalid_inline_{label}_artifact"
        raise ResultIdentityError(
            reason,
            diagnostic={
                "artifact_transport": {
                    "artifact": label,
                    "inline_present": True,
                    "inline_type_valid": False,
                },
                "violations": [reason],
            },
        )
    payload = dict(value)
    try:
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        reason = f"invalid_inline_{label}_artifact"
        raise ResultIdentityError(
            reason,
            diagnostic={
                "artifact_transport": {
                    "artifact": label,
                    "inline_present": True,
                    "inline_json_safe": False,
                },
                "violations": [reason],
            },
        ) from exc
    logger.info("Found finalize-emitted inline %s artifact", label)
    return payload


def _artifact_divergence_error(label: str) -> ResultIdentityError:
    """Return a content-free identity error for disagreeing transports."""

    reason = f"{label}_artifact_divergence"
    return ResultIdentityError(
        reason,
        diagnostic={
            "artifact_transport": {
                "artifact": label,
                "file_present": True,
                "inline_present": True,
                "matches": False,
            },
            "violations": [reason],
        },
    )


class WorkerError(Exception):
    """Base exception for worker operations."""
    pass


class RequiredArtifactError(WorkerError):
    """A modern CodeAgent run lacks its exact finalize-emitted artifact."""


class LeaseLostError(WorkerError):
    """Raised when this process can no longer prove mission ownership."""

    pass


async def _run_isolated_research(
    state: AgentState,
    mission_id: str,
    job_id: str,
    llm_backend: str,
    model: str,
) -> AgentState:
    """Spawn-target adapter that reuses the production research implementation."""

    worker = DeepSearchWorker(
        db_config=DatabaseConfig(
            url="postgresql://isolated-worker@localhost/deepsearch"
        ),
        llm_backend=llm_backend,
        model=model,
        enable_health_server=False,
    )
    return await worker._execute_research(state, mission_id, job_id)


class DeepSearchWorker:
    """Orchestrates mission polling, execution, and result writing.

    The worker is a long-running process that:
    1. Polls for queued missions using MissionPoller
    2. Converts each mission to AgentState
    3. Executes research using LangGraph workflow
    4. Writes results back to TraceLab via ResultWriter
    5. Handles graceful shutdown on SIGTERM/SIGINT

    Usage:
        worker = DeepSearchWorker()
        await worker.run()
    """

    def __init__(
        self,
        db_config: Optional[DatabaseConfig] = None,
        poll_interval: Optional[float] = None,
        llm_backend: str | None = None,
        model: str | None = None,
        health_port: Optional[int] = None,
        enable_health_server: bool = True,
        worker_id: str | None = None,
        lease_seconds: int | None = None,
        heartbeat_seconds: int | None = None,
        max_lease_attempts: int | None = None,
        reclaim_grace_seconds: int | None = None,
        execution_fence_policy: ExecutionFencePolicy | None = None,
        receipt_client: CompletionReceiptClient | None = None,
        tracelab_auth: TraceLabAuth | None = None,
        evidence_ledger_client: EvidenceLedgerClient | None = None,
    ) -> None:
        """Initialize the worker.

        Args:
            db_config: Database configuration (defaults to from environment)
            poll_interval: Seconds between polls (default from env or 5s)
            llm_backend: Optional LLM backend override ("ollama" or "gemini")
            model: Optional model override for the selected backend
            health_port: Port for health HTTP server (default from env or 8080)
            enable_health_server: Whether to start health server (default: True)
            worker_id: Optional stable worker-instance identity override
            lease_seconds: Optional lease duration override
            heartbeat_seconds: Optional heartbeat interval override
            max_lease_attempts: Optional bounded retry override
            reclaim_grace_seconds: Optional crashed-attempt recovery grace
            execution_fence_policy: Optional hard execution-boundary override
            receipt_client: Optional completion-receipt transport override
            tracelab_auth: Optional API-key-first service auth override
            evidence_ledger_client: Optional one-attempt ledger transport override
        """
        settings = DeepSearchSettings.load()
        resolved_backend = (llm_backend or settings.llm_backend).strip().lower() or "deepseek"

        resolved_model = model or _default_model_for_backend(settings, resolved_backend)
        # Preflight and the spawned child must see the same explicit overrides.
        settings.llm_backend = resolved_backend
        model_field = _MODEL_ENV_BY_BACKEND.get(resolved_backend)
        if model_field:
            setattr(settings, model_field.lower(), resolved_model)

        database_config_load_error: str | None = None
        if db_config is None:
            try:
                self.db_config = DatabaseConfig.from_env()
            except DatabaseError:
                # Preserve the invalid state until run() can emit the single
                # structured preflight record before any claim is possible.
                self.db_config = DatabaseConfig(url="")
                database_config_load_error = "DATABASE_URL missing"
            except ValueError:
                self.db_config = DatabaseConfig(url=os.getenv("DATABASE_URL", ""))
                database_config_load_error = (
                    "database pool and timeout settings must be numeric"
                )
        else:
            self.db_config = db_config
        self._database_config_error = (
            database_config_load_error or _database_config_error(self.db_config)
        )
        self.poll_interval = poll_interval or float(
            os.getenv("WORKER_POLL_INTERVAL", "5.0")
        )
        self.llm_backend = resolved_backend
        self.model = resolved_model
        self.health_port = health_port or int(os.getenv("HEALTH_PORT", "8080"))
        self.enable_health_server = enable_health_server
        self.worker_id = (
            worker_id
            or os.getenv("WORKER_INSTANCE_ID")
            or f"deepsearch-{uuid.uuid4()}"
        )
        self.lease_policy = LeasePolicy(
            lease_seconds=_lease_int_setting(
                "WORKER_LEASE_SECONDS", 120, lease_seconds
            ),
            heartbeat_seconds=_lease_int_setting(
                "WORKER_HEARTBEAT_SECONDS", 30, heartbeat_seconds
            ),
            max_attempts=_lease_int_setting(
                "WORKER_MAX_LEASE_ATTEMPTS", 3, max_lease_attempts
            ),
            reclaim_grace_seconds=_lease_int_setting(
                "WORKER_RECLAIM_GRACE_SECONDS", 30, reclaim_grace_seconds
            ),
        )
        provider_limits = _provider_limit_snapshot(settings, resolved_backend)
        if execution_fence_policy is None:
            self.execution_fence_policy = ExecutionFencePolicy(
                heartbeat_timeout_seconds=_fence_float_setting(
                    "WORKER_HEARTBEAT_TIMEOUT_SECONDS", 10.0
                ),
                cancellation_grace_seconds=_fence_float_setting(
                    "WORKER_CANCELLATION_GRACE_SECONDS", 2.0
                ),
                terminate_grace_seconds=_fence_float_setting(
                    "WORKER_TERMINATE_GRACE_SECONDS", 3.0
                ),
                kill_grace_seconds=_fence_float_setting(
                    "WORKER_KILL_GRACE_SECONDS", 3.0
                ),
                safety_margin_seconds=_fence_float_setting(
                    "WORKER_LEASE_SAFETY_MARGIN_SECONDS", 10.0
                ),
                provider_limits=provider_limits,
            )
        else:
            # Timing overrides are testable/configurable, but provider facts
            # are always derived from the actual selected runtime bind.
            self.execution_fence_policy = ExecutionFencePolicy(
                heartbeat_timeout_seconds=(
                    execution_fence_policy.heartbeat_timeout_seconds
                ),
                cancellation_grace_seconds=(
                    execution_fence_policy.cancellation_grace_seconds
                ),
                terminate_grace_seconds=(
                    execution_fence_policy.terminate_grace_seconds
                ),
                kill_grace_seconds=execution_fence_policy.kill_grace_seconds,
                safety_margin_seconds=execution_fence_policy.safety_margin_seconds,
                provider_limits=provider_limits,
                isolation_mode=execution_fence_policy.isolation_mode,
            )
        self._startup_settings = settings
        self._receipt_client = receipt_client or CompletionReceiptClient()
        self._tracelab_auth = tracelab_auth or TraceLabAuth()
        self._evidence_ledger_client = evidence_ledger_client or EvidenceLedgerClient(
            auth=self._tracelab_auth,
            max_attempts=1,
        )

        self._shutdown_requested = False
        self._current_mission_id: Optional[str] = None
        self._current_job_id: Optional[str] = None
        self._db_client: Optional[DatabaseClient] = None
        self._poller: Optional[MissionPoller] = None
        self._writer: Optional[ResultWriter] = None
        self._evidence_outbox_repository: EvidenceOutboxRepository | None = None
        self._evidence_outbox_dispatcher: EvidenceOutboxDispatcher | None = None
        self._health_server: Optional[HealthServer] = None
        self._mission_executor: MissionProcessExecutor | None = None
        self._lease_deadlines_monotonic: dict[tuple[str, str, int], float] = {}
        self._heartbeat_anchors_monotonic: dict[tuple[str, str, int], float] = {}
        self._pending_completion_receipts: dict[CompletionReceipt, bool] = {}
        self._pending_log_handlers: list[TracelabLogHandler] = []
        self._preflight_report: WorkerPreflightReport | None = None
        self._preflight_emitted = False
        self._stats = {
            "missions_processed": 0,
            "missions_completed": 0,
            "missions_failed": 0,
            "terminal_rows_persisted": 0,
            "completed_rows_persisted": 0,
            "completion_receipts_acknowledged": 0,
            "completion_receipts_failed": 0,
            "completion_receipt_backlog_overflowed": 0,
            "last_completion_receipt": None,
            "log_delivery_failures": 0,
            "log_backlog_overflowed_handlers": 0,
            "log_backlog_overflowed_entries": 0,
            "last_log_delivery": None,
            "started_at": None,
        }

    async def run(self) -> None:
        """Run the worker main loop.

        Initializes health server, database connection, starts polling,
        and handles graceful shutdown. Blocks until shutdown is requested.
        """
        self._preflight_report = None
        self._preflight_emitted = False
        self._stats["started_at"] = datetime.now(timezone.utc).isoformat()
        logger.info(
            "DeepSearch worker starting (poll_interval=%.1fs, backend=%s)",
            self.poll_interval,
            self.llm_backend,
        )

        try:
            self._register_signal_handlers()

            # Health starts in not-ready state so a failed preflight remains
            # externally visible even though no mission may be claimed.
            if self.enable_health_server:
                self._health_server = HealthServer(
                    stats_provider=self.get_health_stats,
                    port=self.health_port,
                )
                await self._health_server.start()

            self._db_client = DatabaseClient(self.db_config)
            if self._database_config_error is not None:
                self._preflight_report = await run_worker_preflight(
                    self._db_client,
                    settings=self._startup_settings,
                    lease_policy=self.lease_policy,
                    execution_fence_policy=self.execution_fence_policy,
                    tracelab_auth=self._tracelab_auth,
                    receipt_client=self._receipt_client,
                    evidence_ledger_client=self._evidence_ledger_client,
                    database_config_error=self._database_config_error,
                )
                self._emit_preflight_report()
                raise WorkerError("Worker preflight failed; see redacted health record")
            try:
                await self._db_client.connect()
                logger.info("Database connection established")
            except DatabaseError as exc:
                self._preflight_report = await run_worker_preflight(
                    self._db_client,
                    settings=self._startup_settings,
                    lease_policy=self.lease_policy,
                    execution_fence_policy=self.execution_fence_policy,
                    tracelab_auth=self._tracelab_auth,
                    receipt_client=self._receipt_client,
                    evidence_ledger_client=self._evidence_ledger_client,
                )
                self._emit_preflight_report()
                raise WorkerError(
                    "Database connection failed; see redacted health record"
                ) from exc

            self._preflight_report = await run_worker_preflight(
                self._db_client,
                settings=self._startup_settings,
                lease_policy=self.lease_policy,
                execution_fence_policy=self.execution_fence_policy,
                tracelab_auth=self._tracelab_auth,
                receipt_client=self._receipt_client,
                evidence_ledger_client=self._evidence_ledger_client,
            )
            self._emit_preflight_report()
            if not self._preflight_report.ready:
                raise WorkerError("Worker preflight failed; see redacted health record")

            self._mission_executor = MissionProcessExecutor(
                self.execution_fence_policy
            )
            self._writer = ResultWriter(self._db_client)
            self._evidence_outbox_repository = EvidenceOutboxRepository(
                self._db_client
            )
            self._evidence_outbox_dispatcher = EvidenceOutboxDispatcher(
                self._evidence_outbox_repository,
                self._evidence_ledger_client,
            )
            await self._evidence_outbox_dispatcher.start()
            self._poller = MissionPoller(
                self._db_client,
                lease_owner=self.worker_id,
                lease_policy=self.lease_policy,
                poll_interval=self.poll_interval,
                on_mission=self._handle_mission,
            )
            await self._poller.start()
        except Exception as exc:
            error_details = safe_exception_details(exc)
            logger.error("Worker loop failed: %s", error_details.to_text())
            raise
        finally:
            await self._cleanup()

        logger.info(
            "DeepSearch worker stopped (processed=%d, completed=%d, failed=%d)",
            self._stats["missions_processed"],
            self._stats["missions_completed"],
            self._stats["missions_failed"],
        )

    def _register_signal_handlers(self) -> None:
        """Register SIGTERM and SIGINT handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        def handle_signal(sig: signal.Signals) -> None:
            logger.info(
                "Received %s - initiating graceful shutdown",
                sig.name if hasattr(sig, "name") else sig,
            )
            self._shutdown_requested = True

            if self._poller:
                # Signal poller to stop after current iteration
                asyncio.create_task(self._poller.stop())

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))
                logger.debug("Registered handler for %s", sig.name)
            except NotImplementedError:
                # Windows doesn't support signal handlers in async loops
                logger.warning("Signal handlers not supported on this platform")
                break

    def _emit_preflight_report(self) -> None:
        """Log exactly one redacted startup health record per worker run."""

        if self._preflight_report is None or self._preflight_emitted:
            return
        log_method = logger.info if self._preflight_report.ready else logger.error
        log_method(
            "worker_preflight=%s",
            json.dumps(self._preflight_report.to_dict(), sort_keys=True),
        )
        self._preflight_emitted = True

    async def _cleanup(self) -> None:
        """Clean up resources on shutdown."""
        logger.info("Cleaning up worker resources")

        await self._retry_pending_transports()
        if self._pending_completion_receipts or self._pending_log_handlers:
            logger.error(
                "Worker stopping with unacknowledged transports "
                "(receipts=%d, log_batches=%d)",
                len(self._pending_completion_receipts),
                len(self._pending_log_handlers),
            )

        if self._poller:
            await self._poller.stop()

        if self._evidence_outbox_dispatcher:
            await self._evidence_outbox_dispatcher.stop()

        if self._mission_executor:
            await self._mission_executor.close()

        if self._db_client:
            await self._db_client.close()
            logger.info("Database connection closed")

        if self._health_server:
            await self._health_server.stop()

    async def _handle_mission(self, row: MissionRow) -> None:
        """Handle a claimed mission.

        Converts the row to AgentState, executes research, and writes
        results back to the database.

        Args:
            row: MissionRow from the poller
        """
        job_id = str(uuid.uuid4())
        mission_id = row.id
        self._current_mission_id = mission_id
        self._current_job_id = job_id
        self._stats["missions_processed"] += 1

        start_time = time.perf_counter()
        terminal_persisted = False
        terminal_logs_publishable = False
        heartbeat_task: asyncio.Task[None] | None = None
        final_state: Mapping[str, Any] = {}
        report_markdown: str | None = None
        protocol_payload: Dict[str, Any] | None = None

        # Attach TraceLab log handler for live mission log streaming.
        # No-op if TRACELAB_BASE_URL is not set. All exceptions from the
        # handler are swallowed internally — research run is never affected.
        _deepsearch_logger = logging.getLogger("deepsearch")
        _log_handler = TracelabLogHandler(
            mission_id=mission_id,
            auth=self._tracelab_auth,
            lease=row.lease,
        )
        _deepsearch_logger.addHandler(_log_handler)

        try:
            # -- phase: loading --
            _log_handler.set_phase("loading")
            logger.info(
                "Starting mission %s (job_id=%s, title=%s)",
                mission_id,
                job_id,
                row.title,
                extra={"tracelab_event": "phase_started"},
            )

            # Convert row to AgentState
            state = convert_row_to_state(row)

            # -- phase: compiled --
            _log_handler.set_phase("compiled")
            logger.info(
                "Converted mission %s: %d objectives, max_loops=%d",
                mission_id,
                len(state.get("mission_objectives", [])),
                state.get("max_loops", 3),
                extra={"tracelab_event": "phase_completed"},
            )

            # A freshly returned claim can already be stale after a slow DB
            # response. Prove ownership once immediately before any isolated
            # provider/artifact work, then begin the periodic watcher.
            if self._poller is not None and self._mission_executor is not None:
                await self._confirm_lease_ownership(row)
            if self._poller is not None:
                heartbeat_task = asyncio.create_task(self._heartbeat_lease(row))
                # Quiesce scheduling as soon as the watch fails, even while the
                # isolated child is still unwinding. Normal terminal cancellation
                # preserves the receiver's durable final-flush proof.
                heartbeat_task.add_done_callback(
                    lambda task: None if task.cancelled() else _log_handler.stop_delivery()
                )
                if self._mission_executor is not None:
                    await asyncio.sleep(0)  # Let the watch start before network negotiation.
                    if heartbeat_task.done():
                        await heartbeat_task
                        raise LeaseLostError("Heartbeat stopped before log negotiation")
                    await asyncio.to_thread(_log_handler.start_live_delivery)

            # Execute research (all deepsearch.* logs flow through with phase='running')
            _log_handler.set_phase("running")
            logger.info("Research started", extra={"tracelab_event": "phase_started"})
            final_state = await self._execute_research_with_lease_watch(
                state,
                row,
                job_id,
                heartbeat_task,
            )

            # Calculate duration
            duration = time.perf_counter() - start_time
            completed_at = datetime.now(timezone.utc).replace(tzinfo=None)

            telemetry = normalize_final_telemetry(final_state)
            final_outcome = resolve_final_outcome_payload(
                final_state,
                completed_at=completed_at,
            )
            telemetry["final_outcome"] = final_outcome
            # S83 m05 (decision #803) — emit the flat quality-series record as a
            # sibling of final_outcome so every persisted mission carries a
            # calibration datapoint. report_word_count was persisted onto
            # synthesis_telemetry at finalize, so the worker's final_state has it
            # (a builder fed an isolated fixture would otherwise mask word_count=0).
            _synthesis_telemetry = final_state.get("synthesis_telemetry") or {}
            _quality_record = _synthesis_telemetry.get("quality_record")
            telemetry["quality_record"] = (
                dict(_quality_record)
                if isinstance(_quality_record, Mapping)
                else build_quality_record(
                    _synthesis_telemetry,
                    _synthesis_telemetry.get("report_word_count", 0),
                )
            )
            final_state["telemetry"] = telemetry
            final_state["mission_outcome"] = final_outcome["outcome"]
            final_state["routing_reason"] = final_outcome["routing_reason"]

            metadata = build_execution_metadata(final_state, duration)
            if self._mission_executor is not None:
                # Attempt-local paths are deleted after the fenced result is
                # accepted; never persist dangling local filesystem pointers.
                metadata.contract_path = None
                metadata.telemetry_path = None

            # Prefer the finalize-emitted files, but carry the exact in-memory
            # values across the process fence so a local save/read failure cannot
            # destroy a paid synthesis. When both transports exist they must
            # agree exactly; disagreement is evidence, never a preference choice.
            report_from_file = _read_emitted_text_artifact(
                final_state.get("report_path"),
                label="report",
            )
            report_markdown = report_from_file
            report_inline = _read_inline_text_artifact(
                final_state.get("report_markdown"),
                label="report",
            )
            if report_from_file is not None and report_inline is not None:
                if report_from_file != report_inline:
                    raise _artifact_divergence_error("report")
            elif report_markdown is None:
                report_markdown = report_inline
            markdown_from_artifact = report_markdown is not None

            protocol_from_file = _read_emitted_json_artifact(
                final_state.get("protocol_path"),
                label="protocol",
            )
            protocol_payload = protocol_from_file
            protocol_inline = _read_inline_json_artifact(
                final_state.get("protocol_payload"),
                label="protocol",
            )
            if protocol_from_file is not None and protocol_inline is not None:
                if protocol_identity_sha256(
                    protocol_from_file
                ) != protocol_identity_sha256(protocol_inline):
                    raise _artifact_divergence_error("protocol")
            elif protocol_payload is None:
                protocol_payload = protocol_inline
            modern_blocked_without_required_pair = bool(
                final_state.get("synthesis_used") is True
                and final_outcome["outcome"] == MissionOutcome.BLOCKED.value
            )
            # Resolve both transports before enforcing the modern pair. This
            # lets the failure handler preserve whichever exact artifact did
            # survive instead of discarding a protocol merely because its
            # sibling report was unavailable (or vice versa).
            if report_markdown is None:
                if final_state.get("synthesis_used") is True:
                    if not modern_blocked_without_required_pair:
                        raise RequiredArtifactError(
                            "modern CodeAgent report artifact is unavailable"
                        )
                else:
                    report_markdown = generate_markdown_report(final_state)
            if protocol_payload is None:
                if final_state.get("synthesis_used") is True:
                    if not modern_blocked_without_required_pair:
                        raise RequiredArtifactError(
                            "modern CodeAgent protocol artifact is unavailable"
                        )
                else:
                    protocol_payload = generate_mission_protocol(final_state).model_dump()

            # S84 m03 — writeback identity assertion (decision #823 /
            # CAT-R003): before ANY artifact-persisting write, assert the
            # result belongs to this mission and is the document the outcome
            # gates scored. A violation raises ResultIdentityError, caught
            # below to land the mission BLOCKED with the suspect artifacts
            # preserved and the diagnostic in execution_metadata — never a
            # silent ship.
            assert_result_identity(
                row_mission_id=row.mission_id,
                row_title=row.title,
                result_markdown=report_markdown,
                result_protocol=protocol_payload,
                synthesis_telemetry=final_state.get("synthesis_telemetry"),
                markdown_from_artifact=markdown_from_artifact,
            )

            # -- phase: upload --
            _log_handler.set_phase("upload")
            logger.info(
                "Uploading results for mission %s (duration=%.1fs)",
                mission_id,
                duration,
            )

            outcome = MissionOutcome(final_outcome["outcome"])
            await self._stop_heartbeat_before_terminal(heartbeat_task)
            heartbeat_task = None
            metadata.lease_attempt = self._build_lease_audit(row)
            if outcome is MissionOutcome.COMPLETE:
                await self._writer.mark_completed(
                    mission_id=mission_id,
                    result_markdown=report_markdown,
                    result_protocol=protocol_payload,
                    execution_metadata=metadata,
                    lease=row.lease,
                    job_id=job_id,
                    completed_at=completed_at,
                )
                self._wake_evidence_outbox()
                terminal_persisted = True
                terminal_logs_publishable = True
                self._stats["terminal_rows_persisted"] += 1
                self._stats["completed_rows_persisted"] += 1
                self._stats["missions_completed"] += 1
                acknowledged = await self._deliver_completion_receipt(
                    row.mission_id,
                    job_id,
                    completion_outcome=True,
                )
                if acknowledged:
                    logger.info(
                        "Mission %s completed and acknowledged "
                        "(job_id=%s, duration=%.1fs)",
                        mission_id,
                        job_id,
                        duration,
                    )
                else:
                    logger.error(
                        "Mission %s terminal result persisted but completion "
                        "receipt remains unacknowledged (job_id=%s)",
                        mission_id,
                        job_id,
                    )
            elif self._should_requeue_transient(outcome, final_state, row.lease):
                # S90 (#959) — a provider outage is not a verdict on the
                # mission. Return it to 'queued' so the lease's remaining
                # attempts become reachable instead of landing terminally in
                # 'blocked', which is what cost pedr-retrieval-evaluation-R001
                # its run on 2026-08-13.
                # Requeue is also an ownership release: no transport may remain in flight.
                await asyncio.to_thread(_log_handler.discard)
                await self._writer.mark_requeued(
                    mission_id=mission_id,
                    error_message=(
                        "Transient synthesis outage; requeued for retry "
                        f"(attempt {row.lease.attempt_count} of "
                        f"{self.lease_policy.max_attempts}): "
                        f"{final_outcome['routing_reason'] or outcome.value}"
                    ),
                    lease=row.lease,
                    execution_metadata=metadata,
                )
                # The requeue already resolved the lease (owner/token cleared,
                # status moved off 'in_progress'), so the finally-block release
                # must not also run. It would no-op on its status predicate
                # today; setting this keeps that correctness explicit rather
                # than incidental.
                terminal_persisted = True
                self._stats["missions_requeued"] = (
                    self._stats.get("missions_requeued", 0) + 1
                )
                logger.warning(
                    "Mission %s requeued after transient failure "
                    "(job_id=%s, attempt=%d/%s, duration=%.1fs)",
                    mission_id,
                    job_id,
                    row.lease.attempt_count,
                    self.lease_policy.max_attempts,
                    duration,
                )
            else:
                await self._writer.mark_failed(
                    mission_id=mission_id,
                    error_message=final_outcome["routing_reason"] or outcome.value,
                    outcome=outcome,
                    result_markdown=report_markdown,
                    result_protocol=protocol_payload,
                    execution_metadata=metadata,
                    lease=row.lease,
                    job_id=job_id,
                    completed_at=completed_at,
                )
                if outcome is MissionOutcome.VALIDATION_FAILED:
                    self._wake_evidence_outbox()
                terminal_persisted = True
                terminal_logs_publishable = True
                self._stats["terminal_rows_persisted"] += 1
                self._stats["missions_failed"] += 1
                if outcome is MissionOutcome.VALIDATION_FAILED:
                    await self._deliver_completion_receipt(row.mission_id, job_id)
                logger.info(
                    "Mission %s ended as %s (job_id=%s, duration=%.1fs)",
                    mission_id,
                    outcome.value,
                    job_id,
                    duration,
                )

            # -- phase: complete --
            _log_handler.set_phase("complete")

        except ResultIdentityError as e:
            # S84 m03 — identity violation: land BLOCKED with the suspect
            # artifacts preserved for forensics and the diagnostic riding
            # execution_metadata. If this ever fires in production, that is
            # the decisive CAT-R003 evidence (roadmap flip-trigger:
            # recurrence after this guard -> undiagnosed writer race ->
            # retire signal).
            # type-ignores below: _stats/_writer carry the file-wide
            # Optional-typing debt (same class as the pre-existing persist
            # paths above) — not new debt from this handler.
            self._stats["missions_failed"] += 1  # type: ignore[operator]
            _log_handler.set_phase("error")
            error_details = safe_exception_details(e, category="persistence")
            logger.error(
                "Mission %s writeback identity violation: %s",
                mission_id,
                error_details.to_text(),
            )
            await self._stop_heartbeat_before_terminal(heartbeat_task)
            heartbeat_task = None
            metadata.lease_attempt = self._build_lease_audit(row)
            metadata.identity_violation = e.diagnostic
            try:
                await self._writer.mark_failed(  # type: ignore[union-attr]
                    mission_id=mission_id,
                    error_message=error_details.to_text(),
                    outcome=MissionOutcome.BLOCKED,
                    result_markdown=report_markdown,
                    result_protocol=protocol_payload,
                    execution_metadata=metadata,
                    lease=row.lease,
                    job_id=job_id,
                    completed_at=completed_at,
                )
                terminal_persisted = True
                terminal_logs_publishable = True
                self._stats["terminal_rows_persisted"] += 1
            except WriterError as we:
                writer_error = safe_exception_details(we, category="persistence")
                logger.error(
                    "Failed to mark mission %s blocked after identity violation: %s",
                    mission_id,
                    writer_error.to_text(),
                )

        except ProcessTerminationError:
            self._shutdown_requested = True
            logger.critical(
                "Mission %s could not prove isolated child termination; "
                "worker is stopping without any terminal write",
                mission_id,
            )
            if self._poller is not None:
                await self._poller.stop()
            raise

        except LeaseLostError:
            logger.warning(
                "Mission %s stopped because lease ownership was lost",
                mission_id,
            )
            raise

        except ConversionError as e:
            self._stats["missions_failed"] += 1
            error_details = safe_exception_details(e)
            error_msg = error_details.to_text()
            _log_handler.set_phase("error")
            logger.error(
                "Mission %s conversion failed: %s",
                mission_id,
                error_details.to_text(),
            )
            await self._stop_heartbeat_before_terminal(heartbeat_task)
            heartbeat_task = None
            failure_metadata = build_execution_metadata(
                {},
                time.perf_counter() - start_time,
            )
            failure_metadata.lease_attempt = self._build_lease_audit(row)
            await self._writer.mark_failed(
                mission_id,
                error_msg,
                lease=row.lease,
                job_id=job_id,
                execution_metadata=failure_metadata,
            )
            terminal_persisted = True
            terminal_logs_publishable = True
            self._stats["terminal_rows_persisted"] += 1

        except Exception as e:
            self._stats["missions_failed"] += 1
            error_details = safe_exception_details(e)
            error_msg = error_details.to_text()
            _log_handler.set_phase("error")
            logger.error(
                "Mission %s failed: %s",
                mission_id,
                error_details.to_text(),
            )
            try:
                await self._stop_heartbeat_before_terminal(heartbeat_task)
                heartbeat_task = None
                failure_metadata = build_execution_metadata(
                    dict(final_state),
                    time.perf_counter() - start_time,
                )
                failure_metadata.lease_attempt = self._build_lease_audit(row)
                await self._writer.mark_failed(
                    mission_id,
                    error_msg,
                    result_markdown=report_markdown,
                    result_protocol=protocol_payload,
                    lease=row.lease,
                    job_id=job_id,
                    execution_metadata=failure_metadata,
                )
                terminal_persisted = True
                terminal_logs_publishable = True
                self._stats["terminal_rows_persisted"] += 1
            except WriterError as we:
                writer_error = safe_exception_details(we, category="persistence")
                logger.error(
                    "Failed to mark mission %s as failed: %s",
                    mission_id,
                    writer_error.to_text(),
                )

        finally:
            # Stop capturing before any lease-loss cleanup log is emitted.
            # Terminal attempts flush below. Unfinished attempts stop scheduling
            # immediately, then quiesce transport before releasing ownership.
            _deepsearch_logger.removeHandler(_log_handler)
            if not terminal_logs_publishable:
                _log_handler.stop_delivery()
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with suppress(
                    asyncio.CancelledError,
                    LeaseLostError,
                    PollerError,
                    IsolatedExecutionError,
                ):
                    await heartbeat_task
            # Receiver checks active ownership or durable terminal proof on
            # every batch. Discard waits for in-flight HTTP before release;
            # accepted history retains this attempt identity after reclamation.
            log_close = (
                _log_handler.close
                if terminal_logs_publishable
                else _log_handler.discard
            )
            await asyncio.to_thread(log_close)
            log_delivery = _log_handler.get_telemetry()
            self._record_log_delivery(log_delivery)
            if terminal_logs_publishable and log_delivery["pending_entries"]:
                self._stats["log_delivery_failures"] += 1
                self._retain_log_handler(_log_handler)
            isolated_child_active = bool(
                self._mission_executor is not None
                and self._mission_executor.is_active(job_id)
            )
            if isolated_child_active:
                logger.critical(
                    "Mission %s still has a live isolated child; lease will not be released",
                    mission_id,
                )
            if (
                not terminal_persisted
                and self._poller is not None
                and not isolated_child_active
            ):
                try:
                    released = await self._poller.release(mission_id, row.lease)
                    if released:
                        logger.warning(
                            "Released unfinished mission %s attempt %d for recovery",
                            mission_id,
                            row.lease.attempt_count,
                        )
                except Exception as exc:  # noqa: BLE001 - expiry remains fallback
                    error_details = safe_exception_details(exc)
                    logger.error(
                        "Failed to release mission %s lease: %s",
                        mission_id,
                        error_details.to_text(),
                    )
            if self._poller is not None:
                self._poller.clear_heartbeat_audit(mission_id, row.lease)
            lease_key = (mission_id, row.lease.token, row.lease.attempt_count)
            self._lease_deadlines_monotonic.pop(lease_key, None)
            self._heartbeat_anchors_monotonic.pop(lease_key, None)
            if self._mission_executor is not None and not isolated_child_active:
                try:
                    self._mission_executor.cleanup(job_id)
                except (IsolatedExecutionError, OSError) as exc:
                    error_details = safe_exception_details(exc, category="execution")
                    logger.critical(
                        "Failed to clean isolated mission %s: %s",
                        mission_id,
                        error_details.to_text(),
                    )
            self._current_mission_id = None
            self._current_job_id = None
            # Retry process-retained notifications only after the current
            # claim is terminal or released. Bounded transport waits must
            # never consume or orphan an active mission lease.
            await self._retry_pending_transports(excluded_receipt_job_id=job_id)

    def _should_requeue_transient(
        self,
        outcome: MissionOutcome,
        final_state: Mapping[str, Any],
        lease: LeaseAttempt,
    ) -> bool:
        """Return True when a failed mission should go back to 'queued'.

        S90 (#959). Every condition must hold:

        1. The outcome is BLOCKED. A VALIDATION_FAILED mission produced a
           reviewable artifact — that is a verdict on the work, not an outage,
           and re-running it would discard the artifact.
        2. Synthesis flagged the failure transient. This is
           ``_is_retryable_generation_failure``'s verdict, carried through
           ``synthesis_telemetry``: a transient transport/server condition that
           consumed zero tokens. A deterministic failure fails identically on
           every attempt, and a billed one must never be silently re-run.
        3. Attempts remain. The claim predicate refuses a row at or past
           ``max_attempts``, so requeuing an exhausted mission would strand it
           until the dead-letter sweep catches it — noise instead of recovery.

        A missing or malformed policy resolves to False: without a known
        attempt ceiling there is no bound on the requeue, and an unbounded
        requeue is worse than a terminal failure.
        """
        if outcome is not MissionOutcome.BLOCKED:
            return False

        telemetry = final_state.get("synthesis_telemetry")
        if not isinstance(telemetry, Mapping):
            return False
        if telemetry.get("synthesis_error_transient") is not True:
            return False

        max_attempts = self.lease_policy.max_attempts
        if not isinstance(max_attempts, int) or max_attempts <= 0:
            return False
        return lease.attempt_count < max_attempts

    def _wake_evidence_outbox(self) -> None:
        """Wake durable post-commit delivery after an eligible terminal write."""

        if self._evidence_outbox_dispatcher is not None:
            self._evidence_outbox_dispatcher.wake()

    def _build_lease_audit(self, row: MissionRow) -> dict[str, Any]:
        """Build truthful claim/heartbeat/reset telemetry for a terminal write."""

        audit = row.lease.audit_dict(row.id)
        audit["execution_fence"] = self.execution_fence_policy.evaluate(
            self.lease_policy
        )
        if self._poller is not None:
            audit["heartbeat_runtime"] = self._poller.heartbeat_audit(
                row.id,
                row.lease,
            )
        else:
            audit["heartbeat_runtime"] = {
                "heartbeat_count": 0,
                "last_confirmed_heartbeat_at": None,
                "last_confirmed_expires_at": None,
            }
        audit["terminal_reset"] = {
            "deepsearch_lease_token": "cleared_atomically",
            "deepsearch_lease_expires_at": "cleared_atomically",
        }
        return audit

    @staticmethod
    async def _stop_heartbeat_before_terminal(
        heartbeat_task: asyncio.Task[None] | None,
    ) -> None:
        """Stop lease renewal and surface ownership loss before terminal SQL."""

        if heartbeat_task is None:
            return
        if heartbeat_task.done():
            await heartbeat_task
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task

    async def _deliver_completion_receipt(
        self,
        mission_id: str,
        job_id: str,
        *,
        completion_outcome: bool = False,
        enqueue_on_failure: bool = True,
    ) -> bool:
        """Deliver a post-commit receipt and expose, never raise, failure state."""

        receipt = CompletionReceipt(job_id=job_id, mission_id=mission_id)
        try:
            result = await self._receipt_client.deliver(receipt)
            telemetry = {
                "success": result.success,
                "mission_id": receipt.mission_id,
                "job_id": receipt.job_id,
                "attempts": result.attempts,
                "retryable": result.retryable,
                "status_code": result.final_status_code,
                "error": result.error,
            }
        except Exception as exc:  # noqa: BLE001 - never rewrite terminal result
            error_details = safe_exception_details(exc, category="transport")
            telemetry = {
                "success": False,
                "mission_id": receipt.mission_id,
                "job_id": receipt.job_id,
                "attempts": 0,
                "retryable": True,
                "status_code": None,
                "error": error_details.to_text(),
            }

        self._stats["last_completion_receipt"] = telemetry
        if telemetry["success"]:
            self._stats["completion_receipts_acknowledged"] += 1
            return True

        self._stats["completion_receipts_failed"] += 1
        if enqueue_on_failure and telemetry["retryable"]:
            self._retain_completion_receipt(receipt, completion_outcome)
        logger.error(
            "Completion receipt unacknowledged for mission %s (job_id=%s): %s",
            receipt.mission_id,
            receipt.job_id,
            telemetry["error"],
        )
        return False

    def _retain_completion_receipt(
        self,
        receipt: CompletionReceipt,
        completion_outcome: bool,
    ) -> None:
        """Retain a bounded newest-first recovery hint for TraceLab reconciliation."""

        if receipt in self._pending_completion_receipts:
            self._pending_completion_receipts[receipt] = completion_outcome
            return
        if len(self._pending_completion_receipts) >= MAX_PENDING_COMPLETION_RECEIPTS:
            oldest = next(iter(self._pending_completion_receipts))
            self._pending_completion_receipts.pop(oldest)
            self._stats["completion_receipt_backlog_overflowed"] += 1
            logger.error(
                "Completion-receipt backlog full; oldest hint evicted "
                "(capacity=%d)",
                MAX_PENDING_COMPLETION_RECEIPTS,
            )
        self._pending_completion_receipts[receipt] = completion_outcome

    async def _retry_pending_transports(
        self,
        *,
        excluded_receipt_job_id: str | None = None,
    ) -> None:
        """Retry process-retained transport failures at the next mission fence."""

        pending_receipts = [
            item
            for item in self._pending_completion_receipts.items()
            if item[0].job_id != excluded_receipt_job_id
        ][:MAX_RECEIPT_RETRIES_PER_FENCE]
        for receipt, completion_outcome in pending_receipts:
            acknowledged = await self._deliver_completion_receipt(
                receipt.mission_id,
                receipt.job_id,
                completion_outcome=completion_outcome,
                enqueue_on_failure=False,
            )
            if acknowledged:
                self._pending_completion_receipts.pop(receipt, None)
            else:
                retained_outcome = self._pending_completion_receipts.pop(receipt)
                self._pending_completion_receipts[receipt] = retained_outcome

        pending_handlers = list(
            self._pending_log_handlers[:MAX_LOG_HANDLER_RETRIES_PER_FENCE]
        )
        for handler in pending_handlers:
            try:
                await asyncio.to_thread(handler.flush)
            except Exception as exc:  # noqa: BLE001 - retain and report later
                error_details = safe_exception_details(exc, category="transport")
                logger.error(
                    "Retained TraceLab log retry failed unexpectedly: %s",
                    error_details.to_text(),
                )
                self._pending_log_handlers.remove(handler)
                self._pending_log_handlers.append(handler)
                continue
            telemetry = handler.get_telemetry()
            self._record_log_delivery(telemetry)
            if not telemetry["pending_entries"]:
                self._pending_log_handlers.remove(handler)
            else:
                self._pending_log_handlers.remove(handler)
                self._pending_log_handlers.append(handler)

    def _retain_log_handler(
        self,
        handler: TracelabLogHandler,
    ) -> None:
        """Retain a bounded process-wide backlog, preserving newest missions."""

        if len(self._pending_log_handlers) < MAX_PENDING_LOG_HANDLERS:
            self._pending_log_handlers.append(handler)
            return
        evicted = self._pending_log_handlers.pop(0)
        evicted_telemetry = evicted.get_telemetry()
        overflowed_entries = int(evicted_telemetry.get("pending_entries") or 0)
        self._pending_log_handlers.append(handler)
        self._stats["log_backlog_overflowed_handlers"] += 1
        self._stats["log_backlog_overflowed_entries"] += overflowed_entries
        logger.error(
            "TraceLab retained-log backlog full; oldest mission batch evicted "
            "(capacity=%d, entries=%d)",
            MAX_PENDING_LOG_HANDLERS,
            overflowed_entries,
        )

    def _record_log_delivery(self, telemetry: Mapping[str, Any]) -> None:
        """Promote per-mission log delivery state into worker health."""

        self._stats["last_log_delivery"] = dict(telemetry)

    @staticmethod
    def _consume_background_heartbeat(task: asyncio.Task[bool]) -> None:
        """Retrieve a timed-out heartbeat result without extending its deadline."""

        with suppress(asyncio.CancelledError, Exception):
            task.result()

    @staticmethod
    def _parse_lease_timestamp(value: object) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _latest_confirmed_heartbeat(self, row: MissionRow) -> datetime | None:
        if self._poller is None:
            return None
        audit = self._poller.heartbeat_audit(row.id, row.lease)
        heartbeat_count = audit.get("heartbeat_count")
        if type(heartbeat_count) is not int or heartbeat_count < 1:
            return None
        return self._parse_lease_timestamp(audit.get("last_confirmed_heartbeat_at"))

    def _lease_hard_deadline_monotonic(self, row: MissionRow) -> float:
        """Return the child self-kill deadline before SQL reclaim eligibility."""

        lease_key = (row.id, row.lease.token, row.lease.attempt_count)
        confirmed_deadline = self._lease_deadlines_monotonic.get(lease_key)
        if confirmed_deadline is not None:
            if confirmed_deadline <= time.monotonic():
                raise LeaseLostError(
                    f"Mission {row.id} lease is too near reclaim eligibility to execute"
                )
            return confirmed_deadline
        if self._poller is not None and self._mission_executor is not None:
            raise LeaseLostError(
                f"Mission {row.id} has no locally anchored heartbeat deadline"
            )

        # Direct unit/local callers without the production poller use the
        # claim timestamp as a conservative compatibility fallback. Production
        # always takes the monotonic, clock-skew-independent branch above.
        expiry = row.lease.expires_at
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        assert self.lease_policy.reclaim_grace_seconds is not None
        assert self.execution_fence_policy.safety_margin_seconds is not None
        hard_stop_at = (
            expiry.astimezone(timezone.utc)
            + timedelta(seconds=self.lease_policy.reclaim_grace_seconds)
            - timedelta(seconds=self.execution_fence_policy.safety_margin_seconds)
        )
        remaining = (hard_stop_at - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            raise LeaseLostError(
                f"Mission {row.id} lease is too near reclaim eligibility to execute"
            )
        return time.monotonic() + remaining

    async def _confirm_lease_ownership(self, row: MissionRow) -> float:
        """Bound one DB heartbeat without awaiting a stuck cancellation unwind."""

        assert self._poller is not None
        assert self.execution_fence_policy.heartbeat_timeout_seconds is not None
        heartbeat_started = time.monotonic()
        heartbeat_io = asyncio.create_task(
            self._poller.heartbeat(row.id, row.lease)
        )
        try:
            done, _ = await asyncio.wait(
                (heartbeat_io,),
                timeout=self.execution_fence_policy.heartbeat_timeout_seconds,
            )
        except asyncio.CancelledError:
            heartbeat_io.cancel()
            heartbeat_io.add_done_callback(self._consume_background_heartbeat)
            raise
        if heartbeat_io not in done:
            heartbeat_io.cancel()
            heartbeat_io.add_done_callback(self._consume_background_heartbeat)
            raise LeaseLostError(
                f"Mission {row.id} heartbeat could not prove ownership"
            )
        try:
            confirmed = await heartbeat_io
        except PollerError as exc:
            raise LeaseLostError(
                f"Mission {row.id} heartbeat could not prove ownership"
            ) from exc
        if not confirmed:
            raise LeaseLostError(
                f"Mission {row.id} lease attempt {row.lease.attempt_count} lost"
            )
        confirmed_at = self._latest_confirmed_heartbeat(row)
        if confirmed_at is None:
            raise LeaseLostError(
                f"Mission {row.id} heartbeat returned no database timestamp"
            )
        assert self.lease_policy.lease_seconds is not None
        assert self.lease_policy.reclaim_grace_seconds is not None
        assert self.execution_fence_policy.safety_margin_seconds is not None
        lease_key = (row.id, row.lease.token, row.lease.attempt_count)
        hard_deadline = heartbeat_started + (
            self.lease_policy.lease_seconds
            + self.lease_policy.reclaim_grace_seconds
            - self.execution_fence_policy.safety_margin_seconds
        )
        self._lease_deadlines_monotonic[lease_key] = hard_deadline
        self._heartbeat_anchors_monotonic[lease_key] = heartbeat_started
        if self._mission_executor is not None and self._current_job_id is not None:
            try:
                self._mission_executor.extend_deadline(
                    self._current_job_id,
                    hard_deadline,
                )
            except IsolatedExecutionError as exc:
                raise LeaseLostError(
                    f"Mission {row.id} lost its child deadline watchdog"
                ) from exc
        return heartbeat_started

    async def _heartbeat_lease(self, row: MissionRow) -> None:
        """Keep the active claim recoverable and fail when ownership is lost."""

        assert self.lease_policy.heartbeat_seconds is not None
        lease_key = (row.id, row.lease.token, row.lease.attempt_count)
        heartbeat_anchor = self._heartbeat_anchors_monotonic.get(lease_key)
        while True:
            if heartbeat_anchor is not None:
                delay = max(
                    0.0,
                    heartbeat_anchor
                    + self.lease_policy.heartbeat_seconds
                    - time.monotonic(),
                )
                await asyncio.sleep(delay)
            heartbeat_anchor = await self._confirm_lease_ownership(row)

    async def _execute_research_with_lease_watch(
        self,
        state: AgentState,
        row: MissionRow,
        job_id: str,
        heartbeat_task: asyncio.Task[None] | None,
    ) -> AgentState:
        """Stop all mission work before ownership can become reclaimable."""

        if heartbeat_task is not None and heartbeat_task.done():
            await heartbeat_task
            raise LeaseLostError(
                f"Mission {row.id} heartbeat stopped before research"
            )
        if self._mission_executor is not None:
            return cast(
                AgentState,
                await self._mission_executor.execute(
                    _run_isolated_research,
                    (state, row.id, job_id, self.llm_backend, self.model),
                    job_id=job_id,
                    hard_deadline_monotonic=self._lease_hard_deadline_monotonic(
                        row
                    ),
                    heartbeat_task=heartbeat_task,
                ),
            )

        if heartbeat_task is None:
            return await self._execute_research(state, row.id, job_id)

        cancel_event = threading.Event()
        with bind_execution_cancel_event(cancel_event):
            research_task = asyncio.create_task(
                self._execute_research(state, row.id, job_id)
            )
        try:
            done, _ = await asyncio.wait(
                (research_task, heartbeat_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            cancel_event.set()
            with suppress(asyncio.CancelledError, ExecutionCancelledError):
                await asyncio.shield(research_task)
            raise

        if heartbeat_task in done:
            cancel_event.set()
            heartbeat_error: BaseException | None = None
            try:
                await heartbeat_task
            except BaseException as exc:  # preserve the ownership-loss cause
                heartbeat_error = exc
            try:
                await research_task
            except ExecutionCancelledError:
                pass
            except Exception as exc:  # noqa: BLE001 - ownership loss remains primary
                error_details = safe_exception_details(exc)
                logger.warning(
                    "Research unwind after lease loss failed: %s",
                    error_details.to_text(),
                )
            if heartbeat_error is not None:
                raise heartbeat_error
            raise LeaseLostError(
                f"Mission {row.id} heartbeat stopped during research"
            )

        return await research_task

    async def _execute_research(
        self,
        state: AgentState,
        mission_id: str,
        job_id: str,
    ) -> AgentState:
        """Execute the LangGraph research workflow.

        Args:
            state: Initial AgentState from conversion
            mission_id: Mission identifier for logging
            job_id: Job identifier for tracking

        Returns:
            Final AgentState after research completion
        """
        # Configure LLM environment
        raise_if_execution_cancelled()
        os.environ["LLM_BACKEND"] = self.llm_backend
        os.environ[_MODEL_ENV_BY_BACKEND.get(self.llm_backend, "OLLAMA_MODEL")] = self.model

        settings = DeepSearchSettings.load()
        # Scope worker executions to a unique thread ID per claimed job so a
        # re-submitted production mission cannot silently resume stale
        # checkpoints from a prior attempt.
        thread_id = settings.generate_thread_id(mission_id, execution_token=job_id)
        apply_runtime_context(
            state,
            build_runtime_context(
                settings=settings,
                state=state,
                llm_backend=self.llm_backend,
                model=self.model,
                execution_path="tracelab",
                thread_id=thread_id,
            ),
        )
        # S87-m07: offload mission-contract compilation off the event-loop
        # thread. It runs a synchronous LLM entity-extraction pass; running it
        # inline could freeze the loop and starve the lease heartbeat (same
        # class as the critique blocker in the CAT-R001-S87 incident).
        contract_artifact = await asyncio.to_thread(
            prepare_execution_contract, state, origin="tracelab"
        )
        raise_if_execution_cancelled()
        logger.info(
            "Compiled mission contract %s -> %s",
            state.get("mission_contract_id"),
            state.get("mission_contract_path"),
        )

        # Initialize telemetry
        telemetry_collector = TelemetryCollector(
            mission_id=mission_id,
            project_id=state.get("project_id"),
        )
        telemetry_collector.set_contract(
            contract_id=contract_artifact.contract.contract_id,
            contract_path=str(contract_artifact.path),
            contract_version=contract_artifact.contract.contract_version,
            contract_origin=contract_artifact.contract.origin,
        )

        # Set up checkpointer
        checkpointer = None
        checkpointer_ctx = None
        if os.getenv("WORKER_DISABLE_CHECKPOINTING", "").strip().lower() == "true":
            logger.info("Checkpoint persistence disabled via WORKER_DISABLE_CHECKPOINTING")
        else:
            try:
                checkpointer_ctx = create_checkpointer(settings.checkpoint_db_path)
                checkpointer = await checkpointer_ctx.__aenter__()
            except CheckpointerUnavailableError as exc:
                error_details = safe_exception_details(exc, category="persistence")
                logger.warning(
                    "Checkpoint persistence unavailable: %s",
                    error_details.to_text(),
                )
                checkpointer_ctx = None
            except Exception as exc:
                error_details = safe_exception_details(exc, category="persistence")
                logger.warning(
                    "Checkpoint persistence failed, continuing without: %s",
                    error_details.to_text(),
                )
                checkpointer_ctx = None
                checkpointer = None

        recursion_limit = calculate_recursion_limit(state.get("max_loops", 3) or 3)
        graph_config = thread_config(thread_id, recursion_limit=recursion_limit)

        try:
            # Build and run workflow
            graph = agent_graph.build_workflow(checkpointer=checkpointer)
            logger.info("Executing research workflow for mission %s", mission_id)

            # Checkpointer compatibility is resolved during ``__aenter__``
            # above. Once graph execution begins, never rerun from the original
            # state: the first invocation may already have paid for retrieval
            # and accumulated source_fetch/url_liveness audit rows. Retrying
            # here without a checkpointer would silently erase those attempts
            # from the terminal Evidence Ledger envelope.
            raise_if_execution_cancelled()
            final_state = cast(
                AgentState,
                await graph.ainvoke(state, config=graph_config),
            )
            raise_if_execution_cancelled()

            # Extract telemetry
            record_finding_records_from_state(final_state, telemetry_collector)
            extract_sources_from_state(final_state, telemetry_collector)
            telemetry = normalize_final_telemetry(final_state)
            if final_state.get("degraded_mode"):
                policy = str(final_state.get("degradation_policy") or "")
                telemetry_collector.set_degradation([policy] if policy else ["degraded"])
            termination_reason = str(final_state.get("termination_reason") or "").strip()
            if termination_reason:
                telemetry_collector.set_termination(early=True, reason=termination_reason)
            telemetry_collector.set_cost(float(telemetry.get("estimated_cost_usd") or 0.0))
            telemetry["validation_summary"] = {
                "requested": False,
                "status": "not_requested",
            }
            telemetry_collector.merge_state_telemetry(telemetry)
            mission_telemetry = telemetry_collector.finalize()
            raise_if_execution_cancelled()
            telemetry_writer = TelemetryWriter()
            telemetry_path = telemetry_writer.write(mission_telemetry)
            raise_if_execution_cancelled()
            if telemetry_path:
                telemetry["telemetry_artifact_path"] = telemetry_path
            final_state["telemetry"] = telemetry

            # S84 m04 — report the honest effort signal (the legacy loop
            # counter is a dead constant 0 on the single-node graph).
            logger.info(
                "Research workflow completed for mission %s (steps=%s)",
                mission_id,
                (final_state.get("synthesis_telemetry") or {}).get("step_count", "n/a"),
            )

            return final_state

        finally:
            if checkpointer_ctx is not None:
                await checkpointer_ctx.__aexit__(None, None, None)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            **self._stats,
            "current_mission_id": self._current_mission_id,
            "current_job_id": self._current_job_id,
            "shutdown_requested": self._shutdown_requested,
            "poll_interval": self.poll_interval,
            "llm_backend": self.llm_backend,
            "pending_completion_receipts": len(self._pending_completion_receipts),
            "pending_log_batches": len(self._pending_log_handlers),
        }

    def get_health_stats(self) -> Dict[str, Any]:
        """Get health check statistics for the health endpoint.

        Returns stats formatted for the /health endpoint response.
        """
        return {
            "missions_processed": self._stats["missions_processed"],
            "missions_completed": self._stats["missions_completed"],
            "missions_failed": self._stats["missions_failed"],
            "terminal_rows_persisted": self._stats["terminal_rows_persisted"],
            "completed_rows_persisted": self._stats["completed_rows_persisted"],
            "completion_receipts_acknowledged": self._stats[
                "completion_receipts_acknowledged"
            ],
            "completion_receipts_failed": self._stats["completion_receipts_failed"],
            "completion_receipt_backlog_overflowed": self._stats[
                "completion_receipt_backlog_overflowed"
            ],
            "last_completion_receipt": _public_completion_receipt_health(
                self._stats["last_completion_receipt"]
            ),
            "pending_completion_receipts": len(self._pending_completion_receipts),
            "log_delivery_failures": self._stats["log_delivery_failures"],
            "log_backlog_overflowed_handlers": self._stats[
                "log_backlog_overflowed_handlers"
            ],
            "log_backlog_overflowed_entries": self._stats[
                "log_backlog_overflowed_entries"
            ],
            "last_log_delivery": _public_log_delivery_health(
                self._stats["last_log_delivery"]
            ),
            "pending_log_batches": len(self._pending_log_handlers),
            "evidence_ledger_delivery": (
                self._evidence_outbox_dispatcher.stats
                if self._evidence_outbox_dispatcher is not None
                else {
                    "running": False,
                    "claimed": 0,
                    "acknowledged": 0,
                    "retry_scheduled": 0,
                    "dead_lettered": 0,
                    "exhausted_swept": 0,
                    "lease_conflicts": 0,
                    "dispatch_errors": 0,
                    "pending": 0,
                    "leased": 0,
                    "acked": 0,
                    "dead_letter": 0,
                    "invalid_exhausted_pending": 0,
                    "overflowed": 0,
                    "discarded": 0,
                    "last_delivery": None,
                }
            ),
            "processing": self.is_processing,
            "poll_interval": self.poll_interval,
            "ready": bool(
                self._preflight_report and self._preflight_report.ready
            ),
            "preflight": (
                self._preflight_report.to_dict()
                if self._preflight_report is not None
                else {"ready": False, "status": "pending"}
            ),
        }

    @property
    def is_processing(self) -> bool:
        """Check if worker is currently processing a mission."""
        return self._current_mission_id is not None


__all__ = [
    "DeepSearchWorker",
    "WorkerError",
    "LeaseLostError",
]
