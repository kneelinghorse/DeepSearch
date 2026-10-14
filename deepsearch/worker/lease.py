"""Versioned TraceLab lease contract shared by worker boundary components."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, NamedTuple, cast

LEASE_CONTRACT_REVISION = 2
LEASE_CONTRACT_VERSION = f"tracelab-missions-lease-v{LEASE_CONTRACT_REVISION}"
EXECUTION_FENCE_REVISION = 1
EXECUTION_FENCE_VERSION = f"deepsearch-execution-fence-v{EXECUTION_FENCE_REVISION}"

_REQUIRED_PROVIDER_LIMIT_FIELDS: dict[str, frozenset[str]] = {
    "contract": frozenset(
        {
            "backend",
            "request_timeout_seconds",
            "client_max_retries",
            "configured_max_retries",
            "wrapper_max_retries",
            "wrapper_attempts",
            "hard_bound_mode",
        }
    ),
    "codeagent": frozenset(
        {
            "backend",
            "request_timeout_seconds",
            "client_max_retries",
            "smolagents_rate_limit_retry",
            "synthesis_attempts",
            "max_step_fallback_attempts",
            "recovery_passes",
            "hard_bound_mode",
        }
    ),
    "tools": frozenset(
        {
            "configured_request_timeout_seconds",
            "configured_max_retries",
            "linkup_request_timeout_seconds",
            "linkup_max_retries",
            "source_fetch_request_timeout_seconds",
            "source_fetch_max_redirects",
            "url_liveness_request_timeout_seconds",
            "url_liveness_max_requests",
            "scholar_backend_request_timeout_seconds",
            "scholar_max_backends",
            "hard_bound_mode",
        }
    ),
}


def _is_finite_number(value: object) -> bool:
    """Return whether ``value`` is a non-boolean finite real number."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_bounded_int(value: object, minimum: int, maximum: int) -> bool:
    """Return whether ``value`` is an exact integer within inclusive bounds."""

    return (
        type(value) is int
        and minimum <= value
        and value <= maximum
    )


# Additive TraceLab ``missions`` columns required before this worker may poll.
# The coordinated migration specification lives in
# ``cmos/reports/sprint_87_m05_tracelab_lease_contract.md``.
LEASE_COLUMN_TYPES: dict[str, frozenset[str]] = {
    "deepsearch_lease_owner": frozenset({"text"}),
    "deepsearch_lease_token": frozenset({"text"}),
    "deepsearch_leased_at": frozenset({"timestamp with time zone"}),
    "deepsearch_heartbeat_at": frozenset({"timestamp with time zone"}),
    "deepsearch_lease_expires_at": frozenset({"timestamp with time zone"}),
    "deepsearch_attempt_count": frozenset({"integer"}),
    "deepsearch_result_key": frozenset({"text"}),
}

# Migration 039 is the reproducible schema boundary for lease-v2. These
# constraints are separate from the type map because startup must reject a
# manually-ahead schema whose types happen to match while nullability/defaults
# still make terminal reset or first claim unsafe.
LEASE_NULLABLE_COLUMNS = frozenset(
    {
        "deepsearch_lease_owner",
        "deepsearch_lease_token",
        "deepsearch_leased_at",
        "deepsearch_heartbeat_at",
        "deepsearch_lease_expires_at",
        "deepsearch_result_key",
    }
)
LEASE_NON_NULL_DEFAULTS: dict[str, str] = {
    "deepsearch_attempt_count": "0",
}


class LeaseIndexSpec(NamedTuple):
    """Canonical PostgreSQL index definition required by the lease contract."""

    columns: tuple[str, ...]
    unique: bool
    predicate: str | None


LEASE_INDEX_SPECS: dict[str, LeaseIndexSpec] = {
    "missions_deepsearch_lease_token_active_uq": LeaseIndexSpec(
        columns=("deepsearch_lease_token",),
        unique=True,
        predicate="deepsearch_lease_token IS NOT NULL",
    ),
    "missions_deepsearch_result_key_uq": LeaseIndexSpec(
        columns=("deepsearch_result_key",),
        unique=True,
        predicate="deepsearch_result_key IS NOT NULL",
    ),
    "missions_deepsearch_claim_scan_idx": LeaseIndexSpec(
        columns=("status", "deepsearch_lease_expires_at", "queued_at"),
        unique=False,
        predicate=None,
    ),
}


@dataclass(frozen=True, slots=True)
class LeasePolicy:
    """Runtime lease timing and bounded retry policy.

    ``None`` preserves an invalid environment value until preflight can emit a
    single actionable health record. Polling is never constructed with an
    invalid policy.
    """

    lease_seconds: int | None
    heartbeat_seconds: int | None
    max_attempts: int | None
    reclaim_grace_seconds: int | None = 30

    @property
    def errors(self) -> tuple[str, ...]:
        """Return stable configuration errors without exposing raw values."""

        errors: list[str] = []
        lease_valid = type(self.lease_seconds) is int and self.lease_seconds >= 30
        heartbeat_valid = (
            type(self.heartbeat_seconds) is int and self.heartbeat_seconds >= 1
        )
        attempts_valid = (
            type(self.max_attempts) is int and 1 <= self.max_attempts <= 20
        )
        grace_valid = (
            type(self.reclaim_grace_seconds) is int
            and 0 <= self.reclaim_grace_seconds <= 3600
        )
        if not lease_valid:
            errors.append("WORKER_LEASE_SECONDS must be an integer >= 30")
        if not heartbeat_valid:
            errors.append("WORKER_HEARTBEAT_SECONDS must be a positive integer")
        if not attempts_valid:
            errors.append("WORKER_MAX_LEASE_ATTEMPTS must be between 1 and 20")
        if not grace_valid:
            errors.append(
                "WORKER_RECLAIM_GRACE_SECONDS must be between 0 and 3600"
            )
        if lease_valid and heartbeat_valid:
            lease_seconds = cast(int, self.lease_seconds)
            heartbeat_seconds = cast(int, self.heartbeat_seconds)
            if heartbeat_seconds * 2 >= lease_seconds:
                errors.append(
                    "heartbeat interval must be less than half the lease duration"
                )
        return tuple(errors)

    @property
    def valid(self) -> bool:
        """Return whether the policy can safely drive a poller."""

        return not self.errors


@dataclass(frozen=True, slots=True)
class ExecutionFencePolicy:
    """Hard bounds that keep an expired attempt from overlapping its successor.

    Provider limits are recorded for operational truth, but they do not form
    the termination bound: provider libraries contain retries and transports
    that are not uniformly killable. The finite ``shutdown_bound_seconds`` is
    supplied by the spawned per-mission process boundary.
    """

    heartbeat_timeout_seconds: float | None
    cancellation_grace_seconds: float | None
    terminate_grace_seconds: float | None
    kill_grace_seconds: float | None
    safety_margin_seconds: float | None
    provider_limits: Mapping[str, Any]
    isolation_mode: str = "spawn-process-v1"

    @property
    def errors(self) -> tuple[str, ...]:
        """Return stable configuration errors without exposing credentials."""

        errors: list[str] = []
        positive = {
            "WORKER_HEARTBEAT_TIMEOUT_SECONDS": self.heartbeat_timeout_seconds,
            "WORKER_TERMINATE_GRACE_SECONDS": self.terminate_grace_seconds,
            "WORKER_KILL_GRACE_SECONDS": self.kill_grace_seconds,
        }
        for name, value in positive.items():
            numeric = (
                float(cast(float, value)) if _is_finite_number(value) else None
            )
            if numeric is None or numeric <= 0:
                errors.append(f"{name} must be a finite number > 0")
        cancellation_grace = self.cancellation_grace_seconds
        cancellation_numeric = (
            float(cast(float, cancellation_grace))
            if _is_finite_number(cancellation_grace)
            else None
        )
        if cancellation_numeric is None or cancellation_numeric < 0:
            errors.append(
                "WORKER_CANCELLATION_GRACE_SECONDS must be a finite number >= 0"
            )
        safety_margin = self.safety_margin_seconds
        margin_numeric = (
            float(cast(float, safety_margin))
            if _is_finite_number(safety_margin)
            else None
        )
        if margin_numeric is None or margin_numeric < 1.0:
            errors.append(
                "WORKER_LEASE_SAFETY_MARGIN_SECONDS must be a finite number >= 1"
            )
        if self.isolation_mode != "spawn-process-v1":
            errors.append("execution isolation mode must be spawn-process-v1")
        actual_groups = set(self.provider_limits)
        expected_groups = set(_REQUIRED_PROVIDER_LIMIT_FIELDS)
        if actual_groups != expected_groups:
            missing = sorted(expected_groups - actual_groups)
            unknown = sorted(actual_groups - expected_groups)
            if missing:
                errors.append("missing provider limit groups: " + ", ".join(missing))
            if unknown:
                errors.append("unknown provider limit groups: " + ", ".join(unknown))
        for group in sorted(expected_groups & actual_groups):
            limits = self.provider_limits[group]
            if not isinstance(limits, Mapping):
                errors.append(f"provider limit group {group} must be a mapping")
                continue
            expected_fields = _REQUIRED_PROVIDER_LIMIT_FIELDS[group]
            actual_fields = set(limits)
            if actual_fields != set(expected_fields):
                missing = sorted(set(expected_fields) - actual_fields)
                unknown = sorted(actual_fields - set(expected_fields))
                if missing:
                    errors.append(
                        f"missing provider limits for {group}: " + ", ".join(missing)
                    )
                if unknown:
                    errors.append(
                        f"unknown provider limits for {group}: " + ", ".join(unknown)
                    )
            for name, value in limits.items():
                label = f"provider limit {group}.{name}"
                if name == "backend" and (
                    not isinstance(value, str) or not value.strip()
                ):
                    errors.append(f"{label} must be a non-empty string")
                elif name == "hard_bound_mode" and (
                    not isinstance(value, str) or value != self.isolation_mode
                ):
                    errors.append(f"{label} must match the execution isolation mode")
                elif "timeout_seconds" in name:
                    optional = group == "contract" and name == "request_timeout_seconds"
                    if value is None and not optional:
                        errors.append(f"{label} must be a finite number > 0")
                    elif value is not None and (
                        not _is_finite_number(value) or float(value) <= 0
                    ):
                        errors.append(f"{label} must be a finite number > 0")
                elif (
                    name.endswith("max_retries")
                    and not (
                        group == "contract"
                        and name == "client_max_retries"
                        and value is None
                    )
                    and not _is_bounded_int(value, 0, 20)
                ):
                    errors.append(f"{label} must be an integer between 0 and 20")
                elif name.endswith("attempts") and not _is_bounded_int(value, 1, 20):
                    errors.append(f"{label} must be an integer between 1 and 20")
                elif name.endswith("passes") and not _is_bounded_int(value, 1, 20):
                    errors.append(f"{label} must be an integer between 1 and 20")
                elif name.endswith(
                    ("max_redirects", "max_requests", "max_backends")
                ) and not _is_bounded_int(
                    value,
                    1,
                    20,
                ):
                    errors.append(f"{label} must be an integer between 1 and 20")
                elif name.endswith("retry") and not isinstance(value, bool):
                    errors.append(f"{label} must be boolean")
        contract_limits = self.provider_limits.get("contract")
        if isinstance(contract_limits, Mapping):
            backend = contract_limits.get("backend")
            timeout = contract_limits.get("request_timeout_seconds")
            client_retries = contract_limits.get("client_max_retries")
            if backend not in {"deepseek", "minimax", "gemini", "ollama", "openai"}:
                errors.append("provider limit contract.backend is unsupported")
            elif backend in {"deepseek", "minimax", "openai"} and (
                timeout is None or client_retries is None
            ):
                errors.append(
                    "OpenAI-compatible contract backends require explicit "
                    "transport timeout and retry bounds"
                )
            elif backend in {"gemini", "ollama"} and (
                timeout is not None or client_retries is not None
            ):
                errors.append(
                    "Gemini/Ollama contract transport bounds must be recorded "
                    "as null and delegated to process isolation"
                )
            wrapper_retries = contract_limits.get("wrapper_max_retries")
            wrapper_attempts = contract_limits.get("wrapper_attempts")
            if (
                type(wrapper_retries) is int
                and type(wrapper_attempts) is int
                and wrapper_attempts != wrapper_retries + 1
            ):
                errors.append(
                    "provider limit contract.wrapper_attempts must equal "
                    "wrapper_max_retries plus one"
                )
        codeagent_limits = self.provider_limits.get("codeagent")
        if isinstance(codeagent_limits, Mapping) and codeagent_limits.get(
            "backend"
        ) not in {"deepseek-openai-compatible", "openai-openai-compatible"}:
            errors.append(
                "provider limit codeagent.backend must be "
                "deepseek-openai-compatible or openai-openai-compatible"
            )
        return tuple(errors)

    @property
    def valid(self) -> bool:
        """Return whether every execution-fence component is finite."""

        return not self.errors

    @property
    def shutdown_bound_seconds(self) -> float | None:
        """Return the configured cooperative-to-hard-kill completion bound."""

        values = (
            self.cancellation_grace_seconds,
            self.terminate_grace_seconds,
            self.kill_grace_seconds,
        )
        if any(not _is_finite_number(value) for value in values):
            return None
        return sum(float(cast(float, value)) for value in values)

    def evaluate(self, lease_policy: LeasePolicy) -> dict[str, Any]:
        """Return the machine-readable no-overlap startup invariant.

        A heartbeat can fail immediately after the worker begins sleeping. Its
        loss is therefore known no later than ``H + D`` after the last
        confirmed extension. The isolated child then has at most ``U`` to
        stop. A successor is not eligible until ``L + G`` after that extension.
        Strict inequality preserves an explicit margin at the SQL ``<=`` edge.
        """

        violations = [*lease_policy.errors, *self.errors]
        required: float | None = None
        available: float | None = None
        slack: float | None = None
        shutdown_bound = self.shutdown_bound_seconds
        if (
            not violations
            and lease_policy.heartbeat_seconds is not None
            and lease_policy.lease_seconds is not None
            and lease_policy.reclaim_grace_seconds is not None
            and self.heartbeat_timeout_seconds is not None
            and self.safety_margin_seconds is not None
            and shutdown_bound is not None
        ):
            required = (
                float(lease_policy.heartbeat_seconds)
                + float(self.heartbeat_timeout_seconds)
                + shutdown_bound
                + float(self.safety_margin_seconds)
            )
            available = float(
                lease_policy.lease_seconds + lease_policy.reclaim_grace_seconds
            )
            slack = available - required
            if not all(math.isfinite(value) for value in (required, available, slack)):
                violations.append("lease overlap arithmetic must remain finite")
            elif required >= available:
                violations.append(
                    "heartbeat detection plus process shutdown and safety margin "
                    "must be strictly less than lease duration plus reclaim grace"
                )

        return {
            "version": EXECUTION_FENCE_VERSION,
            "revision": EXECUTION_FENCE_REVISION,
            "equation": "H + D + U + M < L + G",
            "isolation_mode": self.isolation_mode,
            "lease_seconds": lease_policy.lease_seconds,
            "heartbeat_seconds": lease_policy.heartbeat_seconds,
            "heartbeat_timeout_seconds": self.heartbeat_timeout_seconds,
            "cancellation_grace_seconds": self.cancellation_grace_seconds,
            "terminate_grace_seconds": self.terminate_grace_seconds,
            "kill_grace_seconds": self.kill_grace_seconds,
            "shutdown_bound_seconds": shutdown_bound,
            "safety_margin_seconds": self.safety_margin_seconds,
            "reclaim_grace_seconds": lease_policy.reclaim_grace_seconds,
            "required_seconds": required,
            "available_seconds": available,
            "slack_seconds": slack,
            "provider_limits": dict(self.provider_limits),
            "safe": not violations and required is not None and available is not None,
            "violations": violations,
        }


@dataclass(frozen=True, slots=True)
class LeaseAttempt:
    """Opaque ownership proof returned by the atomic claim statement."""

    owner: str
    token: str
    leased_at: datetime
    heartbeat_at: datetime
    expires_at: datetime
    attempt_count: int

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "LeaseAttempt":
        """Build a lease attempt from the atomic claim ``RETURNING`` row."""

        return cls(
            owner=str(record["deepsearch_lease_owner"]),
            token=str(record["deepsearch_lease_token"]),
            leased_at=record["deepsearch_leased_at"],
            heartbeat_at=record["deepsearch_heartbeat_at"],
            expires_at=record["deepsearch_lease_expires_at"],
            attempt_count=int(record["deepsearch_attempt_count"]),
        )

    def persistence_key(self, mission_id: str) -> str:
        """Return the stable idempotency key for this mission attempt."""

        material = (
            f"{LEASE_CONTRACT_VERSION}:{mission_id}:{self.attempt_count}:{self.token}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def audit_dict(self, mission_id: str) -> dict[str, Any]:
        """Return a redacted claim-time snapshot for persisted telemetry.

        ``LeaseAttempt`` is immutable and reflects the row returned by the
        atomic claim. Heartbeats can extend the database lease later, so these
        timestamp names intentionally say ``claim_*`` rather than implying
        they describe the final heartbeat or terminal row.
        """

        return {
            "contract_version": LEASE_CONTRACT_VERSION,
            "contract_revision": LEASE_CONTRACT_REVISION,
            "snapshot_phase": "claim",
            "owner": self.owner,
            "attempt_count": self.attempt_count,
            "claim_leased_at": self.leased_at.isoformat(),
            "claim_heartbeat_at": self.heartbeat_at.isoformat(),
            "claim_expires_at": self.expires_at.isoformat(),
            "token_fingerprint": hashlib.sha256(
                self.token.encode("utf-8")
            ).hexdigest()[:12],
            "persistence_key": self.persistence_key(mission_id),
        }


__all__ = [
    "EXECUTION_FENCE_REVISION",
    "EXECUTION_FENCE_VERSION",
    "LEASE_COLUMN_TYPES",
    "LEASE_CONTRACT_REVISION",
    "LEASE_CONTRACT_VERSION",
    "LEASE_INDEX_SPECS",
    "LEASE_NON_NULL_DEFAULTS",
    "LEASE_NULLABLE_COLUMNS",
    "LeaseAttempt",
    "ExecutionFencePolicy",
    "LeaseIndexSpec",
    "LeasePolicy",
]
