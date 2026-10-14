"""S87 M05 startup preflight and redaction invariants."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepsearch.config import DeepSearchSettings
from deepsearch.worker.database import DatabaseClient, DatabaseConfig, HealthStatus
from deepsearch.worker.lease import (
    LEASE_COLUMN_TYPES,
    LEASE_INDEX_SPECS,
    ExecutionFencePolicy,
    LeasePolicy,
)
from deepsearch.worker.main import DeepSearchWorker, WorkerError
from deepsearch.worker.preflight import (
    EVIDENCE_OUTBOX_CAPABILITY_SQL,
    INDEX_CAPABILITY_SQL,
    MISSION_AUTHORING_COLUMN_TYPES,
    PREFLIGHT_VERSION,
    SCHEMA_CAPABILITY_SQL,
    run_worker_preflight,
)

_VALID_POLICY = LeasePolicy(
    lease_seconds=120,
    heartbeat_seconds=30,
    max_attempts=3,
)
_VALID_PROVIDER_LIMITS = {
    "contract": {
        "backend": "deepseek",
        "request_timeout_seconds": 300.0,
        "client_max_retries": 0,
        "configured_max_retries": 3,
        "wrapper_max_retries": 2,
        "wrapper_attempts": 3,
        "hard_bound_mode": "spawn-process-v1",
    },
    "codeagent": {
        "backend": "deepseek-openai-compatible",
        "request_timeout_seconds": 300.0,
        "client_max_retries": 0,
        "smolagents_rate_limit_retry": False,
        "synthesis_attempts": 3,
        "max_step_fallback_attempts": 1,
        "recovery_passes": 1,
        "hard_bound_mode": "spawn-process-v1",
    },
    "tools": {
        "configured_request_timeout_seconds": 60.0,
        "configured_max_retries": 3,
        "linkup_request_timeout_seconds": 60.0,
        "linkup_max_retries": 3,
        "source_fetch_request_timeout_seconds": 15.0,
        "source_fetch_max_redirects": 5,
        "url_liveness_request_timeout_seconds": 5.0,
        "url_liveness_max_requests": 3,
        "scholar_backend_request_timeout_seconds": 20.0,
        "scholar_max_backends": 2,
        "hard_bound_mode": "spawn-process-v1",
    },
}
_VALID_FENCE = ExecutionFencePolicy(
    heartbeat_timeout_seconds=10.0,
    cancellation_grace_seconds=2.0,
    terminate_grace_seconds=3.0,
    kill_grace_seconds=3.0,
    safety_margin_seconds=10.0,
    provider_limits=_VALID_PROVIDER_LIMITS,
)


class _AsyncpgLikeRecord:
    """Exercise asyncpg.Record's subscript API without Mapping inheritance."""

    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def __getitem__(self, name: str) -> object:
        return self.values[name]


def _settings(
    *,
    deepseek_api_key: str = "deepseek-secret-value",
    linkup_api_key: str = "linkup-secret-value",
) -> DeepSearchSettings:
    return DeepSearchSettings(
        llm_backend="deepseek",
        deepseek_api_key=deepseek_api_key,
        linkup_api_key=linkup_api_key,
        deepseek_model="deepseek-v4-pro",
    )


def _transport_prerequisites() -> dict[str, object]:
    """Return network-free service-auth and receipt preflight doubles."""

    base_url = "https://tracelab.example"
    auth = SimpleNamespace(
        base_url=base_url,
        is_configured=True,
        uses_api_key=True,
        require_service_role=lambda: None,
    )
    return {
        "execution_fence_policy": _VALID_FENCE,
        "tracelab_auth": auth,
        "evidence_ledger_client": SimpleNamespace(
            auth=auth,
            base_url=base_url,
            is_configured=True,
            max_attempts=1,
        ),
        "receipt_client": SimpleNamespace(
            base_url=base_url,
            secret="shared-service-secret",
            is_configured=True,
        ),
    }


def _schema_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "column_name": "project_id",
            "data_type": "uuid",
            "is_nullable": "NO",
            "column_default": None,
        },
        {
            "column_name": "deepsearch_job_id",
            "data_type": "character varying",
            "character_maximum_length": 100,
            "is_nullable": "YES",
            "column_default": None,
        },
    ]
    rows.extend(
        {
            "column_name": name,
            "data_type": next(iter(types)),
            "is_nullable": "YES",
            "column_default": None,
        }
        for name, types in MISSION_AUTHORING_COLUMN_TYPES.items()
    )
    rows.extend(
        {
            "column_name": name,
            "data_type": next(iter(types)),
            "is_nullable": "YES",
            "column_default": None,
        }
        for name, types in LEASE_COLUMN_TYPES.items()
    )
    attempt_count = next(
        row
        for row in rows
        if row["column_name"] == "deepsearch_attempt_count"
    )
    attempt_count["is_nullable"] = "NO"
    attempt_count["column_default"] = "0"
    return rows


def _index_rows() -> list[dict[str, object]]:
    return [
        {
            "index_name": name,
            "columns": list(spec.columns),
            "is_unique": spec.unique,
            "predicate": spec.predicate,
        }
        for name, spec in LEASE_INDEX_SPECS.items()
    ]


def _database() -> MagicMock:
    database = MagicMock(spec=DatabaseClient)
    database.health_check = AsyncMock(
        return_value=HealthStatus(healthy=True, message="healthy")
    )
    database.fetchall = AsyncMock(side_effect=[_schema_rows(), _index_rows()])
    database.fetchone = AsyncMock(
        return_value={
            "table_present": True,
            "columns_exact": True,
            "defaults_exact": True,
            "primary_key_exact": True,
            "foreign_key_exact": True,
            "checks_exact": True,
            "constraint_set_exact": True,
            "delivery_index_exact": True,
            "privileges_present": True,
            "uuid_function_ready": True,
        }
    )
    return database


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["deepseek", "openai"])
async def test_preflight_accepts_every_required_runtime_capability(provider) -> None:
    database = _database()
    database.fetchall.side_effect = [
        [_AsyncpgLikeRecord(row) for row in _schema_rows()],
        [_AsyncpgLikeRecord(row) for row in _index_rows()],
    ]

    from dataclasses import replace

    from deepsearch.worker.main import _provider_limit_snapshot

    settings = _settings()
    settings.llm_backend = provider
    settings.openai_api_key = "openai-offline-secret"
    prerequisites = _transport_prerequisites()
    fence = replace(_VALID_FENCE, provider_limits=_provider_limit_snapshot(settings, provider))
    prerequisites["execution_fence_policy"] = fence
    report = await run_worker_preflight(
        database, settings=settings, lease_policy=_VALID_POLICY,
        **prerequisites,
    )

    assert report.ready is True
    assert report.to_dict()["version"] == PREFLIGHT_VERSION == 5
    assert len(report.checks) == 12
    assert all(check.ok for check in report.checks.values())
    assert "global non-orphan" in report.checks["schema_capabilities"].detail
    assert "global non-orphan" in report.checks["lease_indexes"].detail
    overlap = report.checks["lease_overlap_safety"].to_dict()["data"]
    assert overlap["equation"] == "H + D + U + M < L + G"
    assert overlap["safe"] is True
    assert overlap["provider_limits"] == fence.provider_limits
    schema_call, index_call = database.fetchall.await_args_list
    assert schema_call.args[0] == SCHEMA_CAPABILITY_SQL
    assert set(schema_call.args[1]) == {
        "project_id",
        "deepsearch_job_id",
        *MISSION_AUTHORING_COLUMN_TYPES,
        *LEASE_COLUMN_TYPES,
    }
    assert index_call.args == (
        INDEX_CAPABILITY_SQL,
        list(LEASE_INDEX_SPECS),
    )
    outbox_call = database.fetchone.await_args
    assert outbox_call.args[0] == EVIDENCE_OUTBOX_CAPABILITY_SQL
    assert [item.split("|", 1)[0] for item in outbox_call.args[1]] == [
        "mission_id",
        "deepsearch_job_id",
        "deepsearch_result_key",
        "mission_attempt_count",
        "terminal_status",
        "schema_version",
        "state",
        "delivery_attempt_count",
        "next_attempt_at",
        "lease_token",
        "lease_expires_at",
        "acked_at",
        "last_http_status",
        "last_error_code",
        "created_at",
        "updated_at",
    ]
    assert outbox_call.args[2] == [
        "schema_version=1",
        "state='pending'::charactervarying",
        "delivery_attempt_count=0",
        "next_attempt_at=now()",
        "created_at=now()",
        "updated_at=now()",
    ]
    assert len(outbox_call.args[3]) == 9
    assert outbox_call.args[4] == ["state", "next_attempt_at", "created_at"]


@pytest.mark.asyncio
async def test_preflight_rejects_overlap_equation_before_poller_construction() -> None:
    """Finite individual settings are still unsafe when their envelope overlaps."""

    database = _database()
    transports = _transport_prerequisites()
    transports["execution_fence_policy"] = ExecutionFencePolicy(
        heartbeat_timeout_seconds=60.0,
        cancellation_grace_seconds=20.0,
        terminate_grace_seconds=20.0,
        kill_grace_seconds=20.0,
        safety_margin_seconds=10.0,
        provider_limits=_VALID_PROVIDER_LIMITS,
    )

    report = await run_worker_preflight(
        database,
        settings=_settings(),  # type: ignore[arg-type]
        lease_policy=LeasePolicy(120, 30, 3, reclaim_grace_seconds=0),
        **transports,  # type: ignore[arg-type]
    )

    check = report.checks["lease_overlap_safety"]
    assert report.ready is False
    assert check.ok is False
    assert check.to_dict()["data"]["required_seconds"] == 160.0
    assert check.to_dict()["data"]["available_seconds"] == 120.0


@pytest.mark.asyncio
async def test_preflight_rejects_nonservice_identity_without_leaking_auth_detail() -> None:
    """A working human credential is still unsafe for mission-log writes."""

    database = _database()
    raw_detail = "owner credential secret detail"

    def _reject_role() -> None:
        raise RuntimeError(raw_detail)

    transports = _transport_prerequisites()
    transports["tracelab_auth"] = SimpleNamespace(
        base_url="https://tracelab.example",
        is_configured=True,
        require_service_role=_reject_role,
    )

    report = await run_worker_preflight(
        database,
        settings=_settings(),  # type: ignore[arg-type]
        lease_policy=_VALID_POLICY,
        **transports,  # type: ignore[arg-type]
    )

    serialized = json.dumps(report.to_dict(), sort_keys=True)
    assert report.ready is False
    assert report.checks["tracelab_service_auth"].ok is False
    assert "non-service" in report.checks["tracelab_service_auth"].detail
    assert raw_detail not in serialized


@pytest.mark.asyncio
async def test_preflight_rejects_external_plain_http_before_authentication() -> None:
    """Startup must fail before sending a service credential over plaintext."""

    database = _database()
    role_check = MagicMock()
    transports = _transport_prerequisites()
    transports["tracelab_auth"] = SimpleNamespace(
        base_url="http://tracelab.example",
        is_configured=True,
        require_service_role=role_check,
    )
    transports["receipt_client"] = SimpleNamespace(
        base_url="http://tracelab.example",
        secret="shared-service-secret",
        is_configured=True,
    )

    report = await run_worker_preflight(
        database,
        settings=_settings(),  # type: ignore[arg-type]
        lease_policy=_VALID_POLICY,
        **transports,  # type: ignore[arg-type]
    )

    assert report.ready is False
    assert report.checks["tracelab_transport_security"].ok is False
    assert report.checks["tracelab_service_auth"].ok is False
    assert report.checks["completion_receipt"].ok is False
    role_check.assert_not_called()


@pytest.mark.asyncio
async def test_preflight_rejects_missing_or_cross_target_receipt_secret() -> None:
    """Receipt signing cannot silently target a different TraceLab origin."""

    database = _database()
    transports = _transport_prerequisites()
    transports["receipt_client"] = SimpleNamespace(
        base_url="https://different-tracelab.example",
        secret="short",
        is_configured=True,
    )

    report = await run_worker_preflight(
        database,
        settings=_settings(),  # type: ignore[arg-type]
        lease_policy=_VALID_POLICY,
        **transports,  # type: ignore[arg-type]
    )

    assert report.ready is False
    assert report.checks["completion_receipt"].ok is False
    assert "missing or mismatched" in report.checks["completion_receipt"].detail
    assert "short" not in json.dumps(report.to_dict(), sort_keys=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "capability",
    [
        "table_present",
        "columns_exact",
        "defaults_exact",
        "primary_key_exact",
        "foreign_key_exact",
        "checks_exact",
        "constraint_set_exact",
        "delivery_index_exact",
        "privileges_present",
        "uuid_function_ready",
    ],
)
async def test_preflight_rejects_each_incompatible_outbox_capability(
    capability: str,
) -> None:
    """A same-named but unwritable or weakened 043 table is not runnable."""

    database = _database()
    row = dict(database.fetchone.return_value)
    row[capability] = False
    database.fetchone.return_value = row

    report = await run_worker_preflight(
        database,
        settings=_settings(),  # type: ignore[arg-type]
        lease_policy=_VALID_POLICY,
        **_transport_prerequisites(),  # type: ignore[arg-type]
    )

    check = report.checks["evidence_ledger_delivery"]
    assert report.ready is False
    assert check.ok is False
    assert "missing or incompatible" in check.detail


@pytest.mark.asyncio
async def test_preflight_rejects_cross_auth_or_multi_attempt_ledger_transport() -> None:
    """Durable attempts require the same API-key trust unit and one wire call."""

    database = _database()
    secret_marker = "service-secret-must-not-appear"
    transports = _transport_prerequisites()
    transports["evidence_ledger_client"] = SimpleNamespace(
        auth=SimpleNamespace(api_key=secret_marker),
        base_url="https://tracelab.example",
        is_configured=True,
        max_attempts=4,
    )

    report = await run_worker_preflight(
        database,
        settings=_settings(),  # type: ignore[arg-type]
        lease_policy=_VALID_POLICY,
        **transports,  # type: ignore[arg-type]
    )

    serialized = json.dumps(report.to_dict(), sort_keys=True)
    check = report.checks["evidence_ledger_delivery"]
    assert report.ready is False
    assert check.ok is False
    assert "one wire attempt" in check.detail
    assert secret_marker not in serialized


@pytest.mark.asyncio
async def test_preflight_fails_closed_and_never_reports_secret_values() -> None:
    database = _database()
    raw_deepseek = "replace-me-deepseek"
    raw_linkup = "your-linkup-secret"

    report = await run_worker_preflight(
        database,
        settings=_settings(  # type: ignore[arg-type]
            deepseek_api_key=raw_deepseek,
            linkup_api_key=raw_linkup,
        ),
        lease_policy=LeasePolicy(None, 60, 0),
        database_config_error="DATABASE_URL must include a database host",
        **_transport_prerequisites(),  # type: ignore[arg-type]
    )

    serialized = json.dumps(report.to_dict(), sort_keys=True)
    assert report.ready is False
    assert raw_deepseek not in serialized
    assert raw_linkup not in serialized
    assert "DEEPSEEK_API_KEY" in serialized
    assert "LINKUP_API_KEY" in serialized
    database.health_check.assert_not_awaited()
    database.fetchall.assert_not_awaited()


@pytest.mark.asyncio
async def test_preflight_rejects_missing_or_mistyped_lease_schema() -> None:
    database = _database()
    rows = _schema_rows()
    rows = [
        row
        for row in rows
        if row["column_name"] != "deepsearch_result_key"
    ]
    for row in rows:
        if row["column_name"] == "deepsearch_attempt_count":
            row["data_type"] = "text"
    database.fetchall.side_effect = [rows, _index_rows()]

    report = await run_worker_preflight(
        database,
        settings=_settings(),  # type: ignore[arg-type]
        lease_policy=_VALID_POLICY,
        **_transport_prerequisites(),  # type: ignore[arg-type]
    )

    assert report.ready is False
    detail = report.checks["schema_capabilities"].detail
    assert "missing columns: deepsearch_result_key" in detail
    assert "type mismatch: deepsearch_attempt_count" in detail


@pytest.mark.asyncio
async def test_preflight_rejects_missing_or_mistyped_authoring_schema() -> None:
    """A healthy DB is not runnable when claimed authoring columns would fail."""
    database = _database()
    rows = [
        row
        for row in _schema_rows()
        if row["column_name"] != "references"
    ]
    for row in rows:
        if row["column_name"] == "max_loops":
            row["data_type"] = "text"
    database.fetchall.side_effect = [rows, _index_rows()]

    report = await run_worker_preflight(
        database,
        settings=_settings(),  # type: ignore[arg-type]
        lease_policy=_VALID_POLICY,
        **_transport_prerequisites(),  # type: ignore[arg-type]
    )

    assert report.ready is False
    detail = report.checks["schema_capabilities"].detail
    assert "missing columns: references" in detail
    assert "type mismatch: max_loops" in detail


@pytest.mark.asyncio
async def test_preflight_rejects_nonnullable_authoring_fallback_column() -> None:
    """SQL NULL must remain available as the legacy-context fallback signal."""

    database = _database()
    rows = _schema_rows()
    references = next(
        row for row in rows if row["column_name"] == "references"
    )
    references["is_nullable"] = "NO"
    database.fetchall.side_effect = [rows, _index_rows()]

    report = await run_worker_preflight(
        database,
        settings=_settings(),  # type: ignore[arg-type]
        lease_policy=_VALID_POLICY,
        **_transport_prerequisites(),  # type: ignore[arg-type]
    )

    assert report.ready is False
    assert (
        "nullability mismatch: references"
        in report.checks["schema_capabilities"].detail
    )


@pytest.mark.asyncio
async def test_preflight_rejects_unsafe_terminal_correlation_column() -> None:
    """Terminal persistence cannot be fenced without writable job correlation."""

    database = _database()
    rows = _schema_rows()
    for row in rows:
        if row["column_name"] == "deepsearch_job_id":
            row["data_type"] = "uuid"
            row["is_nullable"] = "NO"
    database.fetchall.side_effect = [rows, _index_rows()]

    report = await run_worker_preflight(
        database,
        settings=_settings(),  # type: ignore[arg-type]
        lease_policy=_VALID_POLICY,
        **_transport_prerequisites(),  # type: ignore[arg-type]
    )

    assert report.ready is False
    detail = report.checks["schema_capabilities"].detail
    assert "type mismatch: deepsearch_job_id" in detail
    assert "nullability mismatch: deepsearch_job_id" in detail


@pytest.mark.asyncio
async def test_preflight_rejects_short_terminal_correlation_column() -> None:
    """A varchar too short for the worker UUID would fail only after research."""

    database = _database()
    rows = _schema_rows()
    job_id = next(
        row for row in rows if row["column_name"] == "deepsearch_job_id"
    )
    job_id["character_maximum_length"] = 10
    database.fetchall.side_effect = [rows, _index_rows()]

    report = await run_worker_preflight(
        database,
        settings=_settings(),  # type: ignore[arg-type]
        lease_policy=_VALID_POLICY,
        **_transport_prerequisites(),  # type: ignore[arg-type]
    )

    assert report.ready is False
    assert (
        "length mismatch: deepsearch_job_id"
        in report.checks["schema_capabilities"].detail
    )


@pytest.mark.asyncio
async def test_preflight_rejects_type_correct_but_unsafe_column_constraints() -> None:
    """Types alone cannot prove first claim and terminal lease reset are safe."""

    database = _database()
    rows = _schema_rows()
    for row in rows:
        if row["column_name"] == "deepsearch_lease_token":
            row["is_nullable"] = "NO"
        if row["column_name"] == "deepsearch_attempt_count":
            row["is_nullable"] = "YES"
            row["column_default"] = None
    database.fetchall.side_effect = [rows, _index_rows()]

    report = await run_worker_preflight(
        database,
        settings=_settings(),  # type: ignore[arg-type]
        lease_policy=_VALID_POLICY,
        **_transport_prerequisites(),  # type: ignore[arg-type]
    )

    assert report.ready is False
    detail = report.checks["schema_capabilities"].detail
    assert "type mismatch" not in detail
    assert (
        "nullability mismatch: deepsearch_attempt_count, deepsearch_lease_token"
        in detail
    )
    assert "default mismatch: deepsearch_attempt_count" in detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("index_name", "field", "value"),
    [
        (
            "missions_deepsearch_lease_token_active_uq",
            "is_unique",
            False,
        ),
        (
            "missions_deepsearch_lease_token_active_uq",
            "predicate",
            "deepsearch_lease_token IS NULL",
        ),
        (
            "missions_deepsearch_result_key_uq",
            "is_unique",
            False,
        ),
        (
            "missions_deepsearch_result_key_uq",
            "predicate",
            "deepsearch_result_key IS NULL",
        ),
    ],
)
async def test_preflight_rejects_wrong_partial_unique_index_definition(
    index_name: str,
    field: str,
    value: object,
) -> None:
    """Same-named indexes cannot weaken attempt or result idempotency."""

    database = _database()
    indexes = _index_rows()
    index = next(row for row in indexes if row["index_name"] == index_name)
    index[field] = value
    database.fetchall.side_effect = [_schema_rows(), indexes]

    report = await run_worker_preflight(
        database,
        settings=_settings(),  # type: ignore[arg-type]
        lease_policy=_VALID_POLICY,
        **_transport_prerequisites(),  # type: ignore[arg-type]
    )

    assert report.ready is False
    assert index_name in report.checks["lease_indexes"].detail
    assert "definition mismatch" in report.checks["lease_indexes"].detail


@pytest.mark.asyncio
async def test_preflight_rejects_missing_result_index() -> None:
    """A result key without its unique partial index is not a safe boundary."""

    database = _database()
    indexes = [
        row
        for row in _index_rows()
        if row["index_name"] != "missions_deepsearch_result_key_uq"
    ]
    database.fetchall.side_effect = [_schema_rows(), indexes]

    report = await run_worker_preflight(
        database,
        settings=_settings(),  # type: ignore[arg-type]
        lease_policy=_VALID_POLICY,
        **_transport_prerequisites(),  # type: ignore[arg-type]
    )

    assert report.ready is False
    assert (
        "missing indexes: missions_deepsearch_result_key_uq"
        in report.checks["lease_indexes"].detail
    )


@pytest.mark.asyncio
async def test_preflight_rejects_project_leading_claim_index() -> None:
    """Exact index order prevents stale project-scoped claim policy returning."""

    database = _database()
    indexes = _index_rows()
    claim_index = next(
        row
        for row in indexes
        if row["index_name"] == "missions_deepsearch_claim_scan_idx"
    )
    claim_index["columns"] = [
        "project_id",
        "status",
        "deepsearch_lease_expires_at",
        "queued_at",
    ]
    database.fetchall.side_effect = [_schema_rows(), indexes]

    report = await run_worker_preflight(
        database,
        settings=_settings(),  # type: ignore[arg-type]
        lease_policy=_VALID_POLICY,
        **_transport_prerequisites(),  # type: ignore[arg-type]
    )

    assert report.ready is False
    assert (
        "definition mismatch: missions_deepsearch_claim_scan_idx"
        in report.checks["lease_indexes"].detail
    )


@pytest.mark.asyncio
async def test_invalid_database_config_logs_one_health_record_before_polling(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _settings()
    database = _database()
    database.connect = AsyncMock()
    database.close = AsyncMock()

    with (
        patch("deepsearch.worker.main.DeepSearchSettings.load", return_value=settings),
        patch("deepsearch.worker.main.DatabaseClient", return_value=database),
        patch("deepsearch.worker.main.MissionPoller") as poller_class,
    ):
        transports = _transport_prerequisites()
        worker = DeepSearchWorker(
            db_config=DatabaseConfig(url="postgresql:///missing-host"),
            enable_health_server=False,
            tracelab_auth=transports["tracelab_auth"],  # type: ignore[arg-type]
            receipt_client=transports["receipt_client"],  # type: ignore[arg-type]
        )
        with caplog.at_level(logging.INFO, logger="deepsearch.worker.main"):
            with pytest.raises(WorkerError, match="preflight failed"):
                await worker.run()

    database.connect.assert_not_awaited()
    poller_class.assert_not_called()
    records = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("worker_preflight=")
    ]
    assert len(records) == 1
    assert "missing-host" not in records[0]
    assert json.loads(records[0].split("=", 1)[1])["ready"] is False


def test_invalid_database_pool_setting_is_preserved_for_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Numeric env parse failures must not escape before health reporting."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://worker@localhost/tracelab")
    monkeypatch.setenv("DB_MIN_POOL_SIZE", "not-an-integer")

    with patch("deepsearch.worker.main.DeepSearchSettings.load", return_value=_settings()):
        worker = DeepSearchWorker(
            enable_health_server=False,
        )

    assert worker._database_config_error == (
        "database pool and timeout settings must be numeric"
    )
