"""Fail-closed worker startup preflight with redacted health reporting."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from deepsearch.config import DeepSearchSettings
from deepsearch.tracelab.auth import TraceLabAuth, is_secure_tracelab_url
from deepsearch.tracelab.evidence_ledger import EvidenceLedgerClient
from deepsearch.tracelab.receipt import CompletionReceiptClient

from .database import DatabaseClient
from .lease import (
    EXECUTION_FENCE_VERSION,
    LEASE_COLUMN_TYPES,
    LEASE_CONTRACT_VERSION,
    LEASE_INDEX_SPECS,
    LEASE_NON_NULL_DEFAULTS,
    LEASE_NULLABLE_COLUMNS,
    ExecutionFencePolicy,
    LeasePolicy,
)

PREFLIGHT_VERSION = 5
SCHEMA_CAPABILITY_SQL = """
    SELECT
        column_name,
        data_type,
        character_maximum_length,
        is_nullable,
        column_default
    FROM information_schema.columns
    WHERE table_schema = current_schema()
      AND table_name = 'missions'
      AND column_name = ANY($1::text[])
"""
INDEX_CAPABILITY_SQL = """
    SELECT
        indexes.indexname AS index_name,
        index_catalog.indisunique AS is_unique,
        ARRAY(
            SELECT pg_get_indexdef(
                index_catalog.indexrelid,
                ordinal.position,
                TRUE
            )
            FROM generate_series(
                1,
                index_catalog.indnkeyatts
            ) AS ordinal(position)
            ORDER BY ordinal.position
        ) AS columns,
        pg_get_expr(
            index_catalog.indpred,
            index_catalog.indrelid,
            TRUE
        ) AS predicate
    FROM pg_indexes AS indexes
    JOIN pg_namespace AS schema_catalog
      ON schema_catalog.nspname = indexes.schemaname
    JOIN pg_class AS index_relation
      ON index_relation.relnamespace = schema_catalog.oid
     AND index_relation.relname = indexes.indexname
    JOIN pg_index AS index_catalog
      ON index_catalog.indexrelid = index_relation.oid
    WHERE indexes.schemaname = current_schema()
      AND indexes.tablename = 'missions'
      AND indexes.indexname = ANY($1::text[])
"""
EVIDENCE_OUTBOX_CAPABILITY_SQL = """
    WITH target AS (
        SELECT relation_catalog.oid
        FROM pg_class AS relation_catalog
        JOIN pg_namespace AS schema_catalog
          ON schema_catalog.oid = relation_catalog.relnamespace
        WHERE schema_catalog.nspname = current_schema()
          AND relation_catalog.relname = 'deepsearch_evidence_outbox'
          AND relation_catalog.relkind = 'r'
          AND relation_catalog.relpersistence = 'p'
          AND NOT relation_catalog.relrowsecurity
          AND NOT relation_catalog.relforcerowsecurity
    ),
    column_catalog AS (
        SELECT
            attribute_catalog.attname,
            attribute_catalog.attnum,
            attribute_catalog.attnotnull,
            format_type(
                attribute_catalog.atttypid,
                attribute_catalog.atttypmod
            ) AS formatted_type,
            pg_get_expr(
                default_catalog.adbin,
                default_catalog.adrelid,
                TRUE
            ) AS default_expression
        FROM target
        JOIN pg_attribute AS attribute_catalog
          ON attribute_catalog.attrelid = target.oid
        LEFT JOIN pg_attrdef AS default_catalog
          ON default_catalog.adrelid = attribute_catalog.attrelid
         AND default_catalog.adnum = attribute_catalog.attnum
        WHERE attribute_catalog.attnum > 0
          AND NOT attribute_catalog.attisdropped
    ),
    constraint_catalog AS (
        SELECT constraint_row.*
        FROM target
        JOIN pg_constraint AS constraint_row
          ON constraint_row.conrelid = target.oid
    ),
    nonconstraint_index_catalog AS (
        SELECT
            index_relation.relname AS index_name,
            access_method.amname AS access_method,
            index_row.*,
            target.oid AS table_oid
        FROM target
        JOIN pg_index AS index_row
          ON index_row.indrelid = target.oid
        JOIN pg_class AS index_relation
          ON index_relation.oid = index_row.indexrelid
        JOIN pg_am AS access_method
          ON access_method.oid = index_relation.relam
        WHERE NOT EXISTS (
            SELECT 1
            FROM pg_constraint AS backing_constraint
            WHERE backing_constraint.conindid = index_row.indexrelid
        )
    )
    SELECT
        EXISTS (SELECT 1 FROM target) AS table_present,
        COALESCE(
            (
                SELECT ARRAY_AGG(
                    attname || '|' || formatted_type || '|' ||
                    CASE WHEN attnotnull THEN 'not_null' ELSE 'nullable' END
                    ORDER BY attnum
                ) = $1::text[]
                FROM column_catalog
            ),
            FALSE
        ) AS columns_exact,
        COALESCE(
            (
                SELECT ARRAY_AGG(
                    attname || '=' || LOWER(
                        REGEXP_REPLACE(
                            default_expression,
                            '[[:space:]]+',
                            '',
                            'g'
                        )
                    )
                    ORDER BY attnum
                ) FILTER (WHERE default_expression IS NOT NULL) = $2::text[]
                FROM column_catalog
            ),
            FALSE
        ) AS defaults_exact,
        COALESCE(
            (
                SELECT COUNT(*) = 1 AND BOOL_AND(
                    conname = 'pk_deepsearch_evidence_outbox'
                    AND ARRAY(
                        SELECT key_attribute.attname
                        FROM UNNEST(conkey) WITH ORDINALITY
                            AS key_column(attnum, ordinal_position)
                        JOIN pg_attribute AS key_attribute
                          ON key_attribute.attrelid = conrelid
                         AND key_attribute.attnum = key_column.attnum
                        ORDER BY key_column.ordinal_position
                    ) = ARRAY['mission_id', 'deepsearch_job_id']::name[]
                    AND NOT condeferrable
                    AND NOT condeferred
                )
                FROM constraint_catalog
                WHERE contype = 'p'
            ),
            FALSE
        ) AS primary_key_exact,
        COALESCE(
            (
                SELECT COUNT(*) = 1 AND BOOL_AND(
                    conname = 'fk_deepsearch_evidence_outbox_mission'
                    AND ARRAY(
                        SELECT key_attribute.attname
                        FROM UNNEST(conkey) WITH ORDINALITY
                            AS key_column(attnum, ordinal_position)
                        JOIN pg_attribute AS key_attribute
                          ON key_attribute.attrelid = conrelid
                         AND key_attribute.attnum = key_column.attnum
                        ORDER BY key_column.ordinal_position
                    ) = ARRAY['mission_id']::name[]
                    AND confrelid = to_regclass(
                        format('%I.%I', current_schema(), 'missions')
                    )
                    AND ARRAY(
                        SELECT key_attribute.attname
                        FROM UNNEST(confkey) WITH ORDINALITY
                            AS key_column(attnum, ordinal_position)
                        JOIN pg_attribute AS key_attribute
                          ON key_attribute.attrelid = confrelid
                         AND key_attribute.attnum = key_column.attnum
                        ORDER BY key_column.ordinal_position
                    ) = ARRAY['id']::name[]
                    AND confdeltype = 'c'
                    AND confupdtype = 'a'
                    AND confmatchtype = 's'
                    AND NOT condeferrable
                    AND NOT condeferred
                )
                FROM constraint_catalog
                WHERE contype = 'f'
            ),
            FALSE
        ) AS foreign_key_exact,
        COALESCE(
            (
                SELECT ARRAY_AGG(
                    conname || '=' || LOWER(
                        REGEXP_REPLACE(
                            pg_get_expr(conbin, conrelid, TRUE),
                            '[[:space:]]+',
                            '',
                            'g'
                        )
                    )
                    ORDER BY conname
                ) = $3::text[]
                FROM constraint_catalog
                WHERE contype = 'c'
            ),
            FALSE
        ) AS checks_exact,
        (SELECT COUNT(*) FROM constraint_catalog) = 11 AS constraint_set_exact,
        COALESCE(
            (
                SELECT COUNT(*) = 1 AND BOOL_AND(
                    index_name = 'ix_deepsearch_evidence_outbox_delivery'
                    AND access_method = 'btree'
                    AND NOT indisunique
                    AND NOT indisprimary
                    AND indisvalid
                    AND indisready
                    AND indislive
                    AND indexprs IS NULL
                    AND indpred IS NULL
                    AND indnkeyatts = 3
                    AND indnatts = 3
                    AND ARRAY(
                        SELECT key_attribute.attname
                        FROM UNNEST(indkey) WITH ORDINALITY
                            AS key_column(attnum, ordinal_position)
                        JOIN pg_attribute AS key_attribute
                          ON key_attribute.attrelid = table_oid
                         AND key_attribute.attnum = key_column.attnum
                        WHERE key_column.ordinal_position <= indnkeyatts
                        ORDER BY key_column.ordinal_position
                    ) = $4::name[]
                )
                FROM nonconstraint_index_catalog
            ),
            FALSE
        ) AS delivery_index_exact,
        COALESCE(
            (
                SELECT
                    has_table_privilege(current_user, oid, 'SELECT')
                    AND has_table_privilege(current_user, oid, 'INSERT')
                    AND has_table_privilege(current_user, oid, 'UPDATE')
                FROM target
            ),
            FALSE
        ) AS privileges_present,
        COALESCE(
            to_regprocedure('gen_random_uuid()') IS NOT NULL
            AND has_function_privilege(
                current_user,
                to_regprocedure('gen_random_uuid()'),
                'EXECUTE'
            ),
            FALSE
        ) AS uuid_function_ready
"""
_EVIDENCE_OUTBOX_COLUMN_SIGNATURES = (
    "mission_id|uuid|not_null",
    "deepsearch_job_id|character varying(100)|not_null",
    "deepsearch_result_key|text|not_null",
    "mission_attempt_count|integer|not_null",
    "terminal_status|character varying(32)|not_null",
    "schema_version|smallint|not_null",
    "state|character varying(16)|not_null",
    "delivery_attempt_count|integer|not_null",
    "next_attempt_at|timestamp with time zone|not_null",
    "lease_token|uuid|nullable",
    "lease_expires_at|timestamp with time zone|nullable",
    "acked_at|timestamp with time zone|nullable",
    "last_http_status|smallint|nullable",
    "last_error_code|character varying(100)|nullable",
    "created_at|timestamp with time zone|not_null",
    "updated_at|timestamp with time zone|not_null",
)
_EVIDENCE_OUTBOX_DEFAULT_SIGNATURES = (
    "schema_version=1",
    "state='pending'::charactervarying",
    "delivery_attempt_count=0",
    "next_attempt_at=now()",
    "created_at=now()",
    "updated_at=now()",
)
_EVIDENCE_OUTBOX_CHECK_SIGNATURES = (
    "ck_deepsearch_evidence_outbox_delivery_attempts="
    "delivery_attempt_count>=0",
    "ck_deepsearch_evidence_outbox_http_status="
    "last_http_statusisnullorlast_http_status>=100andlast_http_status<=599",
    "ck_deepsearch_evidence_outbox_nonempty_job="
    "length(trim(bothfromdeepsearch_job_id))>0",
    "ck_deepsearch_evidence_outbox_nonempty_result_key="
    "length(trim(bothfromdeepsearch_result_key))>0",
    "ck_deepsearch_evidence_outbox_positive_attempt=mission_attempt_count>0",
    "ck_deepsearch_evidence_outbox_schema_version=schema_version=1",
    "ck_deepsearch_evidence_outbox_state="
    "state::text=any(array['pending'::charactervarying,"
    "'leased'::charactervarying,'acked'::charactervarying,"
    "'dead_letter'::charactervarying]::text[])",
    "ck_deepsearch_evidence_outbox_state_coherence="
    "state::text='leased'::textandlease_tokenisnotnulland"
    "lease_expires_atisnotnullandnext_attempt_at=lease_expires_atand"
    "acked_atisnullorstate::text='acked'::textandlease_tokenisnulland"
    "lease_expires_atisnullandacked_atisnotnullor(state::text=any(array["
    "'pending'::charactervarying,'dead_letter'::charactervarying]::text[]))"
    "andlease_tokenisnullandlease_expires_atisnullandacked_atisnull",
    "ck_deepsearch_evidence_outbox_terminal_status="
    "terminal_status::text=any(array['completed'::charactervarying,"
    "'validation_failed'::charactervarying]::text[])",
)
_EVIDENCE_OUTBOX_INDEX_COLUMNS = ("state", "next_attempt_at", "created_at")
# ``project_id`` is not a claim scope - it is the association the result is
# written back onto. The column must exist so the claim can refuse orphan rows.
_PROJECT_COLUMN_TYPES = {"project_id": frozenset({"uuid"})}
_RESULT_COLUMN_TYPES = {
    # TraceLab stores the DeepSearch execution correlation identifier as
    # ``varchar(100)``. Terminal writer SQL consumes this column atomically
    # with the result, so startup must fail before polling if it is absent.
    "deepsearch_job_id": frozenset({"character varying"}),
}
_RESULT_COLUMN_LENGTHS = {"deepsearch_job_id": 100}
MISSION_AUTHORING_COLUMN_TYPES: dict[str, frozenset[str]] = {
    "background": frozenset({"text"}),
    "focus": frozenset({"text"}),
    "references": frozenset({"jsonb"}),
    "required_entities": frozenset({"jsonb"}),
    "excluded_entities": frozenset({"jsonb"}),
    "expected_output_schema": frozenset({"jsonb"}),
    "coverage_thresholds": frozenset({"jsonb"}),
    "validation_thresholds": frozenset({"jsonb"}),
    "deliverable_format": frozenset({"text"}),
    "max_loops": frozenset({"integer"}),
    "min_loops": frozenset({"integer"}),
    "constraints": frozenset({"jsonb"}),
}
_REQUIRED_COLUMN_TYPES = {
    **_PROJECT_COLUMN_TYPES,
    **_RESULT_COLUMN_TYPES,
    **MISSION_AUTHORING_COLUMN_TYPES,
    **LEASE_COLUMN_TYPES,
}
_NULLABLE_RUNTIME_COLUMNS = (
    frozenset(LEASE_NULLABLE_COLUMNS)
    | frozenset(_RESULT_COLUMN_TYPES)
    | frozenset(MISSION_AUTHORING_COLUMN_TYPES)
)
_PLACEHOLDER_SECRETS = frozenset(
    {
        "changeme",
        "placeholder",
        "test",
        "test-key",
        "your-api-key",
        "your-key-here",
    }
)


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """One redacted startup readiness check."""

    ok: bool
    detail: str
    data: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": self.ok, "detail": self.detail}
        if self.data is not None:
            payload["data"] = dict(self.data)
        return payload


@dataclass(frozen=True, slots=True)
class WorkerPreflightReport:
    """Single health record emitted before the poller can be constructed."""

    checked_at: datetime
    checks: Mapping[str, PreflightCheck]

    @property
    def ready(self) -> bool:
        return all(check.ok for check in self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PREFLIGHT_VERSION,
            "lease_contract_version": LEASE_CONTRACT_VERSION,
            "execution_fence_version": EXECUTION_FENCE_VERSION,
            "ready": self.ready,
            "checked_at": self.checked_at.isoformat(),
            "checks": {
                name: check.to_dict() for name, check in self.checks.items()
            },
        }


def _valid_secret(value: str | None) -> bool:
    text = str(value or "").strip()
    lowered = text.lower()
    return (
        len(text) >= 8
        and lowered not in _PLACEHOLDER_SECRETS
        and not lowered.startswith(("your_", "your-", "replace-", "replace_"))
    )


def _schema_check(rows: Any) -> PreflightCheck:
    actual: dict[str, dict[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            try:
                name = str(row["column_name"] or "")
                data_type = str(row["data_type"] or "")
            except (KeyError, TypeError):
                continue
            if not name:
                continue
            try:
                is_nullable = str(row["is_nullable"] or "").upper()
                column_default = row["column_default"]
            except (KeyError, TypeError):
                is_nullable = ""
                column_default = None
            try:
                character_maximum_length = row["character_maximum_length"]
            except (KeyError, TypeError):
                character_maximum_length = None
            actual[name] = {
                "data_type": data_type,
                "character_maximum_length": character_maximum_length,
                "is_nullable": is_nullable,
                "column_default": column_default,
            }

    missing = sorted(set(_REQUIRED_COLUMN_TYPES) - set(actual))
    mismatched = sorted(
        name
        for name, allowed in _REQUIRED_COLUMN_TYPES.items()
        if name in actual and actual[name]["data_type"] not in allowed
    )
    length_mismatched = sorted(
        name
        for name, expected in _RESULT_COLUMN_LENGTHS.items()
        if name in actual
        and actual[name]["character_maximum_length"] != expected
    )
    nullability_mismatched = sorted(
        name
        for name in _NULLABLE_RUNTIME_COLUMNS
        if name in actual and actual[name]["is_nullable"] != "YES"
    )
    nullability_mismatched.extend(
        sorted(
            name
            for name in LEASE_NON_NULL_DEFAULTS
            if name in actual and actual[name]["is_nullable"] != "NO"
        )
    )
    default_mismatched = sorted(
        name
        for name, expected in LEASE_NON_NULL_DEFAULTS.items()
        if name in actual
        and _normalize_default(actual[name]["column_default"]) != expected
    )
    if (
        missing
        or mismatched
        or length_mismatched
        or nullability_mismatched
        or default_mismatched
    ):
        parts = []
        if missing:
            parts.append("missing columns: " + ", ".join(missing))
        if mismatched:
            parts.append("type mismatch: " + ", ".join(mismatched))
        if length_mismatched:
            parts.append("length mismatch: " + ", ".join(length_mismatched))
        if nullability_mismatched:
            parts.append(
                "nullability mismatch: "
                + ", ".join(sorted(set(nullability_mismatched)))
            )
        if default_mismatched:
            parts.append("default mismatch: " + ", ".join(default_mismatched))
        return PreflightCheck(False, "; ".join(parts))
    return PreflightCheck(
        True,
        (
            f"missions authoring, {LEASE_CONTRACT_VERSION} constraints, "
            "and global non-orphan association available"
        ),
    )


def _normalize_default(value: Any) -> str:
    """Normalize PostgreSQL's equivalent rendering of a scalar default."""

    text = "" if value is None else "".join(str(value).split())
    for cast in ("::integer", "::int4", "::pg_catalog.int4"):
        if text.endswith(cast):
            text = text[: -len(cast)]
            break
    while len(text) >= 2 and text[0] == "(" and text[-1] == ")":
        text = text[1:-1]
    return text.strip("'")


def _normalize_predicate(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).replace('"', "").split())
    while len(text) >= 2 and text[0] == "(" and text[-1] == ")":
        text = text[1:-1].strip()
    return text.upper() or None


def _index_check(rows: Any) -> PreflightCheck:
    actual: dict[str, dict[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            try:
                name = str(row["index_name"] or "")
                columns = tuple(str(value).replace('"', "") for value in row["columns"])
                unique = row["is_unique"]
                predicate = row["predicate"]
            except (KeyError, TypeError):
                continue
            if name:
                actual[name] = {
                    "columns": columns,
                    "unique": unique if isinstance(unique, bool) else None,
                    "predicate": _normalize_predicate(predicate),
                }

    missing = sorted(set(LEASE_INDEX_SPECS) - set(actual))
    mismatched: list[str] = []
    for name, expected in LEASE_INDEX_SPECS.items():
        current = actual.get(name)
        if current is None:
            continue
        expected_predicate = _normalize_predicate(expected.predicate)
        if (
            current["columns"] != expected.columns
            or current["unique"] is not expected.unique
            or current["predicate"] != expected_predicate
        ):
            mismatched.append(name)

    if missing or mismatched:
        parts = []
        if missing:
            parts.append("missing indexes: " + ", ".join(missing))
        if mismatched:
            parts.append("definition mismatch: " + ", ".join(sorted(mismatched)))
        return PreflightCheck(False, "; ".join(parts))
    return PreflightCheck(
        True,
        "migration-039 indexes support exact global non-orphan claim order",
    )


def _evidence_outbox_check(row: Any) -> PreflightCheck:
    """Validate exact migration-043 DDL and worker write capability."""

    try:
        ready = all(
            row[name] is True
            for name in (
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
            )
        )
    except (KeyError, TypeError):
        ready = False
    return PreflightCheck(
        ready,
        (
            "migration-043 durable Evidence Ledger outbox available"
            if ready
            else "migration-043 durable Evidence Ledger outbox is missing or incompatible"
        ),
    )


def _evidence_transport_ready(
    client: EvidenceLedgerClient,
    auth: TraceLabAuth,
) -> bool:
    """Require one API-key-authenticated wire attempt per durable lease."""

    try:
        return bool(
            client.auth is auth
            and client.is_configured
            and client.max_attempts == 1
            and client.base_url == auth.base_url.rstrip("/")
        )
    except Exception:  # noqa: BLE001 - emit one redacted readiness fact
        return False


async def run_worker_preflight(
    db_client: DatabaseClient,
    *,
    settings: DeepSearchSettings,
    lease_policy: LeasePolicy,
    execution_fence_policy: ExecutionFencePolicy,
    tracelab_auth: TraceLabAuth,
    receipt_client: CompletionReceiptClient,
    evidence_ledger_client: EvidenceLedgerClient,
    database_config_error: str | None = None,
) -> WorkerPreflightReport:
    """Validate all claim prerequisites without mutating mission state."""

    from deepsearch.llm.runtime import ModelConfigurationError, resolve_effective_model

    try:
        effective = resolve_effective_model(settings)
        credential_ok = _valid_secret(effective.api_key)
        model_check = PreflightCheck(
            credential_ok,
            ("Effective model configuration valid (no inference probe)" if credential_ok
             else "Selected provider credential missing or structurally invalid"),
            data=effective.public_metadata(),
        )
    except ModelConfigurationError as exc:
        model_check = PreflightCheck(False, str(exc))
    linkup_api_key = getattr(settings, "linkup_api_key", None)
    secure_tracelab_destination = is_secure_tracelab_url(tracelab_auth.base_url)
    service_auth_ok = False
    if tracelab_auth.is_configured and secure_tracelab_destination:
        try:
            await asyncio.to_thread(tracelab_auth.require_service_role)
            service_auth_ok = True
        except Exception:  # noqa: BLE001 - emit one redacted readiness fact
            service_auth_ok = False
    receipt_ok = (
        receipt_client.is_configured
        and secure_tracelab_destination
        and is_secure_tracelab_url(receipt_client.base_url)
        and _valid_secret(receipt_client.secret)
        and receipt_client.base_url == tracelab_auth.base_url
    )
    evidence_transport_ok = _evidence_transport_ready(
        evidence_ledger_client,
        tracelab_auth,
    )
    overlap_safety = execution_fence_policy.evaluate(lease_policy)
    checks: dict[str, PreflightCheck] = {
        "tracelab_transport_security": PreflightCheck(
            secure_tracelab_destination,
            (
                "TraceLab transport uses HTTPS or an explicit loopback host"
                if secure_tracelab_destination
                else "TraceLab transport requires HTTPS outside localhost"
            ),
        ),
        "effective_model": model_check,
        "linkup_credentials": PreflightCheck(
            _valid_secret(linkup_api_key),
            (
                "LINKUP_API_KEY configured"
                if _valid_secret(linkup_api_key)
                else "LINKUP_API_KEY missing or structurally invalid"
            ),
        ),
        "tracelab_service_auth": PreflightCheck(
            service_auth_ok,
            (
                "TraceLab service principal verified through /auth/me"
                if service_auth_ok
                else "TraceLab service authentication missing, invalid, or non-service"
            ),
        ),
        "completion_receipt": PreflightCheck(
            receipt_ok,
            (
                "TraceLab receipt destination and HMAC secret configured"
                if receipt_ok
                else "TraceLab receipt destination or HMAC secret missing or mismatched"
            ),
        ),
        "database_configuration": PreflightCheck(
            database_config_error is None,
            database_config_error or "DATABASE_URL structurally valid",
        ),
        "lease_policy": PreflightCheck(
            lease_policy.valid,
            "lease timing and attempt policy valid"
            if lease_policy.valid
            else "; ".join(lease_policy.errors),
            data={
                "lease_seconds": lease_policy.lease_seconds,
                "heartbeat_seconds": lease_policy.heartbeat_seconds,
                "max_attempts": lease_policy.max_attempts,
                "reclaim_grace_seconds": lease_policy.reclaim_grace_seconds,
            },
        ),
        "lease_overlap_safety": PreflightCheck(
            bool(overlap_safety["safe"]),
            (
                "spawned execution terminates before reclaim eligibility"
                if overlap_safety["safe"]
                else "; ".join(overlap_safety["violations"])
            ),
            data=overlap_safety,
        ),
    }

    database_ok = False
    if database_config_error is None:
        try:
            health = await db_client.health_check()
            database_ok = bool(health.healthy)
        except Exception:  # noqa: BLE001 - report one redacted health fact
            database_ok = False
    checks["database"] = PreflightCheck(
        database_ok,
        "database health query succeeded"
        if database_ok
        else "database health query failed",
    )

    if database_ok:
        try:
            rows = await db_client.fetchall(
                SCHEMA_CAPABILITY_SQL,
                list(_REQUIRED_COLUMN_TYPES),
            )
            checks["schema_capabilities"] = _schema_check(rows)
        except Exception:  # noqa: BLE001 - no SQL/credential details in health
            checks["schema_capabilities"] = PreflightCheck(
                False,
                "missions schema capability query failed",
            )
        try:
            rows = await db_client.fetchall(
                INDEX_CAPABILITY_SQL,
                list(LEASE_INDEX_SPECS),
            )
            checks["lease_indexes"] = _index_check(rows)
        except Exception:  # noqa: BLE001 - no SQL/credential details in health
            checks["lease_indexes"] = PreflightCheck(
                False,
                "missions lease index capability query failed",
            )
        try:
            row = await db_client.fetchone(
                EVIDENCE_OUTBOX_CAPABILITY_SQL,
                list(_EVIDENCE_OUTBOX_COLUMN_SIGNATURES),
                list(_EVIDENCE_OUTBOX_DEFAULT_SIGNATURES),
                list(_EVIDENCE_OUTBOX_CHECK_SIGNATURES),
                list(_EVIDENCE_OUTBOX_INDEX_COLUMNS),
            )
            outbox_check = _evidence_outbox_check(row)
        except Exception:  # noqa: BLE001 - no SQL/credential details in health
            outbox_check = PreflightCheck(
                False,
                "migration-043 durable Evidence Ledger outbox query failed",
            )
    else:
        checks["schema_capabilities"] = PreflightCheck(
            False,
            "not checked because database health failed",
        )
        checks["lease_indexes"] = PreflightCheck(
            False,
            "not checked because database health failed",
        )
        outbox_check = PreflightCheck(
            False,
            "not checked because database health failed",
        )

    evidence_ready = evidence_transport_ok and outbox_check.ok
    checks["evidence_ledger_delivery"] = PreflightCheck(
        evidence_ready,
        (
            "service API-key transport and migration-043 durable outbox ready"
            if evidence_ready
            else (
                outbox_check.detail
                if evidence_transport_ok
                else "Evidence Ledger requires the shared TraceLab service API key, "
                "one wire attempt per lease, and the credential-owned HTTPS origin"
            )
        ),
    )

    return WorkerPreflightReport(
        checked_at=datetime.now(timezone.utc),
        checks=checks,
    )


__all__ = [
    "PREFLIGHT_VERSION",
    "EVIDENCE_OUTBOX_CAPABILITY_SQL",
    "INDEX_CAPABILITY_SQL",
    "MISSION_AUTHORING_COLUMN_TYPES",
    "SCHEMA_CAPABILITY_SQL",
    "PreflightCheck",
    "WorkerPreflightReport",
    "run_worker_preflight",
]
