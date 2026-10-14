"""T42.7 — DeepSearch ↔ TraceLab boundary regression test.

Pins the DeepSearch worker's view of TraceLab's ``missions`` table against the
authoritative contract documented at TraceLab repo path
``cmos/contracts/mission-authoring-contract.md`` (TL T41.3, lockstep-updated
at TL T42.2 / alembic migration 028 / TraceLab PR #213).

The contract source of truth lives in the TraceLab repo. This file is the
DeepSearch-side mirror — same-PR-update rule applies (per inbox status_update
``55f0e64f`` from 2026-04-29): when TraceLab ships a contract-doc change, the
coordinating PR updates the constants here in lockstep. The test fails fast
if the two sides drift.

Failure modes guarded
---------------------
1. ``CLAIM_MISSION_SQL`` drifts from the frozen TraceLab column contract.
2. Postgres ``references`` is selected without the exact required quoting.
3. ``MissionRow`` loses a nullable typed authoring slot.
4. JSONB driver shape changes drop authored values during parsing.
5. A legacy ``context`` value overrides a non-null dedicated column,
   including an explicit empty list/object.
6. A parsed field disappears before the compiled mission contract.

The tests are pure Python — no Postgres connection, no network, no LLM call.
They sit inside ``tests/integration/`` because they assert worker-boundary
behaviour, not because they require live infrastructure; they fit comfortably
within the PD48 sub-2-minute integration-suite budget.
"""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime, timezone
from typing import Any, Optional, get_type_hints

import pytest

import deepsearch.mission.contract as mission_contract_module
from deepsearch.agent.contract_to_prompt import build_codeagent_prompts
from deepsearch.mission import NO_ENRICHMENT_MODE, compile_contract_from_state
from deepsearch.worker.converter import convert_row_to_state
from deepsearch.worker.poller import MissionPoller, MissionRow, _coerce_entity_list

# ---------------------------------------------------------------------------
# Canonical contract surface — DS-side mirror of TraceLab's T41.3 doc.
# Update in lockstep with TraceLab repo cmos/contracts/mission-authoring-contract.md.
# ---------------------------------------------------------------------------

# Content/system columns returned by the atomic lease-v2 claim. The twelve
# dedicated authoring columns now reach the same runtime compiler TraceLab's
# preview path targets; explicit columns are no longer a preview-only surface.
WORKER_CONTENT_COLUMNS: tuple[str, ...] = (
    "id",
    "mission_id",
    "title",
    "objective",
    "context",
    "status",
    "queued_at",
    "success_criteria",
    "deliverables",
    "research_phases",
    "background",
    "focus",
    "references",
    "required_entities",
    "excluded_entities",
    "expected_output_schema",
    "coverage_thresholds",
    "validation_thresholds",
    "deliverable_format",
    "max_loops",
    "min_loops",
    "constraints",
)

# Six ownership fields returned only to the worker that won the atomic claim.
# ``deepsearch_result_key`` is a terminal-write idempotency field, not claim
# output, while all seven additive columns remain startup schema capabilities.
WORKER_LEASE_RETURN_COLUMNS: tuple[str, ...] = (
    "deepsearch_lease_owner",
    "deepsearch_lease_token",
    "deepsearch_leased_at",
    "deepsearch_heartbeat_at",
    "deepsearch_lease_expires_at",
    "deepsearch_attempt_count",
)

WORKER_CLAIM_RETURNING_COLUMNS = (
    *WORKER_CONTENT_COLUMNS,
    *WORKER_LEASE_RETURN_COLUMNS,
)

MISSION_AUTHORING_COLUMNS: tuple[str, ...] = (
    "background",
    "focus",
    "references",
    "required_entities",
    "excluded_entities",
    "expected_output_schema",
    "coverage_thresholds",
    "validation_thresholds",
    "deliverable_format",
    "max_loops",
    "min_loops",
    "constraints",
)

# T42.2 lockstep — TraceLab PR #213 / alembic 028 dropped ``research_depth``
# end-to-end on the TraceLab side. ``CLAIM_MISSION_SQL`` removed the column
# in step (deepsearch/worker/poller.py). ``MissionRow.research_depth`` is
# retained as ``Optional[str] = None`` per S76 PD56 to preserve the boundary
# data-carrier slot for the converter; no in-process dispatch consumes it
# post-S76 (depth.py was deleted at S76 m02 commit ``9c9a207``).
T42_2_DROPPED_COLUMNS: tuple[str, ...] = ("research_depth",)

# Postgres reserved keywords that demand double-quoted column references
# whenever they appear in SQL (per TraceLab T41.3 reserved-word callout).
POSTGRES_RESERVED_WORDS: frozenset[str] = frozenset({"references"})

# Attributes on the ``MissionRow`` dataclass post-S91. Same set as
# ``WORKER_CONTENT_COLUMNS`` plus the ``research_depth`` boundary slot and
# lease-v2 ownership proof. Order is not asserted — set semantics only.
MISSIONROW_FIELDS: frozenset[str] = frozenset({
    "id",
    "mission_id",
    "title",
    "objective",
    "context",
    "status",
    "queued_at",
    "success_criteria",
    "deliverables",
    "research_phases",
    "research_depth",
    "background",
    "focus",
    "references",
    "required_entities",
    "excluded_entities",
    "expected_output_schema",
    "coverage_thresholds",
    "validation_thresholds",
    "deliverable_format",
    "max_loops",
    "min_loops",
    "constraints",
    "lease",
})

MISSIONROW_AUTHORING_TYPES: dict[str, Any] = {
    "background": Optional[str],
    "focus": Optional[str],
    "references": Optional[list[dict[str, Any]]],
    "required_entities": Optional[list[str]],
    "excluded_entities": Optional[list[str]],
    "expected_output_schema": Optional[dict[str, Any]],
    "coverage_thresholds": Optional[dict[str, Any]],
    "validation_thresholds": Optional[dict[str, Any]],
    "deliverable_format": Optional[str],
    "max_loops": Optional[int],
    "min_loops": Optional[int],
    "constraints": Optional[list[str]],
}


# ---------------------------------------------------------------------------
# Helpers — SQL inspection
# ---------------------------------------------------------------------------


def _returning_clause(sql: str) -> str:
    """Return the claim statement's final ``RETURNING`` clause."""
    upper = sql.upper()
    returning_idx = upper.rindex("RETURNING")
    return sql[returning_idx + len("RETURNING"):]


def _split_returning_tokens(sql: str) -> list[str]:
    """Split the RETURNING clause on commas and trim whitespace per token."""
    return [
        token.strip()
        for token in _returning_clause(sql).split(",")
        if token.strip()
    ]


def _normalise_column_token(token: str) -> str:
    """Strip the mission alias/quotes and lower-case a column token."""
    if "." in token:
        token = token.rsplit(".", 1)[1]
    if token.startswith('"') and token.endswith('"'):
        token = token[1:-1]
    return token.lower()


def _returning_columns(sql: str) -> tuple[str, ...]:
    """Lower-cased, unquoted identifiers parsed out of ``RETURNING``."""
    return tuple(
        _normalise_column_token(token)
        for token in _split_returning_tokens(sql)
    )


def _column_token_for(sql: str, column: str) -> str | None:
    """Return the raw RETURNING token matching ``column``."""
    target = column.lower()
    for token in _split_returning_tokens(sql):
        if _normalise_column_token(token) == target:
            return token
    return None


# ---------------------------------------------------------------------------
# CLAIM_MISSION_SQL contract tests
# ---------------------------------------------------------------------------


def test_claim_mission_sql_columns_match_contract() -> None:
    """Atomic claim must return exactly the documented columns, in order.

    Order matters operationally only because the surrounding code reads via
    column names rather than positional indices, but TraceLab's contract doc
    pins the canonical ordering and we mirror it. Order divergence still
    flags as a drift signal worth investigating.
    """
    actual = _returning_columns(MissionPoller.CLAIM_MISSION_SQL)
    assert actual == WORKER_CLAIM_RETURNING_COLUMNS, (
        f"CLAIM_MISSION_SQL RETURNING drifted from TraceLab contract.\n"
        f"  expected: {WORKER_CLAIM_RETURNING_COLUMNS}\n"
        f"  actual:   {actual}\n"
        f"If TraceLab's contract doc changed, update the contract constants in "
        f"this test in the same PR (T42.7 same-PR-update rule)."
    )


@pytest.mark.parametrize("column", MISSION_AUTHORING_COLUMNS)  # type: ignore[misc]
def test_authoring_columns_are_claimed_for_runtime_compilation(column: str) -> None:
    """Preview and execution must consume the same authored mission fields."""
    columns = _returning_columns(MissionPoller.CLAIM_MISSION_SQL)
    assert column in columns, (
        f"Authoring column {column!r} is absent from CLAIM_MISSION_SQL; "
        "TraceLab preview and DeepSearch execution would compile different contracts."
    )


@pytest.mark.parametrize("column", T42_2_DROPPED_COLUMNS)  # type: ignore[misc]
def test_t42_2_dropped_columns_absent_from_select(column: str) -> None:
    """T42.2-dropped columns must stay out of CLAIM_MISSION_SQL.

    ``research_depth`` was dropped end-to-end by TraceLab PR #213 / alembic
    028. Re-adding it to the SELECT would query a column that no longer
    exists on the live database and surface as a runtime SQL error.
    """
    columns = _returning_columns(MissionPoller.CLAIM_MISSION_SQL)
    assert column not in columns, (
        f"T42.2-dropped column {column!r} re-appeared in CLAIM_MISSION_SQL. "
        f"TraceLab dropped this column end-to-end (PR #213 / alembic 028). "
        f"Reading it raises a runtime SQL error."
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "word",
    sorted(POSTGRES_RESERVED_WORDS),
)
def test_reserved_word_columns_must_be_double_quoted(word: str) -> None:
    """Postgres reserved words must use the exact qualified quoted token."""
    sql = MissionPoller.CLAIM_MISSION_SQL
    token = _column_token_for(sql, word)
    assert token == f'mission."{word}"', (
        f"Reserved word {word!r} appears in CLAIM_MISSION_SQL as {token!r} "
        f'instead of the required mission."{word}" token.'
    )


def test_claim_spans_all_projects_and_stays_atomic_and_bounded() -> None:
    """Claim eligibility and ownership must remain one bounded statement.

    Claims are deliberately NOT scoped to one project: missions are authored
    across every TraceLab project and access is scoped by RBAC. The only
    project predicate is the orphan guard - the result is written back onto
    this row, so a row without a project would strand its report.
    """
    sql = MissionPoller.CLAIM_MISSION_SQL
    assert "project_id = $" not in sql
    assert "WHERE project_id IS NOT NULL" in sql
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "UPDATE missions AS mission" in sql
    assert "COALESCE(deepsearch_attempt_count, 0) < $4" in sql
    assert "deepsearch_lease_token IS NULL" in sql
    assert "NOW() - ($5 * INTERVAL '1 second')" in sql
    assert "status = 'completed'" not in sql


def test_lease_lifecycle_statements_are_ownership_bound() -> None:
    """Heartbeat, release, and terminal writes cannot act without proof."""
    for sql in (MissionPoller.HEARTBEAT_SQL, MissionPoller.RELEASE_SQL):
        assert "deepsearch_lease_owner = $2" in sql
        assert "deepsearch_lease_token = $3" in sql
        assert "deepsearch_attempt_count = $4" in sql
        assert "status = 'in_progress'" in sql
    assert "deepsearch_lease_expires_at > NOW()" in MissionPoller.HEARTBEAT_SQL


def test_retry_exhaustion_is_idempotent_across_all_projects() -> None:
    """Only released/expired active or queued rows at cap can be dead-lettered.

    The sweep mirrors the claim predicate: every project, orphans excluded.
    """
    sql = MissionPoller.DEAD_LETTER_EXHAUSTED_SQL
    assert "project_id = $" not in sql
    assert "WHERE project_id IS NOT NULL" in sql
    assert "status = 'in_progress'" in sql
    assert "deepsearch_lease_token IS NULL" in sql
    assert "NOW() - ($2 * INTERVAL '1 second')" in sql
    assert "COALESCE(deepsearch_attempt_count, 0) >= $1" in sql
    assert "status = 'blocked'" in sql


# ---------------------------------------------------------------------------
# MissionRow shape tests
# ---------------------------------------------------------------------------


def test_missionrow_dataclass_exposes_documented_fields() -> None:
    """MissionRow attribute set must match the documented content slots."""
    actual = {f.name for f in fields(MissionRow)}
    assert actual == MISSIONROW_FIELDS, (
        f"MissionRow field set drifted from TraceLab contract.\n"
        f"  missing: {sorted(MISSIONROW_FIELDS - actual)}\n"
        f"  extra:   {sorted(actual - MISSIONROW_FIELDS)}\n"
        f"If a field was intentionally added/removed, coordinate with "
        f"TraceLab and update MISSIONROW_FIELDS in the same PR."
    )


def test_missionrow_research_depth_is_optional_boundary_carrier() -> None:
    """``research_depth`` slot retained as ``Optional[str] = None`` post-S76.

    Per S76 PD56, the field stays as a plain data carrier with a ``None``
    default after T42.2 dropped the underlying column. This guards against
    accidental promotion back into the SELECT or removal from the dataclass.
    """
    by_name = {f.name: f for f in fields(MissionRow)}
    assert "research_depth" in by_name, (
        "research_depth boundary slot missing from MissionRow."
    )
    field = by_name["research_depth"]
    assert field.default is None, (
        f"MissionRow.research_depth must default to None per S76 PD56 "
        f"boundary policy; got default={field.default!r}."
    )


def test_authoring_columns_are_nullable_typed_missionrow_fields() -> None:
    """NULL preserves legacy fallback, so every dedicated slot must default to None."""
    by_name = {field.name: field for field in fields(MissionRow)}
    hints = get_type_hints(MissionRow)

    for name, expected_type in MISSIONROW_AUTHORING_TYPES.items():
        assert by_name[name].default is None
        assert hints[name] == expected_type


# ---------------------------------------------------------------------------
# Round-trip — synthetic asyncpg record → MissionRow → AgentState
# ---------------------------------------------------------------------------


def _synthetic_record() -> dict[str, Any]:
    """Build a record dict matching the asyncpg.Record API used by the parser.

    asyncpg's ``Record`` supports both ``record["key"]`` and ``record.get(key)``;
    a plain dict satisfies both. Values are realistic shapes the parser must
    survive — JSONB columns arriving as either Python objects or raw JSON
    strings (driver-config dependent).
    """
    queued_at = datetime(2026, 4, 30, 15, 0, 0, tzinfo=timezone.utc)
    leased_at = datetime(2026, 4, 30, 15, 1, 0, tzinfo=timezone.utc)
    expires_at = datetime(2026, 4, 30, 15, 3, 0, tzinfo=timezone.utc)
    return {
        "id": "11111111-2222-3333-4444-555555555555",
        "mission_id": "T42-7-ROUNDTRIP",
        "title": "Boundary regression test fixture",
        "objective": "Verify every documented field survives row → state.",
        # Every legacy value conflicts with its dedicated-column sentinel. The
        # dedicated non-null value must win all the way into compilation.
        "context": {
            "background": "legacy background",
            "focus": "legacy focus",
            "references": [{"title": "Legacy Reference"}],
            "required_entities": ["LegacyRequired"],
            "excluded_entities": ["LegacyExcluded"],
            "expected_output_schema": {
                "kind": "legacy",
                "title": "Legacy Schema",
                "format": "list",
            },
            "coverage_thresholds": {"entity": 0.1},
            "validation_thresholds": {"structural": 0.2},
            "deliverable_format": "legacy-format",
            "max_loops": 2,
            "min_loops": 1,
            "constraints": ["legacy constraint"],
        },
        "status": "queued",
        "queued_at": queued_at,
        # JSONB arriving as raw JSON string (asyncpg may surface either shape)
        "success_criteria": json.dumps([
            "Surface every documented field on AgentState",
            "Coerce JSONB string columns to lists",
        ]),
        "deliverables": json.dumps(["report", "citations"]),
        "research_phases": {
            "max_loops": 6,
            "min_loops": 3,
            "deliverable_format": "json",
            "depth": "deep",
        },
        "background": "dedicated background",
        "focus": "dedicated focus",
        # Mix native JSONB values with raw JSON strings to cover both asyncpg
        # codec shapes in the same end-to-end sentinel.
        "references": json.dumps([
            {
                "title": "Trusted Sentinel",
                "url": "https://docs.example.com/sentinel",
            }
        ]),
        "required_entities": ["EntityA", "EntityB"],
        "excluded_entities": json.dumps(["EntityZ"]),
        "expected_output_schema": json.dumps(
            {
                "kind": "classification_matrix",
                "title": "Sentinel Matrix",
                "format": "table",
                "columns": ["Entity", "Finding", "Evidence"],
            }
        ),
        "coverage_thresholds": {"entity": 0.91, "objective": 0.82},
        "validation_thresholds": json.dumps({"structural": 0.87}),
        "deliverable_format": "Evidence Matrix",
        "max_loops": 7,
        "min_loops": 4,
        "constraints": json.dumps(["dedicated constraint"]),
        "deepsearch_lease_owner": "worker-boundary-test",
        "deepsearch_lease_token": "opaque-test-token",
        "deepsearch_leased_at": leased_at,
        "deepsearch_heartbeat_at": leased_at,
        "deepsearch_lease_expires_at": expires_at,
        "deepsearch_attempt_count": 2,
    }


def test_synthetic_record_round_trips_to_compiled_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every authored sentinel survives record → row → state → contract."""
    monkeypatch.setattr(
        mission_contract_module,
        "_disambiguate_declared_entities",
        lambda **_kwargs: ({}, {}, "sentinel_declared"),
    )
    record = _synthetic_record()
    row = MissionPoller._parse_mission_row(record)

    assert row.id == record["id"]
    assert row.mission_id == record["mission_id"]
    assert row.title == record["title"]
    assert row.objective == record["objective"]
    assert row.status == record["status"]
    assert row.queued_at == record["queued_at"]
    assert row.context["background"] == "legacy background"
    assert row.success_criteria == [
        "Surface every documented field on AgentState",
        "Coerce JSONB string columns to lists",
    ]
    assert row.deliverables == ["report", "citations"]
    assert row.research_phases == {
        "max_loops": 6,
        "min_loops": 3,
        "deliverable_format": "json",
        "depth": "deep",
    }
    assert row.background == "dedicated background"
    assert row.focus == "dedicated focus"
    assert row.references == [
        {
            "title": "Trusted Sentinel",
            "url": "https://docs.example.com/sentinel",
        }
    ]
    assert row.required_entities == ["EntityA", "EntityB"]
    assert row.excluded_entities == ["EntityZ"]
    assert row.expected_output_schema == {
        "kind": "classification_matrix",
        "title": "Sentinel Matrix",
        "format": "table",
        "columns": ["Entity", "Finding", "Evidence"],
    }
    assert row.coverage_thresholds == {"entity": 0.91, "objective": 0.82}
    assert row.validation_thresholds == {"structural": 0.87}
    assert row.deliverable_format == "Evidence Matrix"
    assert row.max_loops == 7
    assert row.min_loops == 4
    assert row.constraints == ["dedicated constraint"]
    assert row.lease.owner == "worker-boundary-test"
    assert row.lease.token == "opaque-test-token"
    assert row.lease.attempt_count == 2
    assert row.lease.expires_at == record["deepsearch_lease_expires_at"]
    # Boundary slot — column was dropped (T42.2) so SELECT did not populate it.
    assert row.research_depth is None

    state = convert_row_to_state(row)
    assert state["mission_id"] == record["mission_id"]
    # success_criteria become objectives when present (per converter contract).
    assert state["mission_objectives"] == [
        "Surface every documented field on AgentState",
        "Coerce JSONB string columns to lists",
    ]
    # JSONB-hoisted fields land in mission_context.
    ctx = state["mission_context"]
    assert ctx["title"] == record["title"]
    assert ctx["objective"] == record["objective"]
    assert ctx["mission_id"] == record["mission_id"]
    assert ctx["tracelab_uuid"] == record["id"]
    assert ctx["queued_at"] == record["queued_at"].isoformat()
    assert ctx["background"] == "dedicated background"
    assert ctx["focus"] == "dedicated focus"
    assert ctx["references"] == row.references
    assert ctx["required_entities"] == ["EntityA", "EntityB"]
    assert ctx["excluded_entities"] == ["EntityZ"]
    assert ctx["expected_output_schema"] == row.expected_output_schema
    assert ctx["coverage_thresholds"] == {"entity": 0.91, "objective": 0.82}
    assert ctx["validation_thresholds"] == {"structural": 0.87}
    assert ctx["constraints"] == ["dedicated constraint"]
    # Dedicated budget columns beat both research_phases and legacy context.
    assert state["max_loops"] == 7
    assert state["min_loops"] == 4
    assert state["deliverable_format"] == "evidence matrix"
    # research_phases.depth flows through the boundary normaliser.
    assert state["research_depth"] == "deep"

    contract = compile_contract_from_state(state, origin="s91-boundary-sentinel")
    assert contract.background == "dedicated background"
    assert contract.focus == "dedicated focus"
    assert contract.reference_titles == ["Trusted Sentinel"]
    assert contract.authority_seed_domains == ["docs.example.com"]
    assert contract.required_entities == ["EntityA", "EntityB"]
    assert contract.excluded_entities == ["EntityZ"]
    assert contract.coverage_thresholds == {"entity": 0.91, "objective": 0.82}
    assert contract.validation_thresholds == {"structural": 0.87}
    assert "dedicated constraint" in contract.constraints
    assert len(contract.deliverable_schemas) == 1
    assert contract.deliverable_schemas[0].title == "Sentinel Matrix"
    assert contract.execution_budget.deliverable_format == "evidence matrix"
    assert contract.execution_budget.max_loops == 7
    assert contract.execution_budget.min_loops == 4

    changed_focus_state = {
        **state,
        "mission_context": {
            **state["mission_context"],
            "focus": "a materially different research focus",
        },
    }
    canonical_contract = compile_contract_from_state(
        state,
        origin="s91-boundary-sentinel",
        enrichment_mode=NO_ENRICHMENT_MODE,
    )
    changed_focus_contract = compile_contract_from_state(
        changed_focus_state,
        origin="s91-boundary-sentinel",
        enrichment_mode=NO_ENRICHMENT_MODE,
    )
    assert changed_focus_contract.focus == "a materially different research focus"
    assert changed_focus_contract.contract_id != canonical_contract.contract_id
    focused_prompt_state = {
        **state,
        "mission_contract": changed_focus_contract.model_dump(mode="json"),
    }
    system_prompt, _ = build_codeagent_prompts(focused_prompt_state)
    assert "**Focus:** a materially different research focus" in system_prompt


def test_explicit_empty_authoring_columns_clear_legacy_context() -> None:
    """An authored empty value stays authoritative through compilation."""
    record = _synthetic_record()
    record.update(
        {
            "background": "",
            "focus": "",
            "references": "[]",
            "required_entities": [],
            "excluded_entities": "[]",
            "expected_output_schema": "{}",
            "coverage_thresholds": {},
            "validation_thresholds": "{}",
            "deliverable_format": "",
            "max_loops": 9,
            "min_loops": 5,
            "constraints": [],
        }
    )

    row = MissionPoller._parse_mission_row(record)
    state = convert_row_to_state(row)
    context = state["mission_context"]

    for field_name, empty_value in {
        "background": "",
        "focus": "",
        "references": [],
        "required_entities": [],
        "excluded_entities": [],
        "expected_output_schema": {},
        "coverage_thresholds": {},
        "validation_thresholds": {},
        "deliverable_format": "",
        "constraints": [],
    }.items():
        assert context[field_name] == empty_value
    assert state["deliverable_format"] == "markdown"
    assert state["max_loops"] == 9
    assert state["min_loops"] == 5

    contract = compile_contract_from_state(
        state,
        origin="s91-explicit-empty",
        enrichment_mode="none",
    )
    assert contract.background == ""
    assert contract.reference_titles == []
    assert contract.authority_seed_domains == []
    assert contract.required_entities == []
    assert contract.excluded_entities == []
    assert contract.coverage_thresholds == {}
    assert contract.validation_thresholds == {}
    assert contract.deliverable_schemas == []
    assert "legacy constraint" not in contract.constraints


def test_null_authoring_columns_preserve_legacy_context_fallback() -> None:
    """Only SQL NULL permits pre-column mission context to supply a value."""
    record = _synthetic_record()
    record["research_phases"] = {}
    for field_name in MISSION_AUTHORING_COLUMNS:
        record[field_name] = None

    row = MissionPoller._parse_mission_row(record)
    state = convert_row_to_state(row)
    context = state["mission_context"]

    assert context["background"] == "legacy background"
    assert context["focus"] == "legacy focus"
    assert context["references"] == [{"title": "Legacy Reference"}]
    assert context["required_entities"] == ["LegacyRequired"]
    assert context["excluded_entities"] == ["LegacyExcluded"]
    assert context["constraints"] == ["legacy constraint"]
    assert state["deliverable_format"] == "legacy-format"
    assert state["max_loops"] == 2
    assert state["min_loops"] == 1


@pytest.mark.parametrize(  # type: ignore[misc]
    "raw,expected",
    [
        (None, None),
        ([], []),
        (["EntityA"], ["EntityA"]),
        (["EntityA", "EntityB"], ["EntityA", "EntityB"]),
        # JSONB sometimes surfaces as a string per asyncpg config.
        ('["EntityA", "EntityB"]', ["EntityA", "EntityB"]),
        ('[]', []),
        # Whitespace-only entries are dropped; survivors are stripped.
        (["  EntityA  ", "", "EntityB"], ["EntityA", "EntityB"]),
        # Malformed shapes return None — never raise, never poison downstream.
        ("not-json", None),
        (42, None),
        ({"not": "a list"}, None),
    ],
)
def test_required_entities_jsonb_coercion(
    raw: Any,
    expected: list[str] | None,
) -> None:
    """The JSONB normaliser must survive every shape asyncpg can hand us.

    ``required_entities`` and ``excluded_entities`` are JSONB array-of-strings
    columns. asyncpg may surface them as a Python list or a raw JSON string
    depending on driver config. The normaliser must coerce both shapes,
    drop empty / whitespace entries, and never raise on legacy noise — a
    poisoned list would corrupt the contract compiler's authoritative-entity
    path.
    """
    assert _coerce_entity_list(raw) == expected
