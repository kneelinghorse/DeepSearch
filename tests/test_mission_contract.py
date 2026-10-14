"""Tests for compiled mission contract generation and persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - test environment guard
    sys.path.insert(0, str(ROOT))

import deepsearch.mission.contract as mission_contract_module  # noqa: E402
from deepsearch.mission import (  # noqa: E402
    build_retrieval_budget,
    compile_contract_from_state,
    load_mission_from_yaml,
    prepare_execution_contract,
)
from deepsearch.mission.contract import DeliverableSchemaContract  # noqa: E402

FIXTURE_DIR = Path("tests/fixtures/missions")


def _fixture_path(name: str) -> Path:
    return FIXTURE_DIR / name


def test_prepare_execution_contract_from_yaml_persists_artifact(tmp_path: Path) -> None:
    """File-based mission loading should compile and persist an execution contract."""

    state = load_mission_from_yaml(_fixture_path("contract_mission.yaml"))

    prepared = prepare_execution_contract(state, origin="yaml", output_dir=tmp_path)
    payload = json.loads(prepared.path.read_text(encoding="utf-8"))

    assert prepared.path.exists()
    assert payload["mission_id"] == "CONTRACT.1"
    assert payload["contract_id"] == state["mission_contract_id"]
    assert payload["execution_budget"]["max_loops"] == 4
    assert payload["execution_budget"]["research_depth"] == "deep"
    assert "SAPLMA" in payload["named_entities"]
    assert "Kadavath et al. 2022" in payload["named_entities"]
    assert any(
        schema["kind"] == "annotated_bibliography" for schema in payload["deliverable_schemas"]
    )
    assert any(slot["kind"] == "deliverable_schema" for slot in payload["evidence_slots"])
    assert state["mission_contract_path"] == str(prepared.path)
    assert state["telemetry"]["contract_id"] == state["mission_contract_id"]


def test_compile_contract_makes_umbrella_objective_optional_when_success_criteria_exist() -> None:
    state = {
        "mission_id": "FORCE-SMOKE-02",
        "mission_context": {
            "title": "LangGraph Checkpointing Patterns for Long-Running Agents",
            "objective": "Produce a short, readable research brief about checkpointing backends.",
            "success_criteria": [
                "Briefly describe each backend",
                "Explain when to use each backend",
                "Note one concrete operational gotcha for each backend",
            ],
        },
        "research_depth": "baseline",
        "max_loops": 3,
        "min_loops": 2,
    }

    contract = compile_contract_from_state(state, origin="yaml")

    assert contract.objectives[0].id == "objective-0"
    assert contract.objectives[0].required is False
    assert [objective.required for objective in contract.objectives[1:]] == [True, True, True]
    assert "slot-objective-0" not in {slot.id for slot in contract.evidence_slots}
    assert "check-objective-0" not in {check.id for check in contract.acceptance_checks}


def test_compile_contract_excludes_generic_acronyms_from_named_entities() -> None:
    state = {
        "mission_id": "FORCE-SMOKE-03",
        "mission_context": {
            "title": "LangGraph Checkpointing Patterns for Long-Running Agents",
            "objective": (
                "Produce a short, readable research brief with practical guidance "
                "rather than exhaustive API detail. Cover MemorySaver, SqliteSaver, "
                "AsyncSqliteSaver, and PostgresSaver."
            ),
        },
        "research_depth": "baseline",
        "max_loops": 3,
        "min_loops": 2,
    }

    contract = compile_contract_from_state(state, origin="yaml")

    assert "API" not in contract.named_entities
    assert {"MemorySaver", "SqliteSaver", "AsyncSqliteSaver", "PostgresSaver"}.issubset(
        set(contract.named_entities)
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    ("fixture_name", "contract_id", "expected_entities", "expected_kinds"),
    [
        (
            "acceptance_acc_canary_riim_01.yaml",
            "ACC-CANARY-RIIM-01",
            {"SAPLMA", "Kadavath et al. 2022", "Farquhar et al. 2024"},
            {"annotated_bibliography", "table"},
        ),
        (
            "acceptance_acc_table_comp_01.yaml",
            "ACC-TABLE-COMP-01",
            {"RAG", "TAVILY", "EXA", "SERPAPI"},
            {"classification_matrix"},
        ),
        (
            "acceptance_acc_bib_evid_01.yaml",
            "ACC-BIB-EVID-01",
            {"SAPLMA", "CCS"},
            {"annotated_bibliography", "table"},
        ),
        (
            "acceptance_acc_corp_trace_01.yaml",
            "ACC-CORP-TRACE-01",
            {"LANGGRAPH", "RAG", "CMOS"},
            {"annotated_bibliography", "corpus_trace"},
        ),
    ],
)
def test_acceptance_contract_fixtures_compile_expected_shapes(
    tmp_path: Path,
    fixture_name: str,
    contract_id: str,
    expected_entities: set[str],
    expected_kinds: set[str],
) -> None:
    """Acceptance-contract fixtures should compile into the expected contract shapes."""

    state = load_mission_from_yaml(_fixture_path(fixture_name))
    prepared = prepare_execution_contract(
        state,
        origin="yaml",
        output_dir=tmp_path / contract_id.lower(),
    )
    payload = json.loads(prepared.path.read_text(encoding="utf-8"))

    assert payload["mission_id"] == contract_id
    assert expected_entities.issubset(set(payload["named_entities"]))
    assert expected_kinds.issubset({item["kind"] for item in payload["deliverable_schemas"]})
    assert any(slot["kind"] == "named_entity" for slot in payload["evidence_slots"])
    assert any(slot["kind"] == "deliverable_schema" for slot in payload["evidence_slots"])


def test_deliverable_schema_contract_model_validate_json_round_trips_new_fields() -> None:
    """Expanded schema fields should survive JSON serialization round-trips."""

    schema = DeliverableSchemaContract(
        id="deliverable-schema-1",
        kind="corpus_trace",
        title="Corpus Trace Table",
        format="table",
        source_text="Corpus trace table with at least 6 rows",
        min_items=6,
        minimum_citation_count=6,
        columns=["Claim", "Corpus Evidence", "Confidence", "Gap"],
        entity_extraction_required=True,
        entity_types=["author_year", "acronym"],
        row_identifier_column="Claim",
        recommendation_required=False,
        evidence_gaps_section_required=False,
        corpus_reference_required=True,
        claim_to_evidence_mapping_required=True,
        limitations_section_required=True,
        notes=["Trace every claim to cited corpus evidence."],
    )

    round_tripped = DeliverableSchemaContract.model_validate_json(schema.model_dump_json())

    assert round_tripped.minimum_citation_count == 6
    assert round_tripped.entity_extraction_required is True
    assert round_tripped.entity_types == ["author_year", "acronym"]
    assert round_tripped.row_identifier_column == "Claim"
    assert round_tripped.corpus_reference_required is True
    assert round_tripped.claim_to_evidence_mapping_required is True
    assert round_tripped.limitations_section_required is True


@pytest.mark.parametrize(  # type: ignore[misc]
    ("fixture_name", "matcher"),
    [
        (
            "acceptance_acc_table_comp_01.yaml",
            lambda schemas: _schema_by_kind(schemas, "classification_matrix"),
        ),
        (
            "acceptance_acc_bib_evid_01.yaml",
            lambda schemas: _schema_by_kind(schemas, "annotated_bibliography"),
        ),
        (
            "acceptance_acc_canary_riim_01.yaml",
            lambda schemas: schemas[0],
        ),
        (
            "acceptance_acc_corp_trace_01.yaml",
            lambda schemas: _schema_by_kind(schemas, "corpus_trace"),
        ),
    ],
)
def test_acceptance_contract_fixtures_populate_expanded_schema_fields(
    tmp_path: Path,
    fixture_name: str,
    matcher: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> None:
    """Acceptance fixtures should compile the Sprint 31 schema extensions."""

    state = load_mission_from_yaml(_fixture_path(fixture_name))
    prepared = prepare_execution_contract(
        state,
        origin="yaml",
        output_dir=tmp_path / fixture_name.removesuffix(".yaml"),
    )
    payload = json.loads(prepared.path.read_text(encoding="utf-8"))
    schemas = payload["deliverable_schemas"]
    selected = matcher(schemas)

    if fixture_name == "acceptance_acc_table_comp_01.yaml":
        assert selected["row_identifier_column"] == "Backend"
        assert selected["recommendation_required"] is True
    elif fixture_name == "acceptance_acc_bib_evid_01.yaml":
        assert selected["minimum_citation_count"] == 12
        assert selected["evidence_gaps_section_required"] is True
    elif fixture_name == "acceptance_acc_canary_riim_01.yaml":
        assert selected["entity_extraction_required"] is True
        assert "author_year" in selected["entity_types"]
        assert "acronym" in selected["entity_types"]
    elif fixture_name == "acceptance_acc_corp_trace_01.yaml":
        assert selected["row_identifier_column"] == "Claim"
        assert selected["corpus_reference_required"] is True
        assert selected["claim_to_evidence_mapping_required"] is True
        assert selected["limitations_section_required"] is True

    structural_targets = [
        check["target"]
        for check in payload["acceptance_checks"]
        if check["kind"] == "structural_output"
    ]
    assert any(target.get("schema_id") == selected["id"] for target in structural_targets)


def _schema_by_kind(
    schemas: list[dict[str, object]],
    kind: str,
) -> dict[str, object]:
    for schema in schemas:
        if schema.get("kind") == kind:
            return schema
    raise AssertionError(f"Schema with kind {kind!r} not found")


def test_build_retrieval_budget_preserves_token_ceiling_without_count_caps() -> None:
    """Compiled execution budgets should keep token ceilings without count caps."""

    contract = compile_contract_from_state(
        {
            "mission_id": "B28.2-BUDGET",
            "mission_context": {
                "title": "Budget Test",
                "objective": "Verify retrieval budget derivation",
            },
            "mission_objectives": ["Verify retrieval budget derivation"],
            "deliverable_format": "markdown",
            "research_depth": "deep",
            "max_loops": 4,
            "min_loops": 2,
            "depth_config": {
                "max_results_per_query": 5,
                "max_tokens_per_loop": 2400,
            },
        },
        origin="test",
    )

    budget = build_retrieval_budget(contract, query_count=2, lane_count=3)

    assert budget.per_loop is None
    assert budget.per_query is None
    assert budget.per_lane is None
    assert budget.max_tokens_per_loop == 2400


def test_compile_contract_extracts_camel_case_entities_for_force_smoke_shape() -> None:
    """CamelCase backend names should become contract entities for coverage/query planning."""

    contract = compile_contract_from_state(
        {
            "mission_id": "FORCE-SMOKE-01",
            "mission_context": {
                "title": "LangGraph Checkpointing Patterns",
                "objective": (
                    "Document the primary checkpointing patterns LangGraph provides "
                    "for long-running agents."
                ),
                "background": (
                    "Compare MemorySaver, SqliteSaver, AsyncSqliteSaver, and "
                    "PostgresSaver for long-running agents."
                ),
                "success_criteria": [
                    "Describe when to use MemorySaver in practice",
                    "Describe when to use PostgresSaver in practice",
                ],
            },
            "mission_objectives": [
                "Compare MemorySaver, SqliteSaver, AsyncSqliteSaver, and PostgresSaver"
            ],
            "deliverable_format": "markdown",
            "research_depth": "baseline",
            "max_loops": 2,
            "min_loops": 1,
            "depth_config": {},
        },
        origin="test",
    )

    assert {
        "LangGraph",
        "MemorySaver",
        "SqliteSaver",
        "AsyncSqliteSaver",
        "PostgresSaver",
    }.issubset(set(contract.named_entities))


def test_compile_contract_extracts_mixed_acronym_camel_case_entities() -> None:
    """Entities like BM25Retriever should survive contract extraction intact."""

    contract = compile_contract_from_state(
        {
            "mission_id": "FORCE-SMOKE-13",
            "mission_context": {
                "title": "LangChain Retriever Selection for RAG Pipelines",
                "objective": (
                    "Compare BM25Retriever, MultiQueryRetriever, "
                    "ParentDocumentRetriever, and ContextualCompressionRetriever."
                ),
                "success_criteria": [
                    (
                        "Briefly describe BM25Retriever, MultiQueryRetriever, "
                        "ParentDocumentRetriever, and ContextualCompressionRetriever"
                    ),
                ],
            },
            "mission_objectives": [
                (
                    "Briefly describe BM25Retriever, MultiQueryRetriever, "
                    "ParentDocumentRetriever, and ContextualCompressionRetriever"
                ),
            ],
            "deliverable_format": "markdown",
            "research_depth": "baseline",
            "max_loops": 2,
            "min_loops": 1,
            "depth_config": {},
        },
        origin="test",
    )

    assert {
        "BM25Retriever",
        "MultiQueryRetriever",
        "ParentDocumentRetriever",
        "ContextualCompressionRetriever",
    }.issubset(set(contract.named_entities))


def test_compile_contract_excludes_guardrails_and_formatting_criteria_from_objectives() -> None:
    """Scope guardrails and report-format criteria should not become research objectives."""

    contract = compile_contract_from_state(
        {
            "mission_id": "FORCE-SMOKE-11",
            "mission_context": {
                "title": "LangGraph Checkpoint Backend Comparison",
                "objective": (
                    "Compare MemorySaver, SqliteSaver, AsyncSqliteSaver, and "
                    "PostgresSaver in LangGraph."
                ),
                "success_criteria": [
                    "Explain when to use each backend in practice",
                    "Capture one concrete operational gotcha for each backend",
                    (
                        "Keep source selection and findings on-contract: do not include "
                        "RedisSaver or other off-contract sibling backends unless "
                        "explicitly required"
                    ),
                    (
                        "Include a compact comparison table with columns: Backend, "
                        "Use When, Avoid When, Operational Gotcha"
                    ),
                    (
                        "Executive summary should synthesize the backend tradeoffs "
                        "rather than restating an inventory"
                    ),
                ],
            },
            "mission_objectives": [
                "Explain when to use each backend in practice",
                "Capture one concrete operational gotcha for each backend",
            ],
            "deliverable_format": "markdown",
            "research_depth": "baseline",
            "max_loops": 3,
            "min_loops": 1,
            "depth_config": {},
        },
        origin="test",
    )

    assert [objective.text for objective in contract.objectives if objective.required] == [
        "Explain when to use each backend in practice",
        "Capture one concrete operational gotcha for each backend",
    ]
    assert "RedisSaver" not in contract.named_entities
    assert any("RedisSaver" in constraint for constraint in contract.constraints)
    assert any(schema.format == "table" for schema in contract.deliverable_schemas)


def test_compile_contract_negation_criteria_become_acceptance_gates_not_objectives() -> None:
    """Constraint-style success criteria should not create coverage objectives."""

    contract = compile_contract_from_state(
        {
            "mission_id": "B35.8-NEG",
            "mission_context": {
                "title": "JavaScript bundler landscape",
                "objective": (
                    "Compare Vite, esbuild, Webpack, Rollup, and Turbopack "
                    "architectures."
                ),
                "success_criteria": [
                    "Compare Vite, esbuild, Webpack, Rollup, and Turbopack architectures",
                    "No Medium, dev.to, LinkedIn, Towards Data Science used as primary citations",
                ],
            },
            "mission_objectives": [
                "Compare Vite, esbuild, Webpack, Rollup, and Turbopack architectures",
                "No Medium, dev.to, LinkedIn, Towards Data Science used as primary citations",
            ],
            "deliverable_format": "markdown",
            "research_depth": "baseline",
            "max_loops": 3,
            "min_loops": 1,
            "depth_config": {},
        },
        origin="test",
    )

    required_objectives = [
        objective.text for objective in contract.objectives if objective.required
    ]
    assert required_objectives == [
        "Compare Vite, esbuild, Webpack, Rollup, and Turbopack architectures"
    ]
    assert "Medium" not in contract.named_entities
    assert "LinkedIn" not in contract.named_entities
    assert any(
        check.kind == "acceptance_gate"
        and check.target["criterion"]
        == "No Medium, dev.to, LinkedIn, Towards Data Science used as primary citations"
        for check in contract.acceptance_checks
    )


def test_compile_contract_negation_filter_does_not_suppress_positive_subject_entities() -> None:
    """Positive objective entities should survive even when constraints exist."""

    contract = compile_contract_from_state(
        {
            "mission_id": "B35.8-MIXED",
            "mission_context": {
                "title": "JavaScript bundler landscape",
                "objective": (
                    "Compare Vite, esbuild, Webpack, Rollup, and Turbopack "
                    "architectures."
                ),
                "success_criteria": [
                    "Compare Vite, esbuild, Webpack, Rollup, and Turbopack architectures",
                    "No Medium, dev.to, LinkedIn, Towards Data Science used as primary citations",
                    "Include a comparison matrix with columns: Bundler, Architecture, Evidence",
                ],
            },
            "mission_objectives": [
                "Compare Vite, esbuild, Webpack, Rollup, and Turbopack architectures",
                "No Medium, dev.to, LinkedIn, Towards Data Science used as primary citations",
                "Include a comparison matrix with columns: Bundler, Architecture, Evidence",
            ],
            "deliverable_format": "markdown",
            "research_depth": "baseline",
            "max_loops": 3,
            "min_loops": 1,
            "depth_config": {},
        },
        origin="test",
    )

    assert {"Vite", "Webpack", "Rollup", "Turbopack"}.issubset(set(contract.named_entities))
    assert "Medium" not in contract.named_entities
    assert "LinkedIn" not in contract.named_entities
    assert any(check.kind == "acceptance_gate" for check in contract.acceptance_checks)
    assert any(check.kind == "structural_output" for check in contract.acceptance_checks)


def test_compile_contract_ignores_prefixed_mission_id_tokens_in_tracelab_title() -> None:
    """TraceLab titles prefixed with mission IDs should not become named entities."""

    contract = compile_contract_from_state(
        {
            "mission_id": "FORCE-SMOKE-08",
            "mission_context": {
                "title": "FORCE-SMOKE-08 - LangGraph Checkpointing Patterns",
                "objective": (
                    "Compare MemorySaver, SqliteSaver, AsyncSqliteSaver, and "
                    "PostgresSaver for a single-tenant Python agent."
                ),
                "success_criteria": [
                    "Explain when to use each backend in practice",
                    "Capture one concrete operational gotcha for each backend",
                ],
            },
            "mission_objectives": [
                "Explain when to use each backend in practice",
                "Capture one concrete operational gotcha for each backend",
            ],
            "deliverable_format": "markdown",
            "research_depth": "baseline",
            "max_loops": 3,
            "min_loops": 2,
            "depth_config": {},
        },
        origin="test",
    )

    assert "FORCE" not in contract.named_entities
    assert "SMOKE" not in contract.named_entities
    assert {
        "LangGraph",
        "MemorySaver",
        "SqliteSaver",
        "AsyncSqliteSaver",
        "PostgresSaver",
    }.issubset(set(contract.named_entities))


def test_compile_contract_preserves_threshold_overrides() -> None:
    """Contract payload should retain explicit threshold overrides for routing."""

    contract = compile_contract_from_state(
        {
            "mission_id": "THRESHOLD-TEST",
            "mission_context": {
                "title": "Threshold override test",
                "objective": "Verify contract threshold overrides",
                "coverage_thresholds": {"objective": 0.65, "entity": 0.8},
                "validation_thresholds": {"slot": 0.5},
            },
            "mission_objectives": ["Verify contract threshold overrides"],
            "deliverable_format": "markdown",
            "research_depth": "baseline",
            "max_loops": 2,
            "min_loops": 1,
            "depth_config": {},
        },
        origin="test",
    )

    assert contract.coverage_thresholds == {"objective": 0.65, "entity": 0.8}
    assert contract.validation_thresholds == {"slot": 0.5}


def test_compile_contract_reads_and_normalizes_excluded_domains() -> None:
    """excluded_domains should be surfaced as a first-class compiled contract field."""

    contract = compile_contract_from_state(
        {
            "mission_id": "B35.10-DOMAINS",
            "mission_context": {
                "title": "Bundler source constraints",
                "objective": "Compare Vite and Rollup using primary documentation.",
                "excluded_domains": [
                    "Medium.com",
                    "https://www.LinkedIn.com/feed/update/example",
                    "dev.to",
                ],
            },
            "mission_objectives": ["Compare Vite and Rollup using primary documentation."],
            "deliverable_format": "markdown",
            "research_depth": "baseline",
            "max_loops": 2,
            "min_loops": 1,
            "depth_config": {},
        },
        origin="test",
    )

    assert contract.excluded_domains == ["medium.com", "linkedin.com", "dev.to"]


def test_compile_contract_uses_authored_expected_output_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """expected_output_schema should bypass regex-based schema inference."""

    def _guarded_extract(
        mission_deliverables: object,
        success_criteria: object,
    ) -> list[object]:
        if mission_deliverables == []:
            return []
        raise AssertionError("extract_deliverable_schemas should not be called")

    monkeypatch.setattr(mission_contract_module, "extract_deliverable_schemas", _guarded_extract)

    contract = compile_contract_from_state(
        {
            "mission_id": "B35.6-SCHEMA",
            "mission_context": {
                "title": "TraceLab authored schema test",
                "objective": "Validate authored output schemas.",
                "deliverables": ["Annotated bibliography of 10+ sources"],
                "success_criteria": ["Include a comparison matrix in the final report"],
                "expected_output_schema": {
                    "kind": "classification_matrix",
                    "title": "Authored Classification Matrix",
                    "format": "table",
                    "min_items": 5,
                    "columns": ["Technique", "Classification", "Evidence"],
                    "allowed_values": ["foundational", "applied"],
                    "required_sections": ["Method Notes"],
                    "entity_extraction_required": True,
                    "entity_types": ["author_year", "acronym"],
                    "recommendation_required": True,
                    "evidence_gaps_section_required": True,
                    "corpus_reference_required": True,
                    "claim_to_evidence_mapping_required": True,
                    "limitations_section_required": True,
                    "minimum_citation_count": 6,
                    "row_identifier_column": "Technique",
                    "notes": ["One row per technique."],
                },
            },
            "mission_objectives": ["Validate authored output schemas."],
            "deliverable_format": "markdown",
            "research_depth": "baseline",
            "max_loops": 2,
            "min_loops": 1,
            "depth_config": {},
        },
        origin="test",
    )

    assert len(contract.deliverable_schemas) == 1
    schema = contract.deliverable_schemas[0]
    assert schema.kind == "classification_matrix"
    assert schema.title == "Authored Classification Matrix"
    assert schema.columns == ["Technique", "Classification", "Evidence"]
    assert schema.allowed_values == ["foundational", "applied"]
    assert schema.required_sections == ["Method Notes"]
    assert schema.entity_extraction_required is True
    assert schema.entity_types == ["author_year", "acronym"]
    assert schema.recommendation_required is True
    assert schema.evidence_gaps_section_required is True
    assert schema.corpus_reference_required is True
    assert schema.claim_to_evidence_mapping_required is True
    assert schema.limitations_section_required is True
    assert schema.minimum_citation_count == 6
    assert schema.row_identifier_column == "Technique"
    assert schema.notes == ["One row per technique."]
    assert (
        schema.source_text
        == "authored expected_output_schema[1]: Authored Classification Matrix"
    )


def test_expected_output_schema_json_schema_artifact_matches_authored_fields() -> None:
    """Published JSON Schema should expose the agreed authored field surface."""

    schema_path = Path("deepsearch/mission/schemas/expected_output_schema.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    properties = schema["$defs"]["outputSchemaItem"]["properties"]

    assert schema["title"] == "DeepSearch expected_output_schema"
    assert set(properties) == {
        "kind",
        "title",
        "format",
        "min_items",
        "columns",
        "allowed_values",
        "required_sections",
        "entity_extraction_required",
        "entity_types",
        "recommendation_required",
        "evidence_gaps_section_required",
        "corpus_reference_required",
        "claim_to_evidence_mapping_required",
        "limitations_section_required",
        "minimum_citation_count",
        "row_identifier_column",
        "notes",
    }
