"""Compiler regressions for contract entity/schema classification."""

from __future__ import annotations

import deepsearch.mission.contract as mission_contract_module
from deepsearch.mission import compile_contract_from_state


def test_ds101_predicate_phrases_compile_as_deliverable_columns_not_entities() -> None:
    """DS-101-shaped per-candidate attributes must not become entity slots."""

    contract = compile_contract_from_state(
        {
            "mission_id": "DS-101-SHAPE",
            "mission_context": {
                "title": "MCP Servers for Design Systems",
                "objective": (
                    "Identify every publicly available MCP server that provides "
                    "design system capabilities."
                ),
                "success_criteria": [
                    (
                        "For each shipped server or platform, document tool count "
                        "or exposed capability count where available, read vs write "
                        "capability, validation support, rendering support, governance "
                        "or approval mechanisms, and connected design system."
                    ),
                ],
            },
            "research_depth": "baseline",
            "max_loops": 3,
            "min_loops": 1,
        },
        origin="test",
    )

    forbidden_entities = {
        "tool count or exposed capability count where available",
        "read vs write capability",
        "validation support",
        "rendering support",
        "governance or approval mechanisms",
        "connected design system",
    }
    assert forbidden_entities.isdisjoint(set(contract.named_entities))

    schema = next(
        schema for schema in contract.deliverable_schemas if schema.kind == "per_candidate_matrix"
    )
    assert schema.format == "table"
    assert schema.row_identifier_column == "Candidate"
    assert len([column for column in schema.columns if column != "Candidate"]) >= 5
    assert {
        "Tool count or exposed capability count",
        "Read vs write capability",
        "Validation support",
        "Rendering support",
        "Governance or approval mechanisms",
    }.issubset(set(schema.columns))

    entity_slot_ids = {slot.id for slot in contract.evidence_slots if slot.kind == "named_entity"}
    assert not any("validation-support" in slot_id for slot_id in entity_slot_ids)
    assert not any("rendering-support" in slot_id for slot_id in entity_slot_ids)
    assert not any("governance-or-approval" in slot_id for slot_id in entity_slot_ids)
    assert any(slot.kind == "deliverable_schema" for slot in contract.evidence_slots)


def test_declared_required_entities_survive_attribute_filter_and_fragments_drop(
    monkeypatch,
) -> None:
    """S53.2 declared-anchor + S59.2 short-circuit + S60.1 disambiguation:
    declared DSV plot entities flow through unchanged, the prose LLM/regex
    extractor is bypassed entirely, no prose-derived substrings (``SNE``,
    ``plots``) ever enter the candidate pool, and the focused S60.1
    disambiguation pass emits a single ``extraction_path = llm_declared``
    decision row (``regex_fallback`` when the LLM call fails)."""

    # S60.1 — stub both the structured-extractor build and the disambiguation
    # sync wrapper so the test stays unit-pure. Path becomes ``llm_declared``
    # because the call returned a real outcome (not None).
    from deepsearch.mission import contract as contract_module
    from deepsearch.mission import entity_extractor_llm
    import deepsearch.config as config_module
    from deepsearch.mission.entity_extractor_llm import (
        DeclaredEntityDisambiguationOutcome,
    )

    class _StubSettings:
        @classmethod
        def load(cls):
            return cls()

        def build_structured_extractor(self):
            return object()

    monkeypatch.setattr(
        contract_module, "DeepSearchSettings", _StubSettings, raising=False
    )
    monkeypatch.setattr(
        config_module, "DeepSearchSettings", _StubSettings, raising=False
    )
    monkeypatch.setattr(
        entity_extractor_llm,
        "run_declared_entity_disambiguation_sync",
        lambda *args, **kwargs: DeclaredEntityDisambiguationOutcome(
            disambiguation={},
            query_surfaces={},
            path="llm_declared",
        ),
    )

    required_entities = [
        "histograms",
        "boxplots",
        "scatter plots",
        "scatterplot matrices",
        "correlation heatmaps",
        "t-SNE",
        "UMAP",
        "mosaic plots",
        "missingness matrices",
    ]

    contract = compile_contract_from_state(
        {
            "mission_id": "DSV-S53.2-SHAPE",
            "mission_context": {
                "title": "Visualization for Exploratory Data Analysis",
                "objective": (
                    "Catalog plots for feature correlation including scatterplot "
                    "matrices, correlation heatmaps, t-SNE, UMAP, and mosaic plots."
                ),
                "success_criteria": [
                    (
                        "Document a protocol for initial visual data inspection "
                        "covering histograms, boxplots, scatter plots, and "
                        "summary-statistics tables across at least the first four "
                        "inspection steps."
                    ),
                    (
                        "Catalog plots for feature correlation including scatterplot "
                        "matrices, correlation heatmaps, t-SNE, UMAP, and mosaic plots."
                    ),
                ],
                "required_entities": required_entities,
            },
            "research_depth": "baseline",
            "max_loops": 3,
            "min_loops": 1,
        },
        origin="test",
    )

    assert set(required_entities).issubset(set(contract.named_entities))
    assert "SNE" not in contract.named_entities
    # S59.2 — prose substrings like ``plots`` no longer appear because the
    # extractor is short-circuited when required_entities is declared.
    assert "plots" not in contract.named_entities

    entity_slot_names = {
        slot.target.get("entity") for slot in contract.evidence_slots if slot.kind == "named_entity"
    }
    assert set(required_entities).issubset(entity_slot_names)

    decisions = {
        (decision["entity"], decision["decision"])
        for decision in contract.entity_extraction_decisions
    }
    assert ("histograms", "declared") in decisions
    assert ("boxplots", "declared") in decisions
    assert ("scatterplot matrices", "declared") in decisions
    assert ("", "extraction_path") in decisions
    extraction_paths = [
        decision["reason"]
        for decision in contract.entity_extraction_decisions
        if decision.get("decision") == "extraction_path"
    ]
    assert extraction_paths == ["llm_declared"]


def test_entity_extraction_classifier_filters_only_extracted_candidates() -> None:
    """Declared attribute-like entities are pinned; extracted ones are rejected."""

    named_entities, decisions = mission_contract_module._classify_named_entity_candidates(
        ["histograms", "EDA"],
        ["histograms", "validation support", "SNE", "t-SNE", "EDA Visualization"],
    )

    assert "histograms" in named_entities
    assert "EDA" in named_entities
    assert "validation support" not in named_entities
    assert "SNE" not in named_entities
    assert "t-SNE" in named_entities
    assert "EDA Visualization" in named_entities
    assert {"entity": "EDA", "decision": "declared"} in decisions
    assert {"entity": "histograms", "decision": "extracted"} in decisions
    assert {
        "entity": "validation support",
        "decision": "rejected_attribute_phrase",
    } in decisions
    assert {
        "entity": "SNE",
        "decision": "rejected_fragment",
        "reason": "fragment_of:t-SNE",
    } in decisions


def test_interrogative_attribute_phrases_do_not_enter_entity_registry() -> None:
    """Leading what/whether predicates are matrix fields, not searchable entities."""

    contract = compile_contract_from_state(
        {
            "mission_id": "DS-101-INTERROGATIVE-SHAPE",
            "mission_context": {
                "title": "MCP Servers for Design Systems",
                "objective": (
                    "For each shipped MCP server, document what tools it exposes, "
                    "whether it renders output, and whether it governs writes."
                ),
            },
            "research_depth": "baseline",
            "max_loops": 3,
            "min_loops": 1,
        },
        origin="test",
    )

    assert "what tools it exposes" not in contract.named_entities
    assert "whether it renders output" not in contract.named_entities
    assert "whether it governs writes" not in contract.named_entities

    schema = next(
        schema for schema in contract.deliverable_schemas if schema.kind == "per_candidate_matrix"
    )
    assert {
        "What tools it exposes",
        "Whether it renders output",
        "Whether it governs writes",
    }.issubset(set(schema.columns))


def test_single_attribute_tail_does_not_suppress_real_entity_list() -> None:
    """A stray attribute phrase must not cause earlier entity items to be dropped."""

    contract = compile_contract_from_state(
        {
            "mission_id": "MIXED-ENTITY-ATTRIBUTE-SHAPE",
            "mission_context": {
                "title": "JavaScript Bundler Landscape",
                "objective": (
                    "Compare Vite, Webpack, Rollup, and Turbopack, and document "
                    "validation support."
                ),
            },
            "research_depth": "baseline",
            "max_loops": 3,
            "min_loops": 1,
        },
        origin="test",
    )

    assert {"Vite", "Webpack", "Rollup", "Turbopack"}.issubset(set(contract.named_entities))
    assert "validation support" not in contract.named_entities


def test_leading_conjunction_is_stripped_before_entity_normalization() -> None:
    """DS-101's comma-or tail should produce freemium, never or freemium."""

    assert mission_contract_module._normalize_enumerated_entity("or freemium") == ["freemium"]
    assert mission_contract_module._normalize_enumerated_entity("and commercial") == ["commercial"]
    assert mission_contract_module._normalize_enumerated_entity("vs adjacent") == ["adjacent"]
    assert mission_contract_module._normalize_enumerated_entity("nor proprietary") == [
        "proprietary"
    ]


def test_leading_conjunction_strip_is_idempotent_for_clean_span() -> None:
    """Running the pre-normalizer twice on a clean span is a no-op."""

    span = "freemium"

    once = mission_contract_module._strip_leading_entity_conjunction(span)
    twice = mission_contract_module._strip_leading_entity_conjunction(once)

    assert once == "freemium"
    assert twice == "freemium"


def test_comma_or_entity_list_produces_clean_spans() -> None:
    """Comma-or list parsing should not preserve conjunction artifacts."""

    contract = compile_contract_from_state(
        {
            "mission_id": "COMMA-OR-SPAN-SHAPE",
            "mission_context": {
                "title": "Classification Labels",
                "objective": "Compare open source, commercial, freemium, or adjacent.",
            },
            "research_depth": "baseline",
            "max_loops": 3,
            "min_loops": 1,
        },
        origin="test",
    )

    assert {"open source", "commercial", "freemium", "adjacent"}.issubset(
        set(contract.named_entities)
    )
    assert not any(
        entity.startswith(("or ", "and ", "vs ", "nor ")) for entity in contract.named_entities
    )


def _objective_contract(objective_id: str, text: str) -> object:
    """Build an ObjectiveContract instance for direct compiler tests."""

    from deepsearch.mission.contract import ObjectiveContract

    return ObjectiveContract(id=objective_id, text=text, source="test", required=True)


def test_entity_objective_binding_attaches_objective_ids_to_entity_slot_target() -> None:
    """S52.3 / D-S51-C compiler: entity slots receive ``target.objective_ids``
    listing every required objective whose text names the entity (case-insensitive
    whole-token match)."""

    objectives = [
        _objective_contract(
            "objective-1",
            "Identify 3+ distinct RAG architectural patterns used in multi-agent systems.",
        ),
        _objective_contract(
            "objective-2",
            "Cover frameworks: LangGraph, AutoGen, CrewAI, LlamaIndex, LangChain.",
        ),
        _objective_contract(
            "objective-3",
            "Surface recent (2024-2025) benchmark results.",
        ),
    ]
    named_entities = ["LangGraph", "AutoGen", "CrewAI", "LlamaIndex", "LangChain"]

    slots = mission_contract_module._build_evidence_slots(named_entities, objectives, [])

    entity_slots = {slot.id: slot for slot in slots if slot.kind == "named_entity"}
    for entity in named_entities:
        slot_id = f"slot-entity-{mission_contract_module._slug(entity)}"
        assert slot_id in entity_slots, f"missing entity slot for {entity}"
        bindings = entity_slots[slot_id].target.get("objective_ids", [])
        assert bindings == [
            "objective-2"
        ], f"{entity}: expected only objective-2 binding, got {bindings}"


def test_entity_objective_binding_distinguishes_lang_prefix_entities() -> None:
    """Whole-token matching: ``LangChain`` mentions must not bind ``LangGraph``
    (and vice versa). Substring matching would falsely bind both because they
    share the ``Lang`` prefix."""

    objectives = [
        _objective_contract("objective-1", "Use LangChain for tooling integration."),
    ]
    named_entities = ["LangChain", "LangGraph"]

    slots = mission_contract_module._build_evidence_slots(named_entities, objectives, [])
    entity_slots = {slot.id: slot for slot in slots if slot.kind == "named_entity"}

    langchain_bindings = entity_slots[
        f"slot-entity-{mission_contract_module._slug('LangChain')}"
    ].target.get("objective_ids", [])
    langgraph_bindings = entity_slots[
        f"slot-entity-{mission_contract_module._slug('LangGraph')}"
    ].target.get("objective_ids", [])

    assert langchain_bindings == ["objective-1"]
    assert langgraph_bindings == []


def test_entity_objective_binding_omits_field_when_no_objectives_reference_entity() -> None:
    """Entity slots without any objective text mentions must not carry an empty
    ``objective_ids`` field — pre-S52 contract shape preserved exactly."""

    objectives = [
        _objective_contract("objective-1", "Identify architectural patterns."),
    ]
    named_entities = ["LangGraph"]

    slots = mission_contract_module._build_evidence_slots(named_entities, objectives, [])
    entity_slot = next(slot for slot in slots if slot.kind == "named_entity")

    assert "objective_ids" not in entity_slot.target


def test_entity_objective_binding_skips_non_required_objectives() -> None:
    """Optional/disabled objectives must not contribute entity bindings."""

    from deepsearch.mission.contract import ObjectiveContract

    objectives = [
        ObjectiveContract(
            id="objective-1",
            text="Cover LangGraph specifically.",
            source="test",
            required=False,
        ),
    ]
    named_entities = ["LangGraph"]

    slots = mission_contract_module._build_evidence_slots(named_entities, objectives, [])
    entity_slot = next(slot for slot in slots if slot.kind == "named_entity")

    assert "objective_ids" not in entity_slot.target


def test_entity_objective_binding_case_insensitive_whole_token_match() -> None:
    """Mixed case in objective text still binds; partial words still don't bind."""

    objectives = [
        _objective_contract("objective-1", "use langgraph and langgraph2."),
    ]
    named_entities = ["LangGraph"]

    slots = mission_contract_module._build_evidence_slots(named_entities, objectives, [])
    entity_slot = next(slot for slot in slots if slot.kind == "named_entity")

    assert entity_slot.target.get("objective_ids") == ["objective-1"]


def test_entity_objective_binding_handles_hyphenated_entity_names() -> None:
    """Hyphenated entity names must match cleanly when present in objective text."""

    objectives = [
        _objective_contract(
            "objective-1",
            "Use Multi-Agent RAG patterns where MA-RAG is the canonical reference.",
        ),
    ]
    named_entities = ["Multi-Agent RAG", "MA-RAG"]

    slots = mission_contract_module._build_evidence_slots(named_entities, objectives, [])
    entity_slots = {slot.id: slot for slot in slots if slot.kind == "named_entity"}
    multi_agent_bindings = entity_slots[
        f"slot-entity-{mission_contract_module._slug('Multi-Agent RAG')}"
    ].target.get("objective_ids", [])
    ma_rag_bindings = entity_slots[
        f"slot-entity-{mission_contract_module._slug('MA-RAG')}"
    ].target.get("objective_ids", [])

    assert multi_agent_bindings == ["objective-1"]
    assert ma_rag_bindings == ["objective-1"]
