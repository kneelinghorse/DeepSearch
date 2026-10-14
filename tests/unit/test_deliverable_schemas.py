"""Focused unit tests for Sprint 31 deliverable schema extraction."""

from __future__ import annotations

from deepsearch.agent.deliverable_schemas import extract_deliverable_schemas


def test_classification_matrix_extracts_row_identifier_and_recommendation_requirement() -> None:
    """Matrix contracts should expose row identity and recommendation requirements."""

    schemas = extract_deliverable_schemas(
        [
            "Comparison matrix with at least 5 rows and columns: "
            "Backend, Strengths, Weaknesses, Evidence, Recommendation"
        ],
        ["Produce a decision-useful recommendation grounded in RAG evidence"],
    )

    schema = next(item for item in schemas if item.kind == "classification_matrix")

    assert schema.row_identifier_column == "Backend"
    assert schema.recommendation_required is True


def test_annotated_bibliography_extracts_citation_floor_and_gap_requirement() -> None:
    """Bibliography contracts should carry citation and evidence-gap requirements."""

    schemas = extract_deliverable_schemas(
        ["Annotated bibliography of at least 12 papers"],
        ["Call out evidence gaps that still block downstream evaluation work"],
    )

    schema = next(item for item in schemas if item.kind == "annotated_bibliography")

    assert schema.minimum_citation_count == 12
    assert schema.evidence_gaps_section_required is True


def test_canary_contract_extracts_entity_requirements() -> None:
    """Author-year canary contracts should flag named-entity extraction requirements."""

    schemas = extract_deliverable_schemas(
        [
            "Annotated bibliography of at least 8 sources",
            "Comparison table with at least 4 rows and columns: "
            "Technique, Signal, Evidence, Caveat",
        ],
        [
            "Compare SAPLMA with Kadavath et al. 2022 and Farquhar et al. 2024 evidence",
        ],
    )

    schema = schemas[0]

    assert schema.entity_extraction_required is True
    assert schema.entity_types == ("author_year", "acronym")


def test_corpus_trace_contract_extracts_traceability_fields() -> None:
    """Corpus trace contracts should compile trace-specific requirements."""

    schemas = extract_deliverable_schemas(
        [
            "Corpus trace table with at least 6 rows and columns: "
            "Claim, Corpus Evidence, Confidence, Gap"
        ],
        [
            "Trace every major claim back to corpus evidence",
            "Document limitations where corpus support is thin",
        ],
    )

    schema = next(item for item in schemas if item.kind == "corpus_trace")

    assert schema.row_identifier_column == "Claim"
    assert schema.corpus_reference_required is True
    assert schema.claim_to_evidence_mapping_required is True
    assert schema.limitations_section_required is True


def test_structural_table_extracts_pipe_separated_columns() -> None:
    """Pipe-delimited column contracts should preserve every required column."""

    schemas = extract_deliverable_schemas(
        [
            "Compact comparison table with columns: "
            "Classifier | Use When | Avoid When | Operational Gotcha."
        ],
        [],
    )

    schema = next(item for item in schemas if item.kind == "table")

    assert schema.columns == (
        "Classifier",
        "Use When",
        "Avoid When",
        "Operational Gotcha",
    )


def test_non_table_words_do_not_trigger_structural_table_detection() -> None:
    """Words like 'suitable' must not compile into a phantom table schema."""

    schemas = extract_deliverable_schemas(
        ["Short markdown research brief suitable for teammate handoff"],
        [],
    )

    assert schemas == []
