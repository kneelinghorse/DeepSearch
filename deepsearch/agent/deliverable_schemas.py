"""Deliverable schema extraction for synthesis-time output contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import List, Sequence


@dataclass(frozen=True)
class OutputSchema:
    """Structured output contract inferred from mission deliverables."""

    kind: str
    title: str
    format: str
    source_text: str
    min_items: int | None = None
    minimum_citation_count: int | None = None
    columns: tuple[str, ...] = ()
    allowed_values: tuple[str, ...] = ()
    required_sections: tuple[str, ...] = ()
    entity_extraction_required: bool = False
    entity_types: tuple[str, ...] = ()
    row_identifier_column: str | None = None
    recommendation_required: bool = False
    evidence_gaps_section_required: bool = False
    corpus_reference_required: bool = False
    claim_to_evidence_mapping_required: bool = False
    limitations_section_required: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AnnotationDepthFloor:
    """Tier-scaled bibliography annotation depth requirements."""

    min_sentences: int
    min_words: int
    min_content_tokens: int


_ANNOTATION_DEPTH_FLOORS: dict[str, AnnotationDepthFloor] = {
    "baseline": AnnotationDepthFloor(min_sentences=2, min_words=30, min_content_tokens=30),
    "deep": AnnotationDepthFloor(min_sentences=3, min_words=60, min_content_tokens=60),
    "alpha": AnnotationDepthFloor(min_sentences=4, min_words=100, min_content_tokens=100),
}

_COUNT_PATTERNS = (
    re.compile(r"(?:at least|minimum of|min(?:imum)?)\s+(\d+)", re.IGNORECASE),
    re.compile(
        r"(\d+)\+\s*(?:rows?|papers?|entries|items|sources|citations?|techniques|recommendations)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\d+)\s*-\s*\d+\s*(?:rows?|papers?|entries|items|sources|citations?|recommendations)",
        re.IGNORECASE,
    ),
)
_CITATION_COUNT_PATTERNS = (
    re.compile(
        r"(?:at least|minimum of|min(?:imum)?)\s+(\d+)\s+(?:papers?|citations?|sources?|entries)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(\d+)\+\s*(?:papers?|citations?|sources?|entries)\b", re.IGNORECASE),
    re.compile(
        r"(\d+)\s*-\s*\d+\s*(?:papers?|citations?|sources?|entries)\b",
        re.IGNORECASE,
    ),
)
_COLUMNS_PATTERN = re.compile(r"columns?\s*(?:named|:)\s*([^\n]+)", re.IGNORECASE)
_TABLE_WORD_PATTERN = re.compile(r"\btable\b", re.IGNORECASE)
_AUTHOR_YEAR_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+et al\.)?\s+\d{4}\b")
_ACRONYM_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{2,}\b")
_ATTRIBUTE_LIST_INTRO_PATTERN = re.compile(
    r"(?:\bfor\s+each\b|\bfor\s+every\b|\bper\b)[^.!?;\n]*?"
    r"\b(?:document|record|capture|include|report|summarize|compare)\b"
    r"(?P<attrs>[^.!?;\n]+)",
    re.IGNORECASE,
)
_ATTRIBUTE_VERB_INTRO_PATTERN = re.compile(
    r"^\s*(?:document|record|capture|include|report|summarize|compare)\b"
    r"(?P<attrs>[^.!?;\n]+)",
    re.IGNORECASE,
)
_ATTRIBUTE_LEADING_INTERROGATIVE_PATTERN = re.compile(
    r"^(?:whether|what|how|which)\b",
    re.IGNORECASE,
)
_ATTRIBUTE_NOMINAL_PATTERN = re.compile(
    r"\b(?:support|capabilit(?:y|ies)|mechanisms?|counts?)\b",
    re.IGNORECASE,
)
_ATTRIBUTE_RELATIONAL_PATTERN = re.compile(
    r"\bread\s+vs\.?\s+write\b|\btool\s+counts?\b",
    re.IGNORECASE,
)


def extract_deliverable_schemas(
    mission_deliverables: Sequence[str] | None,
    success_criteria: Sequence[str] | None,
) -> List[OutputSchema]:
    """Extract structured output contracts from deliverables and criteria."""

    inputs = [
        str(item).strip()
        for item in [*(mission_deliverables or []), *(success_criteria or [])]
        if str(item).strip()
    ]
    if not inputs:
        return []

    schemas: List[OutputSchema] = []
    seen: set[tuple[str, str]] = set()
    combined_text = "\n".join(inputs)
    combined_lowered = combined_text.lower()
    entity_types = _extract_entity_types(combined_text)

    for text in inputs:
        lowered = text.lower()
        schema: OutputSchema | None = None

        if _is_annotated_bibliography(lowered):
            schema = OutputSchema(
                kind="annotated_bibliography",
                title="Annotated Bibliography",
                format="table",
                source_text=text,
                min_items=_extract_min_items(text),
                minimum_citation_count=_extract_minimum_citation_count(text),
                columns=_extract_columns(text) or _default_bibliography_columns(lowered),
                notes=(
                    "Use one source or paper per row.",
                    "Do not replace the bibliography with bullet-point prose.",
                ),
            )
        elif _is_classification_matrix(lowered):
            schema = OutputSchema(
                kind="classification_matrix",
                title="Classification Matrix",
                format="table",
                source_text=text,
                min_items=_extract_min_items(text),
                columns=_extract_columns(text) or ("Technique", "Classification", "Evidence"),
                row_identifier_column=_extract_row_identifier_column(
                    _extract_columns(text) or ("Technique", "Classification", "Evidence")
                ),
                allowed_values=_extract_allowed_values(lowered),
                notes=(
                    "Include one row per named technique or method discussed in the findings.",
                    "Classification values must be explicit, not implied in prose.",
                ),
            )
        elif _is_prototype_section(lowered):
            schema = OutputSchema(
                kind="prototype_section",
                title="Candidate Metric for Prototyping",
                format="section",
                source_text=text,
                required_sections=(
                    "Metric Definition",
                    "Inputs",
                    "Computation Steps",
                    "Interpretation",
                    "Implementation Notes",
                ),
                notes=(
                    "Make the metric concrete enough that an engineer could prototype it.",
                    "Do not collapse this section into general recommendations.",
                ),
            )
        elif _is_corpus_trace(lowered):
            columns = _extract_columns(text) or (
                "Claim",
                "Corpus Evidence",
                "Confidence",
                "Gap",
            )
            schema = OutputSchema(
                kind="corpus_trace",
                title="Corpus Trace Table",
                format="table",
                source_text=text,
                min_items=_extract_min_items(text),
                columns=columns,
                row_identifier_column=_extract_row_identifier_column(columns),
                notes=(
                    "Map each claim to corpus-backed evidence instead of summarizing loosely.",
                    "Call out gaps where the corpus does not fully support the claim.",
                ),
            )
        elif _is_structural_table(lowered):
            schema = OutputSchema(
                kind="table",
                title=_infer_table_title(text),
                format="table",
                source_text=text,
                min_items=_extract_min_items(text),
                columns=_extract_columns(text),
                notes=("Satisfy the required table structure explicitly; do not narrate it away.",),
            )

        if schema is None:
            attribute_columns = extract_deliverable_attribute_columns(text)
            if len(attribute_columns) >= 2:
                schema = OutputSchema(
                    kind="per_candidate_matrix",
                    title="Per-Candidate Matrix",
                    format="table",
                    source_text=text,
                    columns=("Candidate", *attribute_columns),
                    row_identifier_column="Candidate",
                    notes=(
                        "Use one row per candidate and one column per required attribute.",
                        "Do not turn per-candidate attributes into named-entity coverage gaps.",
                    ),
                )

        if schema is None:
            continue

        schema_key = (schema.kind, schema.title.lower())
        if schema_key in seen:
            continue
        seen.add(schema_key)
        schemas.append(schema)

    return _apply_cross_input_requirements(
        schemas,
        combined_lowered=combined_lowered,
        entity_types=entity_types,
    )


def extract_deliverable_attribute_columns(text: str) -> tuple[str, ...]:
    """Return per-candidate attribute columns declared in objective prose."""

    candidates: List[str] = []
    for raw_attrs in _iter_attribute_list_texts(text):
        parts = _split_attribute_parts(raw_attrs)
        if not _parts_form_attribute_list(parts):
            continue
        for part in parts:
            column = _normalize_attribute_column(part)
            if column:
                candidates.append(column)

    return tuple(_unique_preserving_order(candidates))


def is_deliverable_attribute_phrase(text: str) -> bool:
    """Classify requirement predicates that belong in schema columns, not entities."""

    normalized = " ".join(str(text or "").strip(" .`'\"()").split())
    if not normalized:
        return False
    return bool(
        _ATTRIBUTE_LEADING_INTERROGATIVE_PATTERN.search(normalized)
        or _ATTRIBUTE_NOMINAL_PATTERN.search(normalized)
        or _ATTRIBUTE_RELATIONAL_PATTERN.search(normalized)
    )


def is_deliverable_attribute_list(text: str) -> bool:
    """Return True when text carries a per-candidate attribute enumeration."""

    return bool(extract_deliverable_attribute_columns(text))


def _iter_attribute_list_texts(text: str) -> List[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []

    matches = [
        match.group("attrs")
        for match in _ATTRIBUTE_LIST_INTRO_PATTERN.finditer(cleaned)
        if match.group("attrs").strip()
    ]
    if matches:
        return matches

    match = _ATTRIBUTE_VERB_INTRO_PATTERN.search(cleaned)
    if match and match.group("attrs").strip():
        return [match.group("attrs")]
    return [cleaned]


def _split_attribute_parts(text: str) -> List[str]:
    return [
        item.strip(" .`'\"()")
        for item in re.split(r",|\band\b", str(text or ""))
        if item.strip(" .`'\"()")
    ]


def _parts_form_attribute_list(parts: Sequence[str]) -> bool:
    if len(parts) < 2:
        return False
    attribute_indexes = [
        index for index, part in enumerate(parts) if is_deliverable_attribute_phrase(part)
    ]
    if len(attribute_indexes) < 2:
        return False
    return attribute_indexes[0] <= 1


def _normalize_attribute_column(text: str) -> str | None:
    cleaned = " ".join(str(text or "").strip(" .`'\"()").split())
    if not cleaned:
        return None
    cleaned = re.sub(r"\bwhere\s+available\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.strip(" .").split())
    if not cleaned:
        return None
    return cleaned[0].upper() + cleaned[1:]


def _unique_preserving_order(items: Sequence[str]) -> List[str]:
    unique: List[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        unique.append(item)
        seen.add(key)
    return unique


def resolve_annotation_depth_floor(research_depth: str | None) -> AnnotationDepthFloor:
    """Return bibliography annotation floors for a research depth tier."""

    normalized = str(research_depth or "").strip().lower()
    return _ANNOTATION_DEPTH_FLOORS.get(normalized, _ANNOTATION_DEPTH_FLOORS["baseline"])


def render_output_schema_contracts(
    schemas: Sequence[OutputSchema],
    *,
    research_depth: str | None = None,
) -> str:
    """Render output contracts as prompt instructions."""

    if not schemas:
        return ""

    annotation_floor = resolve_annotation_depth_floor(research_depth)
    blocks = ["**REQUIRED OUTPUT CONTRACTS:**"]
    for index, schema in enumerate(schemas, start=1):
        blocks.append(f"{index}. **{schema.title}**")
        if schema.format == "table":
            blocks.append(
                f"   You MUST produce a markdown table section for `{schema.title}`."
            )
            if schema.min_items is not None:
                blocks.append(f"   The table MUST contain at least {schema.min_items} rows.")
            if schema.columns:
                columns = " | ".join(schema.columns)
                blocks.append(f"   Use these columns exactly: `{columns}`.")
        elif schema.format == "section":
            blocks.append(
                f"   You MUST produce a dedicated markdown section titled "
                f"`## {schema.title}`."
            )
            if schema.required_sections:
                sections = ", ".join(f"`{name}`" for name in schema.required_sections)
                blocks.append(f"   Include these subsections: {sections}.")

        if schema.allowed_values:
            allowed_values = ", ".join(f"`{value}`" for value in schema.allowed_values)
            blocks.append(
                "   Restrict the classification/value field to these labels: "
                f"{allowed_values}."
            )

        if schema.kind == "annotated_bibliography":
            blocks.append(
                "   Each Paper cell MUST begin with the citation index `[n]` "
                "matching the numbered source list (for example `[3] Paper Title`). "
                "The `[n]` anchor prefix is mandatory — a bibliography row without "
                "an `[n]` prefix does not satisfy the contract and downstream "
                "citation-coherence validation cannot score the row."
            )
            blocks.append(
                "   Each Paper cell MUST NOT contain an unescaped `|` pipe "
                "character anywhere inside the paper title or source label. "
                "Markdown-table parsing splits cells on every `|`, so a Paper "
                "rendered as `**[12] Title | Source**` collapses the row and it "
                "is dropped from the bibliography row count. When a source "
                "title or publication venue contains a pipe, replace the pipe "
                "with an em-dash `—` or a colon `: ` before emitting the row "
                "(for example `**[12] Title — Source**` or `**[12] Title: "
                "Source**`). This rule is mandatory and applies to every row."
            )
            blocks.append(
                "   Each annotation row MUST include at least "
                f"{annotation_floor.min_sentences} sentences, "
                f"{annotation_floor.min_words} words, and "
                f"{annotation_floor.min_content_tokens} content tokens "
                f"for the {str(research_depth or 'baseline').strip().upper() or 'BASELINE'} tier."
            )
            blocks.append(
                "   Thin label-only annotations do not satisfy the bibliography contract."
            )

        for note in schema.notes:
            blocks.append(f"   {note}")

        blocks.append(f"   Trigger text: `{schema.source_text}`")

    return "\n".join(blocks)


def _extract_min_items(text: str) -> int | None:
    for pattern in _COUNT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        try:
            value = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if 1 <= value <= 200:
            return value
    return None


def _extract_minimum_citation_count(text: str) -> int | None:
    for pattern in _CITATION_COUNT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        try:
            value = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if 1 <= value <= 200:
            return value
    return _extract_min_items(text)


def _extract_columns(text: str) -> tuple[str, ...]:
    match = _COLUMNS_PATTERN.search(text)
    if not match:
        return ()
    raw = match.group(1).strip()
    raw = re.split(r"[.;]\s+(?=[A-Z])|[.;]\s*$", raw, maxsplit=1)[0].strip()
    parts = [
        item.strip(" .`|")
        for item in re.split(r"\||,|/| and ", raw)
        if item.strip(" .")
    ]
    return tuple(parts)


def _extract_allowed_values(lowered_text: str) -> tuple[str, ...]:
    if "production/research/theoretical" in lowered_text:
        return ("production", "research", "theoretical")
    if (
        "production" in lowered_text
        and "research" in lowered_text
        and "theoretical" in lowered_text
    ):
        return ("production", "research", "theoretical")
    return ()


def _default_bibliography_columns(lowered_text: str) -> tuple[str, ...]:
    if _extract_allowed_values(lowered_text):
        return ("Paper", "Contribution", "Classification")
    return ("Paper", "Contribution", "Evidence")


def _infer_table_title(text: str) -> str:
    if "matrix" in text.lower():
        return "Matrix"
    if "comparison" in text.lower():
        return "Comparison Table"
    return "Required Table"


def _is_annotated_bibliography(lowered_text: str) -> bool:
    return "annotated bibliography" in lowered_text or "bibliography" in lowered_text


def _is_classification_matrix(lowered_text: str) -> bool:
    return (
        "classification matrix" in lowered_text
        or "matrix" in lowered_text
        or "production/research/theoretical" in lowered_text
    )


def _is_prototype_section(lowered_text: str) -> bool:
    return (
        ("prototype" in lowered_text and "metric" in lowered_text)
        or "specify concretely enough to prototype" in lowered_text
        or ("computation steps" in lowered_text and "metric" in lowered_text)
    )


def _is_corpus_trace(lowered_text: str) -> bool:
    return (
        "corpus trace" in lowered_text
        or (
            "corpus" in lowered_text
            and "trace" in lowered_text
            and "claim" in lowered_text
            and "evidence" in lowered_text
        )
        or ("claim" in lowered_text and "corpus evidence" in lowered_text)
    )


def _is_structural_table(lowered_text: str) -> bool:
    return bool(_TABLE_WORD_PATTERN.search(lowered_text))


def _extract_row_identifier_column(columns: tuple[str, ...]) -> str | None:
    if not columns:
        return None
    return columns[0]


def _extract_entity_types(text: str) -> tuple[str, ...]:
    entity_types: List[str] = []
    if _AUTHOR_YEAR_ENTITY_PATTERN.search(text):
        entity_types.append("author_year")
    if _ACRONYM_ENTITY_PATTERN.search(text):
        entity_types.append("acronym")
    return tuple(entity_types)


def _apply_cross_input_requirements(
    schemas: Sequence[OutputSchema],
    *,
    combined_lowered: str,
    entity_types: tuple[str, ...],
) -> List[OutputSchema]:
    updated: List[OutputSchema] = []
    entity_requirement_applied = False
    recommendation_required = "recommendation" in combined_lowered
    evidence_gaps_required = (
        "evidence gap" in combined_lowered or "evidence gaps" in combined_lowered
    )
    corpus_reference_required = "corpus" in combined_lowered
    claim_mapping_required = "claim" in combined_lowered and "evidence" in combined_lowered
    limitations_required = "limitation" in combined_lowered

    for schema in schemas:
        updated_schema = schema
        if schema.kind == "classification_matrix" and recommendation_required:
            updated_schema = replace(updated_schema, recommendation_required=True)
        if schema.kind == "annotated_bibliography" and evidence_gaps_required:
            updated_schema = replace(updated_schema, evidence_gaps_section_required=True)
        if schema.kind == "corpus_trace":
            updated_schema = replace(
                updated_schema,
                corpus_reference_required=corpus_reference_required,
                claim_to_evidence_mapping_required=claim_mapping_required,
                limitations_section_required=limitations_required,
            )
        if not entity_requirement_applied and "author_year" in entity_types:
            updated_schema = replace(
                updated_schema,
                entity_extraction_required=True,
                entity_types=entity_types,
            )
            entity_requirement_applied = True
        updated.append(updated_schema)

    return updated


__all__ = [
    "AnnotationDepthFloor",
    "OutputSchema",
    "extract_deliverable_attribute_columns",
    "extract_deliverable_schemas",
    "is_deliverable_attribute_list",
    "is_deliverable_attribute_phrase",
    "resolve_annotation_depth_floor",
    "render_output_schema_contracts",
]
