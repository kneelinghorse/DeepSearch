"""Structured finding-record primitives and migration helpers."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from pydantic import BaseModel, Field, model_validator

_URL_PATTERN = re.compile(r"https?://[^\s)\]]+")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{1,}")
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def _unique_strings(values: Iterable[Any]) -> List[str]:
    items: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _normalize_text(value)
        if not text or text in seen:
            continue
        items.append(text)
        seen.add(text)
    return items


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(
        token.lower()
        for token in _WORD_PATTERN.findall(text)
        if token.lower() not in _STOPWORDS
    )


def _extract_urls(text: str) -> List[str]:
    return _unique_strings(match.group(0) for match in _URL_PATTERN.finditer(text))


def build_finding_id(summary: str, loop_index: int) -> str:
    """Return a stable finding identifier from summary text and loop index."""

    normalized = _normalize_text(summary).casefold()
    digest = hashlib.sha1(f"{loop_index}:{normalized}".encode("utf-8")).hexdigest()[:12]
    return f"finding-{int(loop_index):03d}-{digest}"


class FindingProvenance(BaseModel):
    """Origin metadata for a structured finding."""

    lane: str = Field(default="", description="Retrieval lane that supplied the primary source")
    query: str = Field(default="", description="Search query associated with the finding")
    loop: int = Field(default=0, ge=0, description="0-indexed research loop")
    source_url: str = Field(default="", description="Primary supporting source URL")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class FindingRecord(BaseModel):
    """Structured finding emitted by research extraction."""

    finding_id: str = Field(default="", description="Stable identifier for the finding")
    concept_id: str | None = Field(
        default=None,
        description="Concept-cluster identifier populated by clustering stages",
    )
    summary: str = Field(description="Concise user-facing finding summary")
    citations: List[str] = Field(
        default_factory=list,
        description="Supporting citation identifiers or URLs",
    )
    objective_ids: List[str] = Field(
        default_factory=list,
        description="Mission objective identifiers supported by this finding",
    )
    contract_slot_tags: List[str] = Field(
        default_factory=list,
        description="Contract evidence-slot identifiers covered by this finding",
    )
    source_urls: List[str] = Field(
        default_factory=list,
        description="Supporting source URLs associated with this finding",
    )
    provenance: FindingProvenance = Field(
        default_factory=FindingProvenance,
        description="Primary provenance metadata for the finding",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from downstream evaluation",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when the finding record was created",
    )
    loop_index: int = Field(
        default=0,
        ge=0,
        description="Compatibility copy of provenance.loop for telemetry consumers",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_payload(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        payload = dict(data)
        objective_id = _normalize_text(payload.pop("objective_id", ""))
        if objective_id and not payload.get("objective_ids"):
            payload["objective_ids"] = [objective_id]

        provenance_payload = payload.get("provenance")
        if isinstance(provenance_payload, Mapping):
            provenance = dict(provenance_payload)
        elif isinstance(provenance_payload, BaseModel):
            provenance = dict(provenance_payload.model_dump())
        else:
            provenance = {}

        loop_index = payload.get("loop_index")
        if loop_index is not None and provenance.get("loop") in (None, ""):
            try:
                provenance["loop"] = int(loop_index)
            except (TypeError, ValueError):
                provenance["loop"] = 0

        source_url = _normalize_text(payload.pop("source_url", ""))
        if source_url and not provenance.get("source_url"):
            provenance["source_url"] = source_url

        if provenance:
            payload["provenance"] = provenance

        summary = _normalize_text(payload.get("summary", ""))
        if summary and not payload.get("finding_id"):
            try:
                normalized_loop = int(payload.get("loop_index") or provenance.get("loop") or 0)
            except (TypeError, ValueError):
                normalized_loop = 0
            payload["finding_id"] = build_finding_id(summary, normalized_loop)

        return payload

    @model_validator(mode="after")
    def _normalize(self) -> "FindingRecord":
        self.summary = _normalize_text(self.summary)
        self.objective_ids = _unique_strings(self.objective_ids)
        self.contract_slot_tags = _unique_strings(self.contract_slot_tags)
        self.source_urls = _unique_strings(
            [*self.source_urls, *(_extract_urls(self.summary) or []), self.provenance.source_url]
        )
        self.citations = _unique_strings(self.citations or self.source_urls)

        if self.loop_index and not self.provenance.loop:
            self.provenance.loop = self.loop_index
        if self.provenance.loop and not self.loop_index:
            self.loop_index = self.provenance.loop
        if not self.provenance.source_url and self.source_urls:
            self.provenance.source_url = self.source_urls[0]
        if not self.finding_id:
            self.finding_id = build_finding_id(self.summary, self.loop_index)
        return self

    @property
    def objective_id(self) -> str:
        """Return the first objective ID for legacy telemetry consumers."""

        return self.objective_ids[0] if self.objective_ids else ""

    def has_coverage(self) -> bool:
        """Return whether the record is tagged to a slot or an objective."""

        return bool(self.objective_ids or self.contract_slot_tags)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def coerce_finding_records(
    records: Sequence[FindingRecord | Mapping[str, Any]],
) -> List[FindingRecord]:
    """Normalize mixed finding-record payloads into ``FindingRecord`` models."""

    normalized: List[FindingRecord] = []
    for record in records:
        if isinstance(record, FindingRecord):
            normalized.append(record.model_copy(deep=True))
            continue
        if isinstance(record, Mapping):
            normalized.append(FindingRecord.model_validate(dict(record)))
    return normalized


def project_finding_summaries(
    records: Sequence[FindingRecord | Mapping[str, Any]],
) -> List[str]:
    """Return the string compatibility view derived from finding records."""

    return [record.summary for record in coerce_finding_records(records) if record.summary]


def coerce_state_finding_records(state: Mapping[str, Any]) -> List[FindingRecord]:
    """Return structured finding records for a state, backfilling legacy strings."""

    raw_records = state.get("finding_records")
    if isinstance(raw_records, Sequence) and not isinstance(raw_records, (str, bytes)):
        records = coerce_finding_records(raw_records)
        if records:
            return records

    legacy_findings = state.get("findings") or []
    if not isinstance(legacy_findings, Sequence) or isinstance(legacy_findings, (str, bytes)):
        return []

    try:
        loop_index = max(int(state.get("research_loop_count") or 0) - 1, 0)
    except (TypeError, ValueError):
        loop_index = 0

    return build_finding_records(
        [item for item in legacy_findings if _normalize_text(item)],
        sources=[],  # S83 m06: sources_found stripped from AgentState at S78 (no writer)
        loop_index=loop_index,
        mission_contract=state.get("mission_contract"),
        mission_objectives=state.get("mission_objectives") or [],
    )


def build_finding_records(
    findings: Sequence[str],
    *,
    sources: Sequence[Mapping[str, Any]],
    loop_index: int,
    mission_contract: Mapping[str, Any] | None,
    mission_objectives: Sequence[str],
    default_query: str = "",
) -> List[FindingRecord]:
    """Build structured finding records from extracted finding strings."""

    objective_catalog = _build_objective_catalog(mission_contract, mission_objectives)
    slot_to_objective = _slot_to_objective_map(mission_contract)
    prepared_sources = [dict(source) for source in sources if isinstance(source, Mapping)]
    records: List[FindingRecord] = []

    for finding_text in findings:
        summary = _normalize_text(finding_text)
        if not summary:
            continue

        supporting_sources = _select_supporting_sources(summary, prepared_sources)
        source_urls = _unique_strings(source.get("url", "") for source in supporting_sources)
        contract_slot_tags = _unique_strings(
            slot
            for source in supporting_sources
            for slot in source.get("target_slots") or []
        )
        objective_ids = _unique_strings(
            slot_to_objective.get(slot, "")
            for slot in contract_slot_tags
            if slot_to_objective.get(slot)
        )
        if not objective_ids:
            objective_ids = _match_objective_ids(summary, objective_catalog)
        if not objective_ids and objective_catalog:
            objective_ids = [objective_catalog[0]["id"]]

        primary_source = supporting_sources[0] if supporting_sources else {}
        provenance = FindingProvenance(
            lane=_normalize_text(primary_source.get("lane", "")),
            query=(
                _normalize_text(primary_source.get("query", ""))
                or _normalize_text(default_query)
            ),
            loop=max(int(loop_index), 0),
            source_url=_normalize_text(primary_source.get("url", ""))
            or (source_urls[0] if source_urls else ""),
        )

        record = FindingRecord(
            finding_id=build_finding_id(summary, loop_index),
            summary=summary,
            citations=_unique_strings([*source_urls, *_extract_urls(summary)]),
            objective_ids=objective_ids,
            contract_slot_tags=contract_slot_tags,
            source_urls=source_urls,
            provenance=provenance,
            confidence=0.0,
            loop_index=max(int(loop_index), 0),
        )
        records.append(record)

    return records


def _build_objective_catalog(
    mission_contract: Mapping[str, Any] | None,
    mission_objectives: Sequence[str],
) -> List[Dict[str, Any]]:
    objectives_payload = []
    if isinstance(mission_contract, Mapping):
        raw = mission_contract.get("objectives")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            objectives_payload = list(raw)

    catalog: List[Dict[str, Any]] = []
    for index, objective in enumerate(objectives_payload):
        if not isinstance(objective, Mapping):
            continue
        identifier = _normalize_text(objective.get("id", ""))
        text = _normalize_text(objective.get("text", ""))
        if not identifier or not text:
            continue
        catalog.append({"id": identifier, "text": text, "tokens": _tokenize(text)})

    if catalog:
        return catalog

    for index, objective_text in enumerate(mission_objectives):
        text = _normalize_text(objective_text)
        if not text:
            continue
        catalog.append(
            {
                "id": f"objective-{index}",
                "text": text,
                "tokens": _tokenize(text),
            }
        )
    return catalog


def _slot_to_objective_map(mission_contract: Mapping[str, Any] | None) -> Dict[str, str]:
    slots_payload = []
    if isinstance(mission_contract, Mapping):
        raw = mission_contract.get("evidence_slots")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            slots_payload = list(raw)

    mapping: Dict[str, str] = {}
    for slot in slots_payload:
        if not isinstance(slot, Mapping):
            continue
        slot_id = _normalize_text(slot.get("id", ""))
        target = slot.get("target")
        objective_id = ""
        if isinstance(target, Mapping):
            objective_id = _normalize_text(target.get("objective_id", ""))
        if slot_id and objective_id:
            mapping[slot_id] = objective_id
    return mapping


def _select_supporting_sources(
    summary: str,
    sources: Sequence[Mapping[str, Any]],
    *,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    if not sources:
        return []

    explicit_urls = set(_extract_urls(summary))
    summary_tokens = _tokenize(summary)
    ranked: List[tuple[float, int, Dict[str, Any]]] = []

    for index, source in enumerate(sources):
        payload = dict(source)
        url = _normalize_text(payload.get("url", ""))
        score = 0.0
        if url and url in explicit_urls:
            score += 100.0

        source_text = " ".join(
            _normalize_text(payload.get(field, ""))
            for field in ("title", "snippet", "content", "query")
        )
        source_tokens = _tokenize(source_text)
        if summary_tokens and source_tokens:
            score += float(len(summary_tokens & source_tokens))
        if payload.get("target_slots"):
            score += 0.25
        if url:
            score += 0.01
        if score <= 0.0 and len(sources) == 1:
            score = 1.0

        ranked.append((score, index, payload))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = [payload for score, _, payload in ranked if score > 0.0][:limit]
    if selected:
        return selected
    return [dict(sources[0])]


def _match_objective_ids(summary: str, catalog: Sequence[Mapping[str, Any]]) -> List[str]:
    if not catalog:
        return []
    if len(catalog) == 1:
        return [_normalize_text(catalog[0].get("id", ""))]

    summary_tokens = _tokenize(summary)
    if not summary_tokens:
        return []

    ranked: List[tuple[int, int, str]] = []
    for index, objective in enumerate(catalog):
        objective_id = _normalize_text(objective.get("id", ""))
        tokens = objective.get("tokens")
        objective_tokens = (
            tokens
            if isinstance(tokens, frozenset)
            else _tokenize(objective.get("text", ""))
        )
        overlap = len(summary_tokens & objective_tokens)
        if objective_id and overlap > 0:
            ranked.append((overlap, index, objective_id))

    if not ranked:
        return []

    ranked.sort(key=lambda item: (-item[0], item[1]))
    best_overlap = ranked[0][0]
    if best_overlap < 2:
        return [ranked[0][2]]

    overlap_floor = best_overlap if best_overlap == 2 else best_overlap - 1
    matched = [
        objective_id
        for overlap, _, objective_id in ranked
        if overlap >= overlap_floor
    ]
    return _unique_strings(matched)[:3]


__all__ = [
    "FindingProvenance",
    "FindingRecord",
    "build_finding_id",
    "build_finding_records",
    "coerce_finding_records",
    "coerce_state_finding_records",
    "project_finding_summaries",
]
