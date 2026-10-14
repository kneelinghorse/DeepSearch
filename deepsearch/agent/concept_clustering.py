"""Concept clustering for structured findings before reflection.

This runs as a dedicated node between research and reflection so clustering is
checkpoint-visible and independently testable instead of being buried in
research extraction or finalize-time synthesis. The default threshold is 0.7:
that keeps clustering conservative enough for near-duplicates while still
folding citation-identical paraphrases into one concept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .finding import (
    FindingRecord,
    coerce_finding_records,
    coerce_state_finding_records,
    project_finding_summaries,
)
from .finding_similarity import (
    content_tokens,
    extract_urls_from_values,
    jaccard_similarity,
    strip_citation_tokens,
    unique_strings,
)
from .state import AgentState
from .telemetry import clone_telemetry, update_finding_counts

StateUpdate = Dict[str, Any]


@dataclass(slots=True)
class ConceptCluster:
    """Aggregated concept cluster derived from one or more finding records."""

    concept_id: str
    members: List[str]
    representative_id: str
    merged_citations: List[str]
    merged_contract_slot_tags: List[str]
    member_count: int

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConceptCluster":
        """Coerce serialized payloads back into a cluster instance."""

        return cls(
            concept_id=str(payload.get("concept_id", "")),
            members=unique_strings(payload.get("members", [])),
            representative_id=str(payload.get("representative_id", "")),
            merged_citations=unique_strings(payload.get("merged_citations", [])),
            merged_contract_slot_tags=unique_strings(payload.get("merged_contract_slot_tags", [])),
            member_count=int(payload.get("member_count", 0) or 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "concept_id": self.concept_id,
            "members": list(self.members),
            "representative_id": self.representative_id,
            "merged_citations": list(self.merged_citations),
            "merged_contract_slot_tags": list(self.merged_contract_slot_tags),
            "member_count": self.member_count,
        }


@dataclass(slots=True)
class _SimilarityProfile:
    summary_tokens: frozenset[str]
    citation_urls: frozenset[str]


@dataclass(slots=True)
class _ConceptAccumulator:
    records: List[FindingRecord] = field(default_factory=list)
    citation_urls: List[str] = field(default_factory=list)
    contract_slot_tags: List[str] = field(default_factory=list)
    representative: FindingRecord | None = None
    representative_profile: _SimilarityProfile | None = None

    def add(self, record: FindingRecord, profile: _SimilarityProfile) -> None:
        self.records.append(record)
        self.citation_urls = unique_strings([*self.citation_urls, *profile.citation_urls])
        self.contract_slot_tags = unique_strings(
            [*self.contract_slot_tags, *record.contract_slot_tags]
        )
        if self.representative is None or _representative_key(record) < _representative_key(
            self.representative
        ):
            self.representative = record
            self.representative_profile = profile

    def render(self) -> ConceptCluster:
        if self.representative is None:
            raise ValueError("Concept accumulator has no representative")
        concept_id = f"concept-{self.representative.finding_id}"
        member_ids = [record.finding_id for record in self.records]
        return ConceptCluster(
            concept_id=concept_id,
            members=member_ids,
            representative_id=self.representative.finding_id,
            merged_citations=list(self.citation_urls),
            merged_contract_slot_tags=list(self.contract_slot_tags),
            member_count=len(member_ids),
        )


def cluster_findings(
    records: Sequence[FindingRecord | Mapping[str, Any]],
    *,
    threshold: float = 0.7,
) -> Tuple[List[FindingRecord], List[ConceptCluster]]:
    """Cluster finding records by concept and assign deterministic concept IDs."""

    normalized_records = coerce_finding_records(records)
    if not normalized_records:
        return [], []

    accumulators: List[_ConceptAccumulator] = []
    concept_by_finding_id: Dict[str, str] = {}

    for record in normalized_records:
        profile = _build_similarity_profile(record)
        matched: _ConceptAccumulator | None = None
        best_score = 0.0

        for accumulator in accumulators:
            candidate_profile = accumulator.representative_profile
            if accumulator.representative is None or candidate_profile is None:
                continue

            score = _similarity_score(profile, candidate_profile)
            if score >= threshold and score > best_score:
                matched = accumulator
                best_score = score

        if matched is None:
            matched = _ConceptAccumulator()
            accumulators.append(matched)
        matched.add(record, profile)

    concept_clusters = [accumulator.render() for accumulator in accumulators]
    for cluster in concept_clusters:
        for member_id in cluster.members:
            concept_by_finding_id[member_id] = cluster.concept_id

    tagged_records = [
        record.model_copy(update={"concept_id": concept_by_finding_id.get(record.finding_id)})
        for record in normalized_records
    ]
    return tagged_records, concept_clusters


def cluster_findings_node(state: AgentState) -> StateUpdate:
    """Project structured findings into concept clusters before reflection."""

    tagged_records, concept_clusters = cluster_findings(coerce_state_finding_records(state))
    telemetry = clone_telemetry(state)
    loop_index = max(int(state.get("research_loop_count", 0) or 0) - 1, 0)
    finding_metrics = summarize_state_findings(
        {
            **state,
            "finding_records": [record.to_dict() for record in tagged_records],
            "concept_clusters": [cluster.to_dict() for cluster in concept_clusters],
        }
    )
    update_finding_counts(
        telemetry,
        raw_findings=int(finding_metrics["raw_findings"]),
        unique_concepts=int(finding_metrics["unique_concepts"]),
        cluster_count=int(finding_metrics["cluster_count"]),
        loop_index=loop_index,
    )
    return {
        "finding_records": [record.to_dict() for record in tagged_records],
        "findings": project_finding_summaries(tagged_records),
        "concept_clusters": [cluster.to_dict() for cluster in concept_clusters],
        "telemetry": telemetry,
    }


def coerce_concept_clusters(
    clusters: Sequence[ConceptCluster | Mapping[str, Any]],
) -> List[ConceptCluster]:
    """Normalize serialized concept clusters into dataclass instances."""

    normalized: List[ConceptCluster] = []
    for cluster in clusters:
        if isinstance(cluster, ConceptCluster):
            normalized.append(cluster)
        elif isinstance(cluster, Mapping):
            normalized.append(ConceptCluster.from_dict(cluster))
    return normalized


def coerce_state_concept_clusters(state: Mapping[str, Any]) -> List[ConceptCluster]:
    """Return concept clusters from state, backfilling from finding records if needed."""

    raw_clusters = state.get("concept_clusters")
    if isinstance(raw_clusters, Sequence) and not isinstance(raw_clusters, (str, bytes)):
        normalized = coerce_concept_clusters(raw_clusters)
        if normalized:
            return normalized
    raw_records = state.get("finding_records")
    if isinstance(raw_records, Sequence) and not isinstance(raw_records, (str, bytes)):
        _, concept_clusters = cluster_findings(coerce_state_finding_records(state))
        return concept_clusters
    return []


def representative_findings(
    records: Sequence[FindingRecord | Mapping[str, Any]],
    concept_clusters: Sequence[ConceptCluster | Mapping[str, Any]],
) -> List[FindingRecord]:
    """Return the representative finding record for each concept cluster."""

    record_by_id = {
        record.finding_id: record
        for record in coerce_finding_records(records)
        if record.finding_id
    }
    representatives: List[FindingRecord] = []
    for cluster in coerce_concept_clusters(concept_clusters):
        representative = record_by_id.get(cluster.representative_id)
        if representative is not None:
            representatives.append(representative)
    return representatives


def build_cluster_summaries(
    records: Sequence[FindingRecord | Mapping[str, Any]],
    concept_clusters: Sequence[ConceptCluster | Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return prompt-friendly summaries for each concept cluster."""

    record_by_id = {
        record.finding_id: record
        for record in coerce_finding_records(records)
        if record.finding_id
    }
    summaries: List[Dict[str, Any]] = []
    for cluster in coerce_concept_clusters(concept_clusters):
        members = [
            record_by_id[member_id]
            for member_id in cluster.members
            if member_id in record_by_id
        ]
        representative = record_by_id.get(cluster.representative_id)
        if representative is None and members:
            representative = members[0]
        if representative is None:
            continue
        objective_ids = unique_strings(
            objective_id
            for member in members
            for objective_id in member.objective_ids
        )
        summaries.append(
            {
                "concept_id": cluster.concept_id,
                "representative_id": cluster.representative_id,
                "representative_summary": representative.summary,
                "member_count": cluster.member_count,
                "objective_ids": objective_ids,
                "contract_slot_tags": list(cluster.merged_contract_slot_tags),
                "citations": list(cluster.merged_citations),
            }
        )
    return summaries


def summarize_state_findings(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return concept-aware display findings plus raw-vs-unique counts."""

    records = coerce_state_finding_records(state)
    concept_clusters = coerce_state_concept_clusters(state)
    raw_strings = unique_strings(state.get("findings") or [])

    raw_findings = len(records) if records else len(raw_strings)
    unique_concepts = len(concept_clusters) if concept_clusters else raw_findings
    cluster_count = len(concept_clusters) if concept_clusters else unique_concepts

    representative_summaries = [
        record.summary
        for record in representative_findings(records, concept_clusters)
        if record.summary
    ]
    display_findings = representative_summaries or project_finding_summaries(records) or raw_strings
    dedup_ratio = (
        round((raw_findings - unique_concepts) / raw_findings, 4)
        if raw_findings
        else 0.0
    )
    return {
        "display_findings": display_findings,
        "raw_findings": raw_findings,
        "unique_concepts": unique_concepts,
        "cluster_count": cluster_count,
        "deduped_findings": unique_concepts,
        "dedup_ratio": dedup_ratio,
    }


def _representative_key(record: FindingRecord) -> tuple[float, int, str]:
    finding_id = record.finding_id or ""
    return (-float(record.confidence), int(record.loop_index), finding_id)


def _build_similarity_profile(record: FindingRecord) -> _SimilarityProfile:
    summary_text = strip_citation_tokens(record.summary) or record.summary
    citation_urls = extract_urls_from_values(
        [*record.citations, *record.source_urls, record.provenance.source_url]
    )
    return _SimilarityProfile(
        summary_tokens=content_tokens(summary_text),
        citation_urls=frozenset(citation_urls),
    )


def _similarity_score(left: _SimilarityProfile, right: _SimilarityProfile) -> float:
    summary_score = jaccard_similarity(left.summary_tokens, right.summary_tokens)
    citation_score = jaccard_similarity(left.citation_urls, right.citation_urls)
    return max(summary_score, citation_score)


__all__ = [
    "ConceptCluster",
    "build_cluster_summaries",
    "cluster_findings",
    "cluster_findings_node",
    "coerce_concept_clusters",
    "coerce_state_concept_clusters",
    "representative_findings",
    "summarize_state_findings",
]
