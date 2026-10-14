"""Unit tests for structured finding concept clustering."""

from __future__ import annotations

from deepsearch.agent.concept_clustering import cluster_findings, cluster_findings_node
from deepsearch.agent.finding import FindingProvenance, FindingRecord


def _record(
    *,
    finding_id: str,
    summary: str,
    loop_index: int,
    confidence: float,
    source_urls: list[str],
    contract_slot_tags: list[str] | None = None,
) -> FindingRecord:
    return FindingRecord(
        finding_id=finding_id,
        summary=summary,
        citations=list(source_urls),
        objective_ids=["objective-0"],
        contract_slot_tags=list(contract_slot_tags or []),
        source_urls=list(source_urls),
        provenance=FindingProvenance(
            loop=loop_index,
            source_url=source_urls[0] if source_urls else "",
        ),
        confidence=confidence,
        loop_index=loop_index,
    )


def test_cluster_findings_is_idempotent() -> None:
    records = [
        _record(
            finding_id="finding-001",
            summary="OAuth 2.0 is the primary authentication mechanism for the API.",
            loop_index=0,
            confidence=0.4,
            source_urls=["https://example.com/auth"],
        ),
        _record(
            finding_id="finding-002",
            summary="The API primarily relies on OAuth 2.0 authentication.",
            loop_index=1,
            confidence=0.6,
            source_urls=["https://example.com/auth"],
        ),
        _record(
            finding_id="finding-003",
            summary="Rate limits are capped at 60 requests per minute for free accounts.",
            loop_index=1,
            confidence=0.5,
            source_urls=["https://example.com/rate-limits"],
        ),
    ]

    first_records, first_clusters = cluster_findings(records)
    second_records, second_clusters = cluster_findings(first_records)

    assert [record.model_dump(mode="json") for record in first_records] == [
        record.model_dump(mode="json") for record in second_records
    ]
    assert [cluster.to_dict() for cluster in first_clusters] == [
        cluster.to_dict() for cluster in second_clusters
    ]


def test_cluster_findings_uses_deterministic_representative_selection() -> None:
    records = [
        _record(
            finding_id="finding-z",
            summary="Alpha finding with stable evidence coverage.",
            loop_index=1,
            confidence=0.9,
            source_urls=["https://example.com/alpha"],
        ),
        _record(
            finding_id="finding-a",
            summary="Alpha finding with stable evidence coverage and added detail.",
            loop_index=1,
            confidence=0.9,
            source_urls=["https://example.com/alpha"],
        ),
        _record(
            finding_id="finding-b",
            summary="Alpha finding with stable evidence coverage.",
            loop_index=0,
            confidence=0.9,
            source_urls=["https://example.com/alpha"],
        ),
    ]

    tagged_records, concept_clusters = cluster_findings(records)

    assert len(concept_clusters) == 1
    assert concept_clusters[0].representative_id == "finding-b"
    assert all(record.concept_id == "concept-finding-b" for record in tagged_records)


def test_cluster_findings_merges_urls_and_slot_tags() -> None:
    records = [
        _record(
            finding_id="finding-010",
            summary="SAPLMA improves retrieval planning for audit workflows.",
            loop_index=0,
            confidence=0.8,
            source_urls=["https://example.com/saplma"],
            contract_slot_tags=["slot-entity-saplma"],
        ),
        _record(
            finding_id="finding-011",
            summary="SAPLMA improves retrieval planning for audit workflows with bounded recall.",
            loop_index=1,
            confidence=0.7,
            source_urls=["https://example.com/audit-bounds"],
            contract_slot_tags=["slot-metric-bounded-recall"],
        ),
    ]

    _, concept_clusters = cluster_findings(records)

    assert concept_clusters[0].merged_citations == [
        "https://example.com/saplma",
        "https://example.com/audit-bounds",
    ]
    assert concept_clusters[0].merged_contract_slot_tags == [
        "slot-entity-saplma",
        "slot-metric-bounded-recall",
    ]
    assert concept_clusters[0].member_count == 2


def test_cluster_findings_node_updates_state_payload() -> None:
    state = {
        "finding_records": [
            _record(
                finding_id="finding-100",
                summary="Citation-linked duplicate finding.",
                loop_index=0,
                confidence=0.5,
                source_urls=["https://example.com/source"],
            ).model_dump(mode="json"),
            _record(
                finding_id="finding-101",
                summary="Citation-linked duplicate finding with slight extra detail.",
                loop_index=1,
                confidence=0.7,
                source_urls=["https://example.com/source"],
            ).model_dump(mode="json"),
        ]
    }

    update = cluster_findings_node(state)

    assert len(update["concept_clusters"]) == 1
    assert update["finding_records"][0]["concept_id"] == (
        update["concept_clusters"][0]["concept_id"]
    )
    assert update["findings"] == [
        "Citation-linked duplicate finding.",
        "Citation-linked duplicate finding with slight extra detail.",
    ]
