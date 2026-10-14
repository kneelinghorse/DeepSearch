"""Evidence capture and telemetry instrumentation for DeepSearch missions.

This module provides structured evidence capture and telemetry persistence
for research loops. All findings, sources, and telemetry are persisted to
JSONL files for TraceLab integration and diagnostic analysis.

Schema version: 1.0
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

from pydantic import BaseModel, Field

from deepsearch.data_paths import data_path

from .finding import (
    FindingProvenance,
    FindingRecord,
    build_finding_id,
    coerce_state_finding_records,
)

logger = logging.getLogger(__name__)

TELEMETRY_DIR_ENV = "DEEPSEARCH_TELEMETRY_DIR"
TELEMETRY_SCHEMA_VERSION = "1.0"


Finding = FindingRecord


class SourceEvidence(BaseModel):
    """Metadata for a source gathered during research."""

    url: str = Field(description="Source URL")
    title: str = Field(default="", description="Source title")
    snippet: str = Field(default="", description="Relevant excerpt from source")
    loop_index: int = Field(default=0, description="Loop iteration when captured")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when source was recorded",
    )
    query: str = Field(default="", description="Search query that found this source")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class DepthMetrics(BaseModel):
    """Depth-tier specific metrics captured during research.

    These metrics are populated based on the research depth tier:
    - sources_filtered_quality: Number of sources filtered by quality floor
    - sources_filtered_diversity: Number of sources filtered for diversity
    - unique_domains: Count of unique domains across all sources
    - contradictions_found: Number of contradictions detected (Alpha only)
    - convergence_loop: Loop index when convergence was detected (if any)
    """

    sources_filtered_quality: int = Field(
        default=0,
        description="Sources filtered out due to quality floor",
    )
    sources_filtered_diversity: int = Field(
        default=0,
        description="Sources filtered out for domain diversity enforcement",
    )
    unique_domains: int = Field(
        default=0,
        description="Count of unique domains across collected sources",
    )
    contradictions_found: int = Field(
        default=0,
        description="Number of contradictions detected (Alpha tier only)",
    )
    convergence_loop: Optional[int] = Field(
        default=None,
        description="Loop index when convergence was triggered (None if no convergence)",
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class LoopTelemetry(BaseModel):
    """Telemetry for a single research loop iteration."""

    loop_index: int = Field(description="Loop iteration number (0-indexed)")
    start_time: str = Field(description="ISO timestamp when loop started")
    end_time: str = Field(default="", description="ISO timestamp when loop ended")
    duration_seconds: float = Field(default=0.0, description="Loop duration in seconds")
    api_calls: Dict[str, int] = Field(
        default_factory=dict,
        description="API calls made during this loop",
    )
    token_usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage during this loop",
    )
    findings_count: int = Field(default=0, description="Number of findings captured")
    sources_count: int = Field(default=0, description="Number of sources gathered")
    raw_findings: int = Field(
        default=0,
        description="Cumulative raw findings recorded after this loop",
    )
    unique_concepts: int = Field(
        default=0,
        description="Cumulative unique concepts represented after this loop",
    )
    cluster_count: int = Field(
        default=0,
        description="Cumulative concept cluster count after this loop",
    )
    degraded: bool = Field(default=False, description="Whether loop ran in degraded mode")
    errors: List[str] = Field(default_factory=list, description="Error messages if any")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class MissionTelemetry(BaseModel):
    """Complete telemetry for a research mission.

    This schema is versioned for TraceLab ingestion compatibility.
    """

    schema_version: str = Field(default=TELEMETRY_SCHEMA_VERSION)
    mission_id: str = Field(description="Mission identifier from YAML")
    project_id: Optional[str] = Field(default=None, description="TraceLab project ID")
    timestamp_start: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Mission start timestamp",
    )
    timestamp_end: str = Field(default="", description="Mission end timestamp")
    duration_seconds: float = Field(default=0.0, description="Total mission duration")
    loop_count: int = Field(default=0, description="Number of research loops executed")
    api_calls: Dict[str, int] = Field(
        default_factory=dict,
        description="Aggregate API calls across all loops",
    )
    token_usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Aggregate token usage",
    )
    estimated_cost_usd: float = Field(default=0.0, description="Estimated LLM cost")
    degradation_flags: List[str] = Field(
        default_factory=list,
        description="Active degradation policies",
    )
    loops: List[LoopTelemetry] = Field(
        default_factory=list,
        description="Per-loop telemetry",
    )
    findings: List[Finding] = Field(
        default_factory=list,
        description="Structured findings with citations",
    )
    sources: List[SourceEvidence] = Field(
        default_factory=list,
        description="All gathered sources",
    )
    early_termination: bool = Field(
        default=False,
        description="Whether mission terminated early",
    )
    termination_reason: str = Field(default="", description="Reason for early termination")
    research_depth: str = Field(
        default="baseline",
        description="Research depth tier (baseline/deep/alpha)",
    )
    contract_id: str = Field(default="", description="Compiled execution contract identity")
    contract_path: str = Field(default="", description="Filesystem path to the compiled contract")
    contract_version: str = Field(default="", description="Compiled contract schema version")
    contract_origin: str = Field(default="", description="Source used to compile the contract")
    runtime_identity: Dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime identity for the code path that executed the run",
    )
    config_fingerprint: str = Field(
        default="",
        description="Deterministic fingerprint of execution-relevant settings",
    )
    config_snapshot: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sanitized execution settings used to compute the fingerprint",
    )
    llm_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-call prompt, context, and token summaries",
    )
    source_counts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate source counts before and after filtering",
    )
    source_filter_runs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-loop source filtering counts",
    )
    finding_counts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw versus deduplicated finding counts",
    )
    routing_trace: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Judge decision and actual routing reasons by loop",
    )
    coverage_gate_trace: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Coverage gate pass/fail trace recorded by reflection routing",
    )
    objective_score_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-loop objective scoring history from reflection",
    )
    convergence_history: List[float] = Field(
        default_factory=list,
        description="Overall coverage score history used for convergence checks",
    )
    query_telemetry: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-query retrieval telemetry across active search lanes",
    )
    token_budget_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pre-flight prompt budgeting events emitted by research/reflection nodes",
    )
    final_outcome: Dict[str, Any] = Field(
        default_factory=dict,
        description="Final validation or completion outcome for the run",
    )
    validation_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validation status and failing checks when validation runs",
    )
    depth_metrics: DepthMetrics = Field(
        default_factory=DepthMetrics,
        description="Depth-tier specific metrics",
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


@dataclass
class TelemetryCollector:
    """Collects telemetry during mission execution.

    Use as a context manager or manually call start()/stop().
    Thread-safe for single mission execution.
    """

    mission_id: str
    project_id: Optional[str] = None
    research_depth: str = "baseline"
    _telemetry: MissionTelemetry = field(init=False)
    _current_loop: Optional[LoopTelemetry] = field(default=None, init=False)
    _loop_start_time: float = field(default=0.0, init=False)
    _start_time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._telemetry = MissionTelemetry(
            mission_id=self.mission_id,
            project_id=self.project_id,
            research_depth=self.research_depth,
        )
        self._start_time = time.perf_counter()

    def start_loop(self, loop_index: int) -> None:
        """Mark the beginning of a research loop."""
        self._loop_start_time = time.perf_counter()
        self._current_loop = LoopTelemetry(
            loop_index=loop_index,
            start_time=datetime.now(timezone.utc).isoformat(),
        )

    def end_loop(
        self,
        *,
        api_calls: Optional[Dict[str, int]] = None,
        token_usage: Optional[Dict[str, int]] = None,
        findings_count: int = 0,
        sources_count: int = 0,
        raw_findings: int = 0,
        unique_concepts: int = 0,
        cluster_count: int = 0,
        degraded: bool = False,
        errors: Optional[List[str]] = None,
    ) -> None:
        """Mark the end of a research loop with its metrics."""
        if self._current_loop is None:
            return

        duration = time.perf_counter() - self._loop_start_time
        self._current_loop.end_time = datetime.now(timezone.utc).isoformat()
        self._current_loop.duration_seconds = round(duration, 3)
        self._current_loop.api_calls = dict(api_calls or {})
        self._current_loop.token_usage = dict(token_usage or {})
        self._current_loop.findings_count = findings_count
        self._current_loop.sources_count = sources_count
        self._current_loop.raw_findings = raw_findings
        self._current_loop.unique_concepts = min(max(unique_concepts, 0), max(raw_findings, 0))
        self._current_loop.cluster_count = max(cluster_count, 0)
        self._current_loop.degraded = degraded
        self._current_loop.errors = list(errors or [])

        self._telemetry.loops.append(self._current_loop)
        self._telemetry.loop_count = len(self._telemetry.loops)
        dedup_ratio = (
            round(
                (
                    self._current_loop.raw_findings
                    - self._current_loop.unique_concepts
                )
                / self._current_loop.raw_findings,
                4,
            )
            if self._current_loop.raw_findings
            else 0.0
        )
        self._telemetry.finding_counts = {
            "raw_findings": self._current_loop.raw_findings,
            "unique_concepts": self._current_loop.unique_concepts,
            "cluster_count": self._current_loop.cluster_count,
            "deduped_findings": self._current_loop.unique_concepts,
            "dedup_ratio": dedup_ratio,
            "loops": [
                {
                    "loop_index": loop.loop_index,
                    "raw_findings": loop.raw_findings,
                    "unique_concepts": loop.unique_concepts,
                    "cluster_count": loop.cluster_count,
                    "deduped_findings": loop.unique_concepts,
                    "dedup_ratio": (
                        round(
                            (loop.raw_findings - loop.unique_concepts) / loop.raw_findings,
                            4,
                        )
                        if loop.raw_findings
                        else 0.0
                    ),
                }
                for loop in self._telemetry.loops
            ],
        }

        # Aggregate API calls
        for key, value in (api_calls or {}).items():
            self._telemetry.api_calls[key] = self._telemetry.api_calls.get(key, 0) + value

        # Aggregate token usage
        for key, value in (token_usage or {}).items():
            self._telemetry.token_usage[key] = self._telemetry.token_usage.get(key, 0) + value

        self._current_loop = None

    def record_finding(
        self,
        objective_id: str = "",
        summary: str = "",
        citations: Optional[List[str]] = None,
        confidence: float = 0.0,
        *,
        objective_ids: Optional[List[str]] = None,
        contract_slot_tags: Optional[List[str]] = None,
        source_urls: Optional[List[str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        concept_id: Optional[str] = None,
        finding_id: Optional[str] = None,
        loop_index: Optional[int] = None,
    ) -> Finding:
        """Record a structured finding with citations."""

        normalized_loop = loop_index if loop_index is not None else len(self._telemetry.loops)
        provenance_payload = (
            dict(provenance) if isinstance(provenance, Mapping) else {}
        )
        if provenance_payload.get("loop") in (None, ""):
            provenance_payload["loop"] = normalized_loop
        objective_id_list = list(objective_ids or [])
        if objective_id and not objective_id_list:
            objective_id_list = [objective_id]

        finding = Finding(
            summary=summary,
            citations=list(citations or []),
            objective_ids=objective_id_list,
            contract_slot_tags=list(contract_slot_tags or []),
            source_urls=list(source_urls or []),
            provenance=FindingProvenance.model_validate(provenance_payload),
            confidence=confidence,
            concept_id=concept_id,
            finding_id=finding_id or build_finding_id(summary, normalized_loop),
            loop_index=normalized_loop,
        )
        self._telemetry.findings.append(finding)
        return finding

    def record_source(
        self,
        url: str,
        title: str = "",
        snippet: str = "",
        query: str = "",
    ) -> SourceEvidence:
        """Record a gathered source."""
        loop_index = len(self._telemetry.loops)
        source = SourceEvidence(
            url=url,
            title=title,
            snippet=snippet,
            loop_index=loop_index,
            query=query,
        )
        self._telemetry.sources.append(source)
        return source

    def set_degradation(self, flags: List[str]) -> None:
        """Set active degradation policies."""
        self._telemetry.degradation_flags = list(flags)

    def set_termination(self, early: bool, reason: str = "") -> None:
        """Set early termination status."""
        self._telemetry.early_termination = early
        self._telemetry.termination_reason = reason

    def set_cost(self, cost_usd: float) -> None:
        """Set estimated LLM cost."""
        self._telemetry.estimated_cost_usd = round(cost_usd, 6)

    def set_contract(
        self,
        *,
        contract_id: str,
        contract_path: str,
        contract_version: str,
        contract_origin: str,
    ) -> None:
        """Attach compiled contract metadata to the telemetry payload."""
        self._telemetry.contract_id = contract_id
        self._telemetry.contract_path = contract_path
        self._telemetry.contract_version = contract_version
        self._telemetry.contract_origin = contract_origin

    def merge_state_telemetry(self, telemetry: Mapping[str, Any]) -> None:
        """Copy state-level observability details into the persisted artifact."""

        runtime_identity = telemetry.get("runtime_identity")
        if isinstance(runtime_identity, Mapping):
            self._telemetry.runtime_identity = dict(runtime_identity)

        config_snapshot = telemetry.get("config_snapshot")
        if isinstance(config_snapshot, Mapping):
            self._telemetry.config_snapshot = dict(config_snapshot)

        config_fingerprint = str(telemetry.get("config_fingerprint") or "").strip()
        if config_fingerprint:
            self._telemetry.config_fingerprint = config_fingerprint

        llm_calls = telemetry.get("llm_calls")
        if isinstance(llm_calls, list):
            self._telemetry.llm_calls = [
                dict(item) for item in llm_calls if isinstance(item, Mapping)
            ]

        source_counts = telemetry.get("source_counts")
        if isinstance(source_counts, Mapping):
            self._telemetry.source_counts = dict(source_counts)

        source_filter_runs = telemetry.get("source_filter_runs")
        if isinstance(source_filter_runs, list):
            self._telemetry.source_filter_runs = [
                dict(item) for item in source_filter_runs if isinstance(item, Mapping)
            ]

        finding_counts = telemetry.get("finding_counts")
        if isinstance(finding_counts, Mapping):
            self._telemetry.finding_counts = dict(finding_counts)
            if not self._telemetry.loops:
                loop_counts = finding_counts.get("loops")
                if isinstance(loop_counts, list):
                    synthesized_loops: List[LoopTelemetry] = []
                    default_time = self._telemetry.timestamp_start
                    for item in loop_counts:
                        if not isinstance(item, Mapping):
                            continue
                        synthesized_loops.append(
                            LoopTelemetry(
                                loop_index=int(item.get("loop_index", len(synthesized_loops)) or 0),
                                start_time=default_time,
                                end_time=default_time,
                                findings_count=int(item.get("raw_findings", 0) or 0),
                                raw_findings=int(item.get("raw_findings", 0) or 0),
                                unique_concepts=int(
                                    item.get(
                                        "unique_concepts",
                                        item.get("deduped_findings", 0),
                                    )
                                    or 0
                                ),
                                cluster_count=int(
                                    item.get(
                                        "cluster_count",
                                        item.get(
                                            "unique_concepts",
                                            item.get("deduped_findings", 0),
                                        ),
                                    )
                                    or 0
                                ),
                            )
                        )
                    if synthesized_loops:
                        self._telemetry.loops = synthesized_loops
                        self._telemetry.loop_count = len(synthesized_loops)

        routing_trace = telemetry.get("routing_trace")
        if isinstance(routing_trace, list):
            self._telemetry.routing_trace = [
                dict(item) for item in routing_trace if isinstance(item, Mapping)
            ]

        coverage_gate_trace = telemetry.get("coverage_gate_trace")
        if isinstance(coverage_gate_trace, list):
            self._telemetry.coverage_gate_trace = [
                dict(item) for item in coverage_gate_trace if isinstance(item, Mapping)
            ]

        objective_score_history = telemetry.get("objective_score_history")
        if isinstance(objective_score_history, list):
            self._telemetry.objective_score_history = [
                dict(item) for item in objective_score_history if isinstance(item, Mapping)
            ]

        convergence_history = telemetry.get("convergence_history")
        if isinstance(convergence_history, list):
            normalized_history: List[float] = []
            for item in convergence_history:
                try:
                    normalized_history.append(float(item))
                except (TypeError, ValueError):
                    continue
            self._telemetry.convergence_history = normalized_history

        query_telemetry = telemetry.get("query_telemetry")
        if isinstance(query_telemetry, list):
            self._telemetry.query_telemetry = [
                dict(item) for item in query_telemetry if isinstance(item, Mapping)
            ]

        token_budget_events = telemetry.get("token_budget_events")
        if isinstance(token_budget_events, list):
            self._telemetry.token_budget_events = [
                dict(item) for item in token_budget_events if isinstance(item, Mapping)
            ]

        final_outcome = telemetry.get("final_outcome")
        if isinstance(final_outcome, Mapping):
            self._telemetry.final_outcome = dict(final_outcome)

        validation_summary = telemetry.get("validation_summary")
        if isinstance(validation_summary, Mapping):
            self._telemetry.validation_summary = dict(validation_summary)

    def set_depth_metrics(
        self,
        *,
        sources_filtered_quality: Optional[int] = None,
        sources_filtered_diversity: Optional[int] = None,
        unique_domains: Optional[int] = None,
        contradictions_found: Optional[int] = None,
        convergence_loop: Optional[int] = None,
    ) -> None:
        """Set depth-tier specific metrics.

        Only updates fields that are explicitly provided (not None).
        This allows incremental updates as metrics become available.

        Args:
            sources_filtered_quality: Number of sources filtered by quality floor.
            sources_filtered_diversity: Number of sources filtered for diversity.
            unique_domains: Count of unique domains across all sources.
            contradictions_found: Number of contradictions detected (Alpha only).
            convergence_loop: Loop index when convergence was triggered.
        """
        metrics = self._telemetry.depth_metrics
        if sources_filtered_quality is not None:
            metrics.sources_filtered_quality = sources_filtered_quality
        if sources_filtered_diversity is not None:
            metrics.sources_filtered_diversity = sources_filtered_diversity
        if unique_domains is not None:
            metrics.unique_domains = unique_domains
        if contradictions_found is not None:
            metrics.contradictions_found = contradictions_found
        if convergence_loop is not None:
            metrics.convergence_loop = convergence_loop

    def increment_filtered_quality(self, count: int = 1) -> None:
        """Increment the count of sources filtered by quality floor.

        Use when processing source filtering incrementally across loops.

        Args:
            count: Number to add to the filtered quality count.
        """
        self._telemetry.depth_metrics.sources_filtered_quality += count

    def increment_filtered_diversity(self, count: int = 1) -> None:
        """Increment the count of sources filtered for diversity.

        Use when processing source filtering incrementally across loops.

        Args:
            count: Number to add to the filtered diversity count.
        """
        self._telemetry.depth_metrics.sources_filtered_diversity += count

    def set_convergence_loop(self, loop_index: int) -> None:
        """Mark the loop where convergence was detected.

        Args:
            loop_index: The loop iteration (0-indexed) where convergence triggered.
        """
        self._telemetry.depth_metrics.convergence_loop = loop_index

    def get_depth_metrics(self) -> DepthMetrics:
        """Get current depth metrics."""
        return self._telemetry.depth_metrics

    def finalize(self) -> MissionTelemetry:
        """Finalize and return the complete telemetry."""
        self._telemetry.timestamp_end = datetime.now(timezone.utc).isoformat()
        self._telemetry.duration_seconds = round(time.perf_counter() - self._start_time, 3)
        return self._telemetry

    def get_telemetry(self) -> MissionTelemetry:
        """Get current telemetry state (may not be finalized)."""
        return self._telemetry


class TelemetryWriter:
    """Async-safe writer for persisting telemetry to JSONL files.

    Telemetry is written to checkpoints/telemetry/mission-<id>.jsonl
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
    ) -> None:
        override = os.getenv(TELEMETRY_DIR_ENV)
        if output_dir is not None:
            self._output_dir = Path(output_dir)
        elif override:
            self._output_dir = Path(override).expanduser().resolve()
        else:
            self._output_dir = data_path("telemetry")

    def _ensure_dir(self) -> None:
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Unable to create telemetry directory: %s", exc)

    def _get_path(self, mission_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in mission_id)
        return self._output_dir / f"mission-{safe_id}.jsonl"

    def write(self, telemetry: MissionTelemetry) -> Optional[str]:
        """Write telemetry to JSONL file. Returns path on success."""
        self._ensure_dir()
        path = self._get_path(telemetry.mission_id)

        record = telemetry.to_dict()
        record["_written_at"] = datetime.now(timezone.utc).isoformat()

        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("Telemetry written to %s", path)
            return str(path)
        except OSError as exc:
            logger.error("Failed to write telemetry: %s", exc)
            return None

    def read_all(self, mission_id: str) -> List[MissionTelemetry]:
        """Read all telemetry entries for a mission."""
        path = self._get_path(mission_id)
        if not path.exists():
            return []

        entries: List[MissionTelemetry] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entries.append(MissionTelemetry.model_validate(data))
                    except (json.JSONDecodeError, Exception) as exc:
                        logger.warning("Skipping malformed telemetry line: %s", exc)
        except OSError as exc:
            logger.error("Failed to read telemetry file: %s", exc)

        return entries


# Module-level convenience functions


def record_finding(
    collector: TelemetryCollector,
    objective_id: str = "",
    summary: str = "",
    citations: Optional[List[str]] = None,
    confidence: float = 0.0,
    *,
    objective_ids: Optional[List[str]] = None,
    contract_slot_tags: Optional[List[str]] = None,
    source_urls: Optional[List[str]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    concept_id: Optional[str] = None,
    finding_id: Optional[str] = None,
    loop_index: Optional[int] = None,
) -> Finding:
    """Record a structured finding to the collector.

    Args:
        collector: Active telemetry collector.
        objective_id: Legacy single objective identifier for this finding.
        summary: Concise summary of the finding.
        citations: URLs or source references.
        confidence: Confidence score from reflection (0-1).

    Returns:
        The recorded Finding instance.
    """
    return collector.record_finding(
        objective_id=objective_id,
        summary=summary,
        citations=citations,
        confidence=confidence,
        objective_ids=objective_ids,
        contract_slot_tags=contract_slot_tags,
        source_urls=source_urls,
        provenance=provenance,
        concept_id=concept_id,
        finding_id=finding_id,
        loop_index=loop_index,
    )


def record_source(
    collector: TelemetryCollector,
    metadata: Mapping[str, Any],
) -> SourceEvidence:
    """Record a source from a metadata dictionary.

    Args:
        collector: Active telemetry collector.
        metadata: Source metadata with url, title, snippet, query fields.

    Returns:
        The recorded SourceEvidence instance.
    """
    return collector.record_source(
        url=str(metadata.get("url", "")),
        title=str(metadata.get("title", "")),
        snippet=str(metadata.get("snippet", metadata.get("content", ""))),
        query=str(metadata.get("query", "")),
    )


@contextmanager
def telemetry_context(
    mission_id: str,
    project_id: Optional[str] = None,
    *,
    research_depth: str = "baseline",
    persist: bool = True,
    output_dir: Optional[Path] = None,
) -> Iterator[TelemetryCollector]:
    """Context manager for collecting and persisting telemetry.

    Usage:
        with telemetry_context("DRM.0.1", research_depth="alpha") as collector:
            collector.start_loop(0)
            # ... do research ...
            collector.end_loop(api_calls={"tavily": 1})
            collector.record_finding("obj-1", "Key insight", ["https://..."])
            collector.set_depth_metrics(unique_domains=5, contradictions_found=1)
        # Telemetry is auto-persisted on exit

    Args:
        mission_id: Mission identifier.
        project_id: Optional TraceLab project ID.
        research_depth: Research depth tier (baseline/deep/alpha).
        persist: Whether to write to JSONL on exit (default True).
        output_dir: Override telemetry output directory.

    Yields:
        TelemetryCollector instance for recording metrics.
    """
    collector = TelemetryCollector(
        mission_id=mission_id,
        project_id=project_id,
        research_depth=research_depth,
    )
    try:
        yield collector
    finally:
        telemetry = collector.finalize()
        if persist:
            writer = TelemetryWriter(output_dir=output_dir)
            writer.write(telemetry)


def record_finding_records_from_state(
    state: Mapping[str, Any],
    collector: TelemetryCollector,
) -> List[Finding]:
    """Record structured finding records from AgentState into telemetry.

    Args:
        state: AgentState dictionary.
        collector: Active telemetry collector.

    Returns:
        List of recorded Finding instances.
    """
    finding_records = coerce_state_finding_records(state)
    recorded: List[Finding] = []
    for finding_record in finding_records:
        recorded.append(
            collector.record_finding(
                summary=finding_record.summary,
                citations=list(finding_record.citations),
                confidence=float(finding_record.confidence),
                objective_ids=list(finding_record.objective_ids),
                contract_slot_tags=list(finding_record.contract_slot_tags),
                source_urls=list(finding_record.source_urls),
                provenance=finding_record.provenance.to_dict(),
                concept_id=finding_record.concept_id,
                finding_id=finding_record.finding_id,
                loop_index=finding_record.loop_index,
            )
        )

    return recorded


def extract_sources_from_state(
    state: Mapping[str, Any],
    collector: TelemetryCollector,
) -> List[SourceEvidence]:
    """Extract sources from AgentState and record to collector.

    Args:
        state: AgentState dictionary.
        collector: Active telemetry collector.

    Returns:
        List of recorded SourceEvidence instances.
    """
    # S83 m06 (PD83-b) CARVE-OUT: like report/evidence.py:generate_evidence_list,
    # extract_sources_from_state is a tested, exported generic source-extraction
    # utility. Kept reading the (dead-in-production) sources_found rather than
    # gutted; excluded from the never-reappears guards.
    sources_raw = state.get("sources_found") or []
    recorded: List[SourceEvidence] = []

    for source in sources_raw:
        if not isinstance(source, dict):
            continue
        url = source.get("url", "")
        if not url:
            continue
        recorded.append(record_source(collector, source))

    return recorded


__all__ = [
    "DepthMetrics",
    "Finding",
    "LoopTelemetry",
    "MissionTelemetry",
    "SourceEvidence",
    "TelemetryCollector",
    "TelemetryWriter",
    "TELEMETRY_SCHEMA_VERSION",
    "extract_sources_from_state",
    "record_finding",
    "record_finding_records_from_state",
    "record_source",
    "telemetry_context",
]
