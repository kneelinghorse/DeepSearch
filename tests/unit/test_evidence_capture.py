"""Unit tests for evidence capture and telemetry instrumentation."""

from __future__ import annotations

import json
from pathlib import Path

from deepsearch.agent.evidence import (
    TELEMETRY_SCHEMA_VERSION,
    Finding,
    LoopTelemetry,
    MissionTelemetry,
    SourceEvidence,
    TelemetryCollector,
    TelemetryWriter,
    extract_sources_from_state,
    record_finding,
    record_finding_records_from_state,
    record_source,
    telemetry_context,
)
from deepsearch.agent.state import AgentState


class TestFindingModel:
    """Tests for the Finding Pydantic model."""

    def test_finding_default_values(self) -> None:
        """Finding should have sensible defaults."""
        finding = Finding(objective_ids=["obj-1"], summary="Test finding")
        assert finding.objective_ids == ["obj-1"]
        assert finding.summary == "Test finding"
        assert finding.citations == []
        assert finding.loop_index == 0
        assert finding.confidence == 0.0
        assert finding.timestamp  # Should be auto-generated

    def test_finding_with_citations(self) -> None:
        """Finding should accept citations list."""
        finding = Finding(
            objective_ids=["obj-2"],
            summary="Key insight discovered",
            citations=["https://example.com/source1", "https://example.com/source2"],
            confidence=0.85,
            contract_slot_tags=["slot-objective-0"],
            source_urls=["https://example.com/source1"],
        )
        assert len(finding.citations) == 2
        assert finding.confidence == 0.85

    def test_finding_to_dict(self) -> None:
        """Finding.to_dict should return serializable dict."""
        finding = Finding(
            objective_ids=["obj-3"],
            summary="Serializable finding",
            loop_index=2,
            provenance={"loop": 2},
        )
        data = finding.to_dict()
        assert data["objective_ids"] == ["obj-3"]
        assert data["loop_index"] == 2
        assert isinstance(data, dict)


class TestSourceEvidence:
    """Tests for the SourceEvidence model."""

    def test_source_evidence_creation(self) -> None:
        """SourceEvidence should capture URL and metadata."""
        source = SourceEvidence(
            url="https://example.com/article",
            title="Research Article",
            snippet="Key information about the topic...",
            query="research topic query",
        )
        assert source.url == "https://example.com/article"
        assert source.title == "Research Article"
        assert source.snippet.startswith("Key")
        assert source.query == "research topic query"

    def test_source_evidence_defaults(self) -> None:
        """SourceEvidence should have sensible defaults."""
        source = SourceEvidence(url="https://example.com")
        assert source.title == ""
        assert source.snippet == ""
        assert source.loop_index == 0


class TestLoopTelemetry:
    """Tests for per-loop telemetry."""

    def test_loop_telemetry_creation(self) -> None:
        """LoopTelemetry should capture loop metrics."""
        loop = LoopTelemetry(
            loop_index=0,
            start_time="2025-12-04T10:00:00Z",
            end_time="2025-12-04T10:05:00Z",
            duration_seconds=300.0,
            api_calls={"tavily_web_search": 1, "llm_research": 1},
            token_usage={"prompt_tokens": 500, "completion_tokens": 200},
            findings_count=3,
            sources_count=5,
            raw_findings=5,
            unique_concepts=3,
            cluster_count=3,
        )
        assert loop.loop_index == 0
        assert loop.duration_seconds == 300.0
        assert loop.api_calls["tavily_web_search"] == 1
        assert loop.findings_count == 3
        assert loop.raw_findings == 5
        assert loop.unique_concepts == 3
        assert loop.cluster_count == 3


class TestMissionTelemetry:
    """Tests for complete mission telemetry."""

    def test_mission_telemetry_schema_version(self) -> None:
        """MissionTelemetry should include schema version."""
        telemetry = MissionTelemetry(mission_id="TEST.0.1")
        assert telemetry.schema_version == TELEMETRY_SCHEMA_VERSION
        assert telemetry.mission_id == "TEST.0.1"

    def test_mission_telemetry_aggregations(self) -> None:
        """MissionTelemetry should hold aggregate metrics."""
        telemetry = MissionTelemetry(
            mission_id="TEST.0.2",
            loop_count=3,
            api_calls={"tavily_web_search": 3, "llm_research": 3, "llm_reflection": 15},
            token_usage={"prompt_tokens": 1500, "completion_tokens": 600},
            estimated_cost_usd=0.0025,
        )
        assert telemetry.loop_count == 3
        assert telemetry.api_calls["llm_reflection"] == 15
        assert telemetry.estimated_cost_usd == 0.0025

    def test_mission_telemetry_serialization(self) -> None:
        """MissionTelemetry.to_dict should be JSON serializable."""
        telemetry = MissionTelemetry(
            mission_id="TEST.0.3",
            findings=[Finding(objective_ids=["obj-1"], summary="Test")],
            sources=[SourceEvidence(url="https://example.com")],
        )
        data = telemetry.to_dict()
        # Should be JSON serializable
        json_str = json.dumps(data)
        assert "TEST.0.3" in json_str
        assert "obj-1" in json_str


class TestTelemetryCollector:
    """Tests for the TelemetryCollector class."""

    def test_collector_initialization(self) -> None:
        """TelemetryCollector should initialize with mission metadata."""
        collector = TelemetryCollector(mission_id="TEST.0.4", project_id="proj-123")
        telemetry = collector.get_telemetry()
        assert telemetry.mission_id == "TEST.0.4"
        assert telemetry.project_id == "proj-123"

    def test_collector_loop_lifecycle(self) -> None:
        """Collector should track loop start/end with metrics."""
        collector = TelemetryCollector(mission_id="TEST.0.5")

        collector.start_loop(0)
        collector.end_loop(
            api_calls={"tavily_web_search": 1},
            token_usage={"prompt_tokens": 100},
            findings_count=2,
            sources_count=3,
            raw_findings=3,
            unique_concepts=2,
            cluster_count=2,
        )

        telemetry = collector.get_telemetry()
        assert telemetry.loop_count == 1
        assert len(telemetry.loops) == 1
        assert telemetry.loops[0].findings_count == 2
        assert telemetry.loops[0].raw_findings == 3
        assert telemetry.loops[0].unique_concepts == 2
        assert telemetry.finding_counts["raw_findings"] == 3
        assert telemetry.finding_counts["unique_concepts"] == 2
        assert telemetry.api_calls["tavily_web_search"] == 1

    def test_collector_multiple_loops_aggregate(self) -> None:
        """Collector should aggregate metrics across loops."""
        collector = TelemetryCollector(mission_id="TEST.0.6")

        for i in range(3):
            collector.start_loop(i)
            collector.end_loop(
                api_calls={"llm_research": 1},
                token_usage={"prompt_tokens": 100, "completion_tokens": 50},
            )

        telemetry = collector.get_telemetry()
        assert telemetry.loop_count == 3
        assert telemetry.api_calls["llm_research"] == 3
        assert telemetry.token_usage["prompt_tokens"] == 300

    def test_collector_record_finding(self) -> None:
        """Collector should record structured findings."""
        collector = TelemetryCollector(mission_id="TEST.0.7")

        finding = collector.record_finding(
            summary="Key insight about topic",
            citations=["https://example.com/ref1"],
            confidence=0.9,
            objective_ids=["objective-1"],
            contract_slot_tags=["slot-objective-1"],
            source_urls=["https://example.com/ref1"],
            provenance={"lane": "google", "query": "topic", "loop": 0},
        )

        assert finding.objective_ids == ["objective-1"]
        assert finding.confidence == 0.9
        telemetry = collector.get_telemetry()
        assert len(telemetry.findings) == 1

    def test_collector_record_source(self) -> None:
        """Collector should record source evidence."""
        collector = TelemetryCollector(mission_id="TEST.0.8")

        source = collector.record_source(
            url="https://example.com/article",
            title="Research Article",
            snippet="Relevant excerpt...",
        )

        assert source.url == "https://example.com/article"
        telemetry = collector.get_telemetry()
        assert len(telemetry.sources) == 1

    def test_collector_degradation_flags(self) -> None:
        """Collector should track degradation status."""
        collector = TelemetryCollector(mission_id="TEST.0.9")
        collector.set_degradation(["skip_web_search", "conservative_llm"])

        telemetry = collector.get_telemetry()
        assert "skip_web_search" in telemetry.degradation_flags

    def test_collector_early_termination(self) -> None:
        """Collector should track early termination."""
        collector = TelemetryCollector(mission_id="TEST.1.0")
        collector.set_termination(early=True, reason="convergence_detected")

        telemetry = collector.get_telemetry()
        assert telemetry.early_termination is True
        assert telemetry.termination_reason == "convergence_detected"

    def test_collector_finalize(self) -> None:
        """Finalize should set end timestamp and duration."""
        collector = TelemetryCollector(mission_id="TEST.1.1")
        collector.start_loop(0)
        collector.end_loop()

        telemetry = collector.finalize()

        assert telemetry.timestamp_end != ""
        assert telemetry.duration_seconds >= 0.0

    def test_collector_merges_state_telemetry_summary(self) -> None:
        """Collector should carry runtime and routing telemetry into JSONL artifacts."""
        collector = TelemetryCollector(mission_id="TEST.1.1B")
        collector.merge_state_telemetry(
            {
                "runtime_identity": {"execution_path": "yaml", "build_hash": "abc123"},
                "config_fingerprint": "fp-123",
                "config_snapshot": {"search_backend": "linkup"},
                "llm_calls": [{"call_type": "extraction", "system_prompt_chars": 42}],
                "source_counts": {"sources_before_filtering": 6},
                "source_filter_runs": [{"loop_index": 0, "sources_before_filtering": 6}],
                "finding_counts": {
                    "raw_findings": 5,
                    "unique_concepts": 3,
                    "cluster_count": 3,
                    "deduped_findings": 3,
                    "loops": [
                        {
                            "loop_index": 0,
                            "raw_findings": 2,
                            "unique_concepts": 1,
                            "cluster_count": 1,
                            "deduped_findings": 1,
                        },
                        {
                            "loop_index": 1,
                            "raw_findings": 5,
                            "unique_concepts": 3,
                            "cluster_count": 3,
                            "deduped_findings": 3,
                        },
                    ],
                },
                "routing_trace": [{"judge_decision": "finalize", "next_action": "continue"}],
                "coverage_gate_trace": [{"loop": 1, "passed": False, "failed_gates": ["slot"]}],
                "objective_score_history": [
                    {
                        "loop": 1,
                        "scores": [{"objective_id": "objective-0", "score": 0.65}],
                    }
                ],
                "convergence_history": [0.55, 0.65],
                "final_outcome": {"outcome": "complete"},
                "validation_summary": {"requested": False, "status": "not_requested"},
            }
        )

        telemetry = collector.finalize()

        assert telemetry.runtime_identity["execution_path"] == "yaml"
        assert telemetry.config_fingerprint == "fp-123"
        assert telemetry.config_snapshot["search_backend"] == "linkup"
        assert telemetry.llm_calls[0]["system_prompt_chars"] == 42
        assert telemetry.source_counts["sources_before_filtering"] == 6
        assert telemetry.finding_counts["unique_concepts"] == 3
        assert telemetry.finding_counts["deduped_findings"] == 3
        assert telemetry.loops[0].raw_findings == 2
        assert telemetry.loops[0].unique_concepts == 1
        assert telemetry.loops[1].raw_findings == 5
        assert telemetry.loops[1].unique_concepts == 3
        assert telemetry.routing_trace[0]["next_action"] == "continue"
        assert telemetry.coverage_gate_trace[0]["failed_gates"] == ["slot"]
        assert telemetry.objective_score_history[0]["scores"][0]["score"] == 0.65
        assert telemetry.convergence_history == [0.55, 0.65]
        assert telemetry.final_outcome["outcome"] == "complete"

    def test_collector_preserves_raw_vs_unique_invariant_across_loops(self) -> None:
        """Every loop snapshot should keep raw finding counts above unique concepts."""
        collector = TelemetryCollector(mission_id="TEST.1.1C")

        collector.start_loop(0)
        collector.end_loop(raw_findings=3, unique_concepts=2, cluster_count=2)
        collector.start_loop(1)
        collector.end_loop(raw_findings=6, unique_concepts=3, cluster_count=3)

        telemetry = collector.finalize()

        assert telemetry.finding_counts["raw_findings"] == 6
        assert telemetry.finding_counts["unique_concepts"] == 3
        assert all(loop.raw_findings >= loop.unique_concepts for loop in telemetry.loops)


class TestTelemetryWriter:
    """Tests for JSONL persistence."""

    def test_writer_creates_directory(self, tmp_path: Path) -> None:
        """Writer should create telemetry directory if missing."""
        output_dir = tmp_path / "telemetry"
        writer = TelemetryWriter(output_dir=output_dir)

        telemetry = MissionTelemetry(mission_id="TEST.1.2")
        path = writer.write(telemetry)

        assert path is not None
        assert output_dir.exists()

    def test_writer_appends_jsonl(self, tmp_path: Path) -> None:
        """Writer should append to JSONL file."""
        writer = TelemetryWriter(output_dir=tmp_path)

        # Write two entries
        telemetry1 = MissionTelemetry(mission_id="TEST.1.3", loop_count=1)
        telemetry2 = MissionTelemetry(mission_id="TEST.1.3", loop_count=2)

        writer.write(telemetry1)
        writer.write(telemetry2)

        # Read back
        entries = writer.read_all("TEST.1.3")
        assert len(entries) == 2
        assert entries[0].loop_count == 1
        assert entries[1].loop_count == 2

    def test_writer_sanitizes_mission_id(self, tmp_path: Path) -> None:
        """Writer should sanitize mission ID for filename safety."""
        writer = TelemetryWriter(output_dir=tmp_path)

        telemetry = MissionTelemetry(mission_id="TEST/0.1:dangerous")
        path = writer.write(telemetry)

        assert path is not None
        assert "/" not in Path(path).name
        assert ":" not in Path(path).name


class TestTelemetryContextManager:
    """Tests for the telemetry_context helper."""

    def test_context_manager_basic_flow(self, tmp_path: Path) -> None:
        """Context manager should collect and persist telemetry."""
        with telemetry_context(
            "TEST.1.4",
            persist=True,
            output_dir=tmp_path,
        ) as collector:
            collector.start_loop(0)
            collector.record_finding(summary="Test finding", objective_ids=["obj-1"])
            collector.end_loop(api_calls={"llm_research": 1})

        # Verify file was created
        files = list(tmp_path.glob("mission-*.jsonl"))
        assert len(files) == 1

    def test_context_manager_no_persist(self, tmp_path: Path) -> None:
        """Context manager should skip persistence when disabled."""
        with telemetry_context(
            "TEST.1.5",
            persist=False,
            output_dir=tmp_path,
        ) as collector:
            collector.record_finding(summary="Transient finding", objective_ids=["obj-1"])

        files = list(tmp_path.glob("mission-*.jsonl"))
        assert len(files) == 0


class TestStateExtraction:
    """Tests for recording evidence from AgentState."""

    def test_record_finding_records_from_state(self) -> None:
        """Should record structured finding records from state."""
        state: AgentState = {
            "mission_id": "TEST.1.6",
            "finding_records": [
                {
                    "finding_id": "finding-000-a",
                    "summary": "Finding one",
                    "objective_ids": ["objective-0"],
                    "contract_slot_tags": ["slot-objective-0"],
                    "source_urls": ["https://example.com/a"],
                    "citations": ["https://example.com/a"],
                    "provenance": {
                        "lane": "google",
                        "query": "query a",
                        "loop": 0,
                        "source_url": "https://example.com/a",
                    },
                    "confidence": 0.8,
                },
                {
                    "finding_id": "finding-000-b",
                    "summary": "Finding two",
                    "objective_ids": ["objective-1"],
                    "contract_slot_tags": [],
                    "source_urls": ["https://example.com/b"],
                    "citations": ["https://example.com/b"],
                    "provenance": {
                        "lane": "exa",
                        "query": "query b",
                        "loop": 0,
                        "source_url": "https://example.com/b",
                    },
                    "confidence": 0.6,
                },
            ],
        }

        collector = TelemetryCollector(mission_id="TEST.1.6")
        findings = record_finding_records_from_state(state, collector)

        assert len(findings) == 2
        telemetry = collector.get_telemetry()
        assert len(telemetry.findings) == 2
        assert telemetry.findings[0].contract_slot_tags == ["slot-objective-0"]
        assert telemetry.findings[1].provenance.lane == "exa"

    def test_extract_sources_from_state(self) -> None:
        """Should extract sources from state and record to collector."""
        state: AgentState = {
            "mission_id": "TEST.1.7",
            "sources_found": [
                {"url": "https://example.com/a", "title": "Source A", "snippet": "Content A"},
                {"url": "https://example.com/b", "title": "Source B"},
            ],
        }

        collector = TelemetryCollector(mission_id="TEST.1.7")
        sources = extract_sources_from_state(state, collector)

        assert len(sources) == 2
        telemetry = collector.get_telemetry()
        assert len(telemetry.sources) == 2


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_record_finding_function(self) -> None:
        """record_finding should delegate to collector."""
        collector = TelemetryCollector(mission_id="TEST.1.8")
        finding = record_finding(
            collector,
            summary="Test summary",
            citations=["https://example.com"],
            objective_ids=["obj-1"],
        )

        assert finding.objective_ids == ["obj-1"]
        assert finding.summary == "Test summary"

    def test_record_source_function(self) -> None:
        """record_source should accept metadata dict."""
        collector = TelemetryCollector(mission_id="TEST.1.9")
        source = record_source(
            collector,
            {
                "url": "https://example.com",
                "title": "Source Title",
                "snippet": "Excerpt",
            },
        )

        assert source.url == "https://example.com"
        assert source.title == "Source Title"


class TestDepthMetrics:
    """Tests for depth-tier specific telemetry metrics."""

    def test_depth_metrics_defaults(self) -> None:
        """DepthMetrics should have sensible defaults."""
        from deepsearch.agent.evidence import DepthMetrics

        metrics = DepthMetrics()
        assert metrics.sources_filtered_quality == 0
        assert metrics.sources_filtered_diversity == 0
        assert metrics.unique_domains == 0
        assert metrics.contradictions_found == 0
        assert metrics.convergence_loop is None

    def test_depth_metrics_to_dict(self) -> None:
        """DepthMetrics.to_dict should return serializable dict."""
        from deepsearch.agent.evidence import DepthMetrics

        metrics = DepthMetrics(
            sources_filtered_quality=3,
            sources_filtered_diversity=2,
            unique_domains=5,
            contradictions_found=1,
            convergence_loop=4,
        )
        data = metrics.to_dict()

        assert data["sources_filtered_quality"] == 3
        assert data["sources_filtered_diversity"] == 2
        assert data["unique_domains"] == 5
        assert data["contradictions_found"] == 1
        assert data["convergence_loop"] == 4

    def test_mission_telemetry_includes_research_depth(self) -> None:
        """MissionTelemetry should include research_depth field."""
        telemetry = MissionTelemetry(
            mission_id="TEST.2.0",
            research_depth="alpha",
        )
        assert telemetry.research_depth == "alpha"
        data = telemetry.to_dict()
        assert data["research_depth"] == "alpha"

    def test_mission_telemetry_includes_depth_metrics(self) -> None:
        """MissionTelemetry should include depth_metrics object."""
        from deepsearch.agent.evidence import DepthMetrics

        telemetry = MissionTelemetry(
            mission_id="TEST.2.1",
            research_depth="deep",
            depth_metrics=DepthMetrics(
                sources_filtered_quality=5,
                unique_domains=8,
            ),
        )
        assert telemetry.depth_metrics.sources_filtered_quality == 5
        assert telemetry.depth_metrics.unique_domains == 8

        data = telemetry.to_dict()
        assert "depth_metrics" in data
        assert data["depth_metrics"]["sources_filtered_quality"] == 5

    def test_mission_telemetry_default_depth_metrics(self) -> None:
        """MissionTelemetry should have default DepthMetrics."""
        telemetry = MissionTelemetry(mission_id="TEST.2.2")
        assert telemetry.research_depth == "baseline"
        assert telemetry.depth_metrics.sources_filtered_quality == 0
        assert telemetry.depth_metrics.convergence_loop is None


class TestTelemetryCollectorDepthMethods:
    """Tests for depth metrics methods on TelemetryCollector."""

    def test_collector_accepts_research_depth(self) -> None:
        """TelemetryCollector should accept research_depth parameter."""
        collector = TelemetryCollector(
            mission_id="TEST.2.3",
            research_depth="alpha",
        )
        telemetry = collector.get_telemetry()
        assert telemetry.research_depth == "alpha"

    def test_collector_set_depth_metrics(self) -> None:
        """Collector should update depth metrics via set_depth_metrics."""
        collector = TelemetryCollector(
            mission_id="TEST.2.4",
            research_depth="deep",
        )

        collector.set_depth_metrics(
            sources_filtered_quality=3,
            sources_filtered_diversity=2,
            unique_domains=6,
            contradictions_found=1,
            convergence_loop=3,
        )

        metrics = collector.get_depth_metrics()
        assert metrics.sources_filtered_quality == 3
        assert metrics.sources_filtered_diversity == 2
        assert metrics.unique_domains == 6
        assert metrics.contradictions_found == 1
        assert metrics.convergence_loop == 3

    def test_collector_set_depth_metrics_partial_update(self) -> None:
        """set_depth_metrics should only update provided fields."""
        collector = TelemetryCollector(mission_id="TEST.2.5")

        # First update
        collector.set_depth_metrics(unique_domains=5)

        # Second update - should not clear unique_domains
        collector.set_depth_metrics(sources_filtered_quality=2)

        metrics = collector.get_depth_metrics()
        assert metrics.unique_domains == 5
        assert metrics.sources_filtered_quality == 2

    def test_collector_increment_filtered_quality(self) -> None:
        """increment_filtered_quality should add to existing count."""
        collector = TelemetryCollector(mission_id="TEST.2.6")

        collector.increment_filtered_quality(3)
        collector.increment_filtered_quality(2)

        metrics = collector.get_depth_metrics()
        assert metrics.sources_filtered_quality == 5

    def test_collector_increment_filtered_diversity(self) -> None:
        """increment_filtered_diversity should add to existing count."""
        collector = TelemetryCollector(mission_id="TEST.2.7")

        collector.increment_filtered_diversity(2)
        collector.increment_filtered_diversity()  # default 1

        metrics = collector.get_depth_metrics()
        assert metrics.sources_filtered_diversity == 3

    def test_collector_set_convergence_loop(self) -> None:
        """set_convergence_loop should mark convergence point."""
        collector = TelemetryCollector(mission_id="TEST.2.8")

        assert collector.get_depth_metrics().convergence_loop is None

        collector.set_convergence_loop(4)

        assert collector.get_depth_metrics().convergence_loop == 4

    def test_collector_depth_metrics_in_finalized_telemetry(self) -> None:
        """Finalized telemetry should include depth metrics."""
        collector = TelemetryCollector(
            mission_id="TEST.2.9",
            research_depth="alpha",
        )

        collector.set_depth_metrics(
            sources_filtered_quality=10,
            sources_filtered_diversity=5,
            unique_domains=12,
            contradictions_found=2,
            convergence_loop=5,
        )

        telemetry = collector.finalize()

        assert telemetry.research_depth == "alpha"
        assert telemetry.depth_metrics.sources_filtered_quality == 10
        assert telemetry.depth_metrics.sources_filtered_diversity == 5
        assert telemetry.depth_metrics.unique_domains == 12
        assert telemetry.depth_metrics.contradictions_found == 2
        assert telemetry.depth_metrics.convergence_loop == 5

        # Verify serialization
        data = telemetry.to_dict()
        assert data["research_depth"] == "alpha"
        assert data["depth_metrics"]["contradictions_found"] == 2


class TestTelemetryContextWithDepth:
    """Tests for telemetry_context with research_depth."""

    def test_context_manager_accepts_research_depth(self, tmp_path: Path) -> None:
        """telemetry_context should accept research_depth parameter."""
        with telemetry_context(
            "TEST.3.0",
            research_depth="alpha",
            persist=True,
            output_dir=tmp_path,
        ) as collector:
            collector.set_depth_metrics(
                unique_domains=7,
                contradictions_found=1,
            )

        # Verify persisted telemetry
        files = list(tmp_path.glob("mission-*.jsonl"))
        assert len(files) == 1

        # Read and verify
        import json
        with files[0].open() as f:
            data = json.loads(f.readline())
            assert data["research_depth"] == "alpha"
            assert data["depth_metrics"]["unique_domains"] == 7
            assert data["depth_metrics"]["contradictions_found"] == 1

    def test_context_manager_default_research_depth(self, tmp_path: Path) -> None:
        """telemetry_context should default to baseline depth."""
        with telemetry_context(
            "TEST.3.1",
            persist=True,
            output_dir=tmp_path,
        ):
            pass

        files = list(tmp_path.glob("mission-*.jsonl"))
        import json
        with files[0].open() as f:
            data = json.loads(f.readline())
            assert data["research_depth"] == "baseline"
