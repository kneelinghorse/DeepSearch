"""Unit tests for contradiction detection (B10.9).

Tests cover:
- Contradiction dataclass construction and serialization
- ContradictionReport creation and properties
- Prompt building for contradiction detection
- Response parsing from LLM output
- Confidence penalty calculation
- Integration with mock LLM responses
"""

import json
import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

from deepsearch.agent.contradiction import (
    Contradiction,
    ContradictionReport,
    HIGH_CONFIDENCE_THRESHOLD,
    CONFIDENCE_PENALTY_PER_CONTRADICTION,
    build_contradiction_detection_prompt,
    parse_contradiction_response,
    detect_contradictions,
    apply_contradiction_penalty,
)


class TestContradiction:
    """Tests for Contradiction dataclass."""

    def test_basic_construction(self) -> None:
        """Create a basic contradiction."""
        c = Contradiction(
            claim_a="The market size is $10 billion",
            claim_b="The market size is $100 billion",
            topic="market size",
            confidence=0.9,
        )
        assert c.claim_a == "The market size is $10 billion"
        assert c.claim_b == "The market size is $100 billion"
        assert c.topic == "market size"
        assert c.confidence == 0.9

    def test_confidence_clamped_to_max(self) -> None:
        """Confidence above 1.0 is clamped."""
        c = Contradiction(
            claim_a="A",
            claim_b="B",
            topic="test",
            confidence=1.5,
        )
        assert c.confidence == 1.0

    def test_confidence_clamped_to_min(self) -> None:
        """Confidence below 0.0 is clamped."""
        c = Contradiction(
            claim_a="A",
            claim_b="B",
            topic="test",
            confidence=-0.5,
        )
        assert c.confidence == 0.0

    def test_to_dict(self) -> None:
        """to_dict() includes all fields."""
        c = Contradiction(
            claim_a="Claim A",
            claim_b="Claim B",
            topic="topic",
            confidence=0.8,
            source_a="https://a.com",
            source_b="https://b.com",
            explanation="They directly conflict",
        )
        d = c.to_dict()
        assert d["claim_a"] == "Claim A"
        assert d["claim_b"] == "Claim B"
        assert d["topic"] == "topic"
        assert d["confidence"] == 0.8
        assert d["source_a"] == "https://a.com"
        assert d["source_b"] == "https://b.com"
        assert d["explanation"] == "They directly conflict"

    def test_from_dict(self) -> None:
        """from_dict() reconstructs contradiction."""
        data = {
            "claim_a": "X is 10",
            "claim_b": "X is 100",
            "topic": "numbers",
            "confidence": 0.95,
            "explanation": "Direct numerical conflict",
        }
        c = Contradiction.from_dict(data)
        assert c.claim_a == "X is 10"
        assert c.claim_b == "X is 100"
        assert c.confidence == 0.95

    def test_is_high_confidence_true(self) -> None:
        """Contradictions at or above threshold are high confidence."""
        c = Contradiction(
            claim_a="A", claim_b="B", topic="T",
            confidence=HIGH_CONFIDENCE_THRESHOLD,
        )
        assert c.is_high_confidence is True

    def test_is_high_confidence_false(self) -> None:
        """Contradictions below threshold are not high confidence."""
        c = Contradiction(
            claim_a="A", claim_b="B", topic="T",
            confidence=HIGH_CONFIDENCE_THRESHOLD - 0.01,
        )
        assert c.is_high_confidence is False

    def test_optional_fields_default_to_none(self) -> None:
        """Optional fields default to None."""
        c = Contradiction(
            claim_a="A", claim_b="B", topic="T", confidence=0.5,
        )
        assert c.source_a is None
        assert c.source_b is None
        assert c.explanation is None


class TestContradictionReport:
    """Tests for ContradictionReport dataclass."""

    def test_empty_report(self) -> None:
        """Empty report has no contradictions."""
        report = ContradictionReport.empty()
        assert report.has_contradictions is False
        assert report.high_confidence_count == 0
        assert report.confidence_penalty == 0.0
        assert report.analysis_complete is True

    def test_error_report(self) -> None:
        """Error report captures error message."""
        report = ContradictionReport.error("LLM timeout")
        assert report.analysis_complete is False
        assert report.error_message == "LLM timeout"
        assert report.has_contradictions is False

    def test_has_contradictions_property(self) -> None:
        """has_contradictions is True when contradictions exist."""
        report = ContradictionReport(
            contradictions=[
                Contradiction("A", "B", "T", 0.6),
            ]
        )
        assert report.has_contradictions is True

    def test_high_confidence_count(self) -> None:
        """Count only high-confidence contradictions."""
        report = ContradictionReport(
            contradictions=[
                Contradiction("A", "B", "T", 0.9),  # High
                Contradiction("C", "D", "T", 0.7),  # Not high
                Contradiction("E", "F", "T", 0.85),  # High
            ]
        )
        assert report.high_confidence_count == 2

    def test_confidence_penalty_single_high(self) -> None:
        """Single high-confidence contradiction applies one penalty."""
        report = ContradictionReport(
            contradictions=[
                Contradiction("A", "B", "T", 0.9),
            ]
        )
        expected = CONFIDENCE_PENALTY_PER_CONTRADICTION
        assert report.confidence_penalty == expected

    def test_confidence_penalty_multiple_high(self) -> None:
        """Multiple high-confidence contradictions stack penalties."""
        report = ContradictionReport(
            contradictions=[
                Contradiction("A", "B", "T", 0.9),
                Contradiction("C", "D", "T", 0.85),
                Contradiction("E", "F", "T", 0.95),
            ]
        )
        expected = 3 * CONFIDENCE_PENALTY_PER_CONTRADICTION
        assert report.confidence_penalty == expected

    def test_confidence_penalty_capped_at_50_percent(self) -> None:
        """Confidence penalty capped at 50%."""
        # Create enough high-confidence contradictions to exceed 50%
        contradictions = [
            Contradiction("A", "B", f"T{i}", 0.9)
            for i in range(10)  # Would be 100% without cap
        ]
        report = ContradictionReport(contradictions=contradictions)
        assert report.confidence_penalty == 0.5

    def test_confidence_penalty_no_high_confidence(self) -> None:
        """No penalty when no high-confidence contradictions."""
        report = ContradictionReport(
            contradictions=[
                Contradiction("A", "B", "T", 0.6),
                Contradiction("C", "D", "T", 0.7),
            ]
        )
        assert report.confidence_penalty == 0.0

    def test_to_dict(self) -> None:
        """to_dict() includes all computed properties."""
        report = ContradictionReport(
            contradictions=[
                Contradiction("A", "B", "T", 0.9),
            ],
            sources_analyzed=5,
            claims_extracted=10,
        )
        d = report.to_dict()
        assert d["sources_analyzed"] == 5
        assert d["claims_extracted"] == 10
        assert d["has_contradictions"] is True
        assert d["high_confidence_count"] == 1
        assert d["confidence_penalty"] == CONFIDENCE_PENALTY_PER_CONTRADICTION
        assert len(d["contradictions"]) == 1

    def test_from_dict(self) -> None:
        """from_dict() reconstructs report with contradictions."""
        data = {
            "contradictions": [
                {"claim_a": "X", "claim_b": "Y", "topic": "Z", "confidence": 0.8}
            ],
            "sources_analyzed": 3,
            "claims_extracted": 6,
            "analysis_complete": True,
        }
        report = ContradictionReport.from_dict(data)
        assert report.has_contradictions is True
        assert report.sources_analyzed == 3
        assert len(report.contradictions) == 1


class TestBuildContradictionDetectionPrompt:
    """Tests for build_contradiction_detection_prompt()."""

    def test_returns_system_and_human_prompts(self) -> None:
        """Returns tuple of (system_prompt, human_prompt)."""
        findings = ["Finding 1", "Finding 2"]
        sources = [{"url": "https://example.com", "title": "Example"}]
        objectives = ["Objective 1"]

        system, human = build_contradiction_detection_prompt(
            findings, sources, objectives
        )

        assert isinstance(system, str)
        assert isinstance(human, str)
        assert len(system) > 0
        assert len(human) > 0

    def test_system_prompt_includes_json_schema(self) -> None:
        """System prompt describes JSON output schema."""
        system, _ = build_contradiction_detection_prompt(
            ["F1"], [{"url": "u"}], ["O1"]
        )
        assert "contradictions" in system
        assert "claim_a" in system
        assert "claim_b" in system
        assert "confidence" in system

    def test_human_prompt_includes_findings(self) -> None:
        """Human prompt includes all findings."""
        findings = ["Market is growing", "Revenue increased"]
        _, human = build_contradiction_detection_prompt(
            findings, [], []
        )
        assert "Market is growing" in human
        assert "Revenue increased" in human

    def test_human_prompt_includes_sources(self) -> None:
        """Human prompt includes source information."""
        sources = [
            {"url": "https://a.com", "title": "Source A"},
            {"url": "https://b.com", "title": "Source B"},
        ]
        _, human = build_contradiction_detection_prompt(
            ["F"], sources, []
        )
        assert "Source A" in human
        assert "Source B" in human

    def test_human_prompt_includes_objectives(self) -> None:
        """Human prompt includes mission objectives."""
        objectives = ["Understand market dynamics"]
        _, human = build_contradiction_detection_prompt(
            ["F"], [], objectives
        )
        assert "Understand market dynamics" in human

    def test_findings_numbered(self) -> None:
        """Findings are numbered for reference."""
        findings = ["First", "Second", "Third"]
        _, human = build_contradiction_detection_prompt(findings, [], [])
        assert "[1]" in human
        assert "[2]" in human
        assert "[3]" in human


class TestParseContradictionResponse:
    """Tests for parse_contradiction_response()."""

    def test_parse_valid_json(self) -> None:
        """Parse valid JSON response."""
        response = json.dumps({
            "contradictions": [
                {
                    "claim_a": "X is 10",
                    "claim_b": "X is 100",
                    "topic": "numbers",
                    "confidence": 0.9,
                    "explanation": "Direct conflict",
                }
            ]
        })
        result = parse_contradiction_response(response)
        assert len(result) == 1
        assert result[0].claim_a == "X is 10"
        assert result[0].confidence == 0.9

    def test_parse_json_with_markdown_code_block(self) -> None:
        """Parse JSON wrapped in markdown code block."""
        inner = json.dumps({
            "contradictions": [
                {"claim_a": "A", "claim_b": "B", "topic": "T", "confidence": 0.8}
            ]
        })
        response = f"```json\n{inner}\n```"
        result = parse_contradiction_response(response)
        assert len(result) == 1

    def test_parse_json_without_language_tag(self) -> None:
        """Parse JSON in code block without language tag."""
        inner = json.dumps({
            "contradictions": [
                {"claim_a": "A", "claim_b": "B", "topic": "T", "confidence": 0.8}
            ]
        })
        response = f"```\n{inner}\n```"
        result = parse_contradiction_response(response)
        assert len(result) == 1

    def test_parse_empty_contradictions(self) -> None:
        """Parse response with no contradictions."""
        response = json.dumps({"contradictions": []})
        result = parse_contradiction_response(response)
        assert len(result) == 0

    def test_filter_low_confidence(self) -> None:
        """Filter out low-confidence contradictions (below 0.5)."""
        response = json.dumps({
            "contradictions": [
                {"claim_a": "A", "claim_b": "B", "topic": "T", "confidence": 0.3},
                {"claim_a": "C", "claim_b": "D", "topic": "T", "confidence": 0.7},
            ]
        })
        result = parse_contradiction_response(response)
        assert len(result) == 1
        assert result[0].claim_a == "C"

    def test_handle_invalid_json(self) -> None:
        """Return empty list for invalid JSON."""
        response = "This is not JSON at all"
        result = parse_contradiction_response(response)
        assert len(result) == 0

    def test_handle_empty_response(self) -> None:
        """Return empty list for empty response."""
        result = parse_contradiction_response("")
        assert len(result) == 0

    def test_skip_items_with_empty_claims(self) -> None:
        """Skip items with empty claim fields."""
        response = json.dumps({
            "contradictions": [
                {"claim_a": "", "claim_b": "B", "topic": "T", "confidence": 0.8},
                {"claim_a": "A", "claim_b": "", "topic": "T", "confidence": 0.8},
                {"claim_a": "A", "claim_b": "B", "topic": "T", "confidence": 0.8},
            ]
        })
        result = parse_contradiction_response(response)
        assert len(result) == 1

    def test_handle_missing_contradictions_key(self) -> None:
        """Return empty list if contradictions key missing."""
        response = json.dumps({"other_key": "value"})
        result = parse_contradiction_response(response)
        assert len(result) == 0

    def test_extract_json_from_surrounding_text(self) -> None:
        """Extract JSON object embedded in text."""
        response = 'Here is my analysis:\n{"contradictions": [{"claim_a": "A", "claim_b": "B", "topic": "T", "confidence": 0.8}]}\nEnd of response.'
        result = parse_contradiction_response(response)
        assert len(result) == 1


class TestApplyContradictionPenalty:
    """Tests for apply_contradiction_penalty()."""

    def test_no_contradictions_no_penalty(self) -> None:
        """No penalty when no contradictions."""
        report = ContradictionReport.empty()
        result = apply_contradiction_penalty(0.8, report)
        assert result == 0.8

    def test_single_high_confidence_penalty(self) -> None:
        """Single high-confidence contradiction applies penalty."""
        report = ContradictionReport(
            contradictions=[Contradiction("A", "B", "T", 0.9)]
        )
        result = apply_contradiction_penalty(0.8, report)
        expected = 0.8 - CONFIDENCE_PENALTY_PER_CONTRADICTION
        assert result == pytest.approx(expected)

    def test_multiple_high_confidence_stacks(self) -> None:
        """Multiple high-confidence contradictions stack."""
        report = ContradictionReport(
            contradictions=[
                Contradiction("A", "B", "T", 0.9),
                Contradiction("C", "D", "T", 0.85),
            ]
        )
        result = apply_contradiction_penalty(0.8, report)
        expected = 0.8 - (2 * CONFIDENCE_PENALTY_PER_CONTRADICTION)
        assert result == pytest.approx(expected)

    def test_penalty_does_not_go_negative(self) -> None:
        """Penalty cannot make confidence negative."""
        # Many high-confidence contradictions
        contradictions = [Contradiction("A", "B", f"T{i}", 0.9) for i in range(10)]
        report = ContradictionReport(contradictions=contradictions)
        result = apply_contradiction_penalty(0.3, report)
        assert result == 0.0

    def test_low_confidence_contradictions_no_penalty(self) -> None:
        """Low-confidence contradictions don't apply penalty."""
        report = ContradictionReport(
            contradictions=[
                Contradiction("A", "B", "T", 0.6),  # Below threshold
                Contradiction("C", "D", "T", 0.7),  # Below threshold
            ]
        )
        result = apply_contradiction_penalty(0.8, report)
        assert result == 0.8


class TestDetectContradictions:
    """Tests for detect_contradictions() async function."""

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM that returns JSON response."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock()
        return llm

    @pytest.fixture
    def sample_findings(self) -> List[str]:
        """Sample findings for testing."""
        return [
            "The market size is $10 billion according to research firm A.",
            "Market analysis shows total addressable market of $100 billion.",
            "Growth rate is expected at 15% annually.",
        ]

    @pytest.fixture
    def sample_sources(self) -> List[Dict[str, Any]]:
        """Sample sources for testing."""
        return [
            {"url": "https://research-a.com", "title": "Research A"},
            {"url": "https://research-b.com", "title": "Research B"},
        ]

    @pytest.mark.asyncio
    async def test_returns_report_with_contradictions(
        self, mock_llm: MagicMock, sample_findings: List[str], sample_sources: List[Dict]
    ) -> None:
        """Returns report with detected contradictions."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "contradictions": [
                {
                    "claim_a": "Market size is $10 billion",
                    "claim_b": "Market size is $100 billion",
                    "topic": "market size",
                    "confidence": 0.95,
                    "explanation": "10x difference in stated market size",
                }
            ]
        })
        mock_llm.ainvoke.return_value = mock_response

        report = await detect_contradictions(
            findings=sample_findings,
            sources=sample_sources,
            mission_objectives=["Analyze market size"],
            llm=mock_llm,
        )

        assert report.analysis_complete is True
        assert report.has_contradictions is True
        assert len(report.contradictions) == 1
        assert report.contradictions[0].topic == "market size"

    @pytest.mark.asyncio
    async def test_returns_empty_report_no_contradictions(
        self, mock_llm: MagicMock, sample_findings: List[str], sample_sources: List[Dict]
    ) -> None:
        """Returns empty report when no contradictions found."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({"contradictions": []})
        mock_llm.ainvoke.return_value = mock_response

        report = await detect_contradictions(
            findings=sample_findings,
            sources=sample_sources,
            mission_objectives=["Analyze growth"],
            llm=mock_llm,
        )

        assert report.analysis_complete is True
        assert report.has_contradictions is False
        assert len(report.contradictions) == 0

    @pytest.mark.asyncio
    async def test_handles_insufficient_findings(
        self, mock_llm: MagicMock, sample_sources: List[Dict]
    ) -> None:
        """Returns empty report with insufficient findings."""
        report = await detect_contradictions(
            findings=["Only one finding"],
            sources=sample_sources,
            mission_objectives=[],
            llm=mock_llm,
        )

        assert report.analysis_complete is True
        assert report.has_contradictions is False
        # LLM should not be called
        mock_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_empty_findings(
        self, mock_llm: MagicMock, sample_sources: List[Dict]
    ) -> None:
        """Returns empty report with no findings."""
        report = await detect_contradictions(
            findings=[],
            sources=sample_sources,
            mission_objectives=[],
            llm=mock_llm,
        )

        assert report.analysis_complete is True
        mock_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_llm_error(
        self, mock_llm: MagicMock, sample_findings: List[str], sample_sources: List[Dict]
    ) -> None:
        """Returns error report when LLM fails."""
        mock_llm.ainvoke.side_effect = Exception("LLM timeout")

        report = await detect_contradictions(
            findings=sample_findings,
            sources=sample_sources,
            mission_objectives=[],
            llm=mock_llm,
        )

        assert report.analysis_complete is False
        assert "LLM timeout" in report.error_message

    @pytest.mark.asyncio
    async def test_tracks_sources_and_claims(
        self, mock_llm: MagicMock, sample_findings: List[str], sample_sources: List[Dict]
    ) -> None:
        """Report includes count of analyzed sources and claims."""
        mock_response = MagicMock()
        mock_response.content = json.dumps({"contradictions": []})
        mock_llm.ainvoke.return_value = mock_response

        report = await detect_contradictions(
            findings=sample_findings,
            sources=sample_sources,
            mission_objectives=[],
            llm=mock_llm,
        )

        assert report.sources_analyzed == len(sample_sources)
        assert report.claims_extracted == len(sample_findings)

    @pytest.mark.asyncio
    async def test_uses_sync_invoke_fallback(
        self, sample_findings: List[str], sample_sources: List[Dict]
    ) -> None:
        """Falls back to sync invoke if ainvoke not available."""
        mock_llm = MagicMock(spec=[])  # No ainvoke
        mock_llm.invoke = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({"contradictions": []})
        mock_llm.invoke.return_value = mock_response

        report = await detect_contradictions(
            findings=sample_findings,
            sources=sample_sources,
            mission_objectives=[],
            llm=mock_llm,
        )

        assert report.analysis_complete is True
        mock_llm.invoke.assert_called_once()


class TestContradictionIntegrationScenarios:
    """Integration-style tests with realistic scenarios."""

    def test_numerical_contradiction_high_confidence(self) -> None:
        """Direct numerical contradictions score high confidence."""
        response = json.dumps({
            "contradictions": [
                {
                    "claim_a": "Revenue grew 50% year over year",
                    "claim_b": "Revenue declined 10% compared to last year",
                    "topic": "revenue growth",
                    "confidence": 0.95,
                    "explanation": "Direct contradiction: growth vs decline",
                }
            ]
        })
        result = parse_contradiction_response(response)
        assert len(result) == 1
        assert result[0].is_high_confidence is True

    def test_temporal_context_not_contradiction(self) -> None:
        """Different time periods should not be flagged as contradictions."""
        # This tests that low-confidence items are filtered
        response = json.dumps({
            "contradictions": [
                {
                    "claim_a": "In 2020, the company had 100 employees",
                    "claim_b": "In 2024, the company has 500 employees",
                    "topic": "employee count",
                    "confidence": 0.3,  # Low confidence - not a real contradiction
                    "explanation": "Different time periods",
                }
            ]
        })
        result = parse_contradiction_response(response)
        assert len(result) == 0  # Filtered due to low confidence

    def test_penalty_integration(self) -> None:
        """Full integration of penalty calculation."""
        report = ContradictionReport(
            contradictions=[
                Contradiction(
                    "Market size is $10B",
                    "Market size is $100B",
                    "market size",
                    0.95,
                ),
                Contradiction(
                    "Growth is accelerating",
                    "Growth is slowing",
                    "growth rate",
                    0.85,
                ),
            ]
        )

        initial_confidence = 0.9
        adjusted = apply_contradiction_penalty(initial_confidence, report)

        # 2 high-confidence contradictions = 20% penalty
        expected = 0.9 - 0.2
        assert adjusted == pytest.approx(expected)

    def test_report_serialization_roundtrip(self) -> None:
        """Report can be serialized and deserialized."""
        original = ContradictionReport(
            contradictions=[
                Contradiction("A", "B", "T", 0.9, "src_a", "src_b", "expl"),
            ],
            sources_analyzed=10,
            claims_extracted=20,
        )

        as_dict = original.to_dict()
        restored = ContradictionReport.from_dict(as_dict)

        assert restored.has_contradictions == original.has_contradictions
        assert restored.sources_analyzed == original.sources_analyzed
        assert len(restored.contradictions) == len(original.contradictions)
        assert restored.contradictions[0].claim_a == "A"


class TestContradictionConstants:
    """Tests for module constants."""

    def test_high_confidence_threshold_reasonable(self) -> None:
        """High confidence threshold is in reasonable range."""
        assert 0.7 <= HIGH_CONFIDENCE_THRESHOLD <= 0.9

    def test_penalty_per_contradiction_reasonable(self) -> None:
        """Penalty per contradiction is reasonable."""
        assert 0.05 <= CONFIDENCE_PENALTY_PER_CONTRADICTION <= 0.2
