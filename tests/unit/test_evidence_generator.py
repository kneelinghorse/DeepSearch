"""Unit tests for TraceLab Evidence generation from AgentState sources."""

from __future__ import annotations

import pytest

from deepsearch.agent.state import AgentState
from deepsearch.report.evidence import (
    Evidence,
    _build_summary,
    _extract_tags,
    deduplicate_by_url,
    generate_evidence_list,
)


class TestEvidenceModel:
    """Tests for the Evidence Pydantic model."""

    def test_evidence_required_fields(self) -> None:
        """Evidence should require evidence_id, source, and summary."""
        evidence = Evidence(
            evidence_id="EV-001",
            source="https://example.com",
            summary="Test summary",
        )
        assert evidence.evidence_id == "EV-001"
        assert evidence.source == "https://example.com"
        assert evidence.summary == "Test summary"

    def test_evidence_optional_fields_default_none(self) -> None:
        """Optional fields should default to None or empty list."""
        evidence = Evidence(
            evidence_id="EV-002",
            source="https://example.com",
            summary="Test summary",
        )
        assert evidence.chunk_id is None
        assert evidence.insight_id is None
        assert evidence.source_type is None
        assert evidence.relevance_score is None
        assert evidence.tags == []

    def test_evidence_with_all_fields(self) -> None:
        """Evidence should accept all optional fields."""
        evidence = Evidence(
            evidence_id="EV-003",
            source="https://example.com/article",
            summary="Comprehensive research article",
            chunk_id="chunk-123",
            insight_id="insight-456",
            source_type="web_search",
            relevance_score=0.95,
            tags=["research", "example.com"],
        )
        assert evidence.chunk_id == "chunk-123"
        assert evidence.insight_id == "insight-456"
        assert evidence.source_type == "web_search"
        assert evidence.relevance_score == 0.95
        assert len(evidence.tags) == 2

    def test_evidence_relevance_score_validation(self) -> None:
        """Relevance score should be between 0.0 and 1.0."""
        # Valid score
        evidence = Evidence(
            evidence_id="EV-004",
            source="https://example.com",
            summary="Test",
            relevance_score=0.5,
        )
        assert evidence.relevance_score == 0.5

        # Score at boundaries
        evidence_min = Evidence(
            evidence_id="EV-005",
            source="https://example.com",
            summary="Test",
            relevance_score=0.0,
        )
        assert evidence_min.relevance_score == 0.0

        evidence_max = Evidence(
            evidence_id="EV-006",
            source="https://example.com",
            summary="Test",
            relevance_score=1.0,
        )
        assert evidence_max.relevance_score == 1.0

    def test_evidence_relevance_score_out_of_range(self) -> None:
        """Relevance score outside 0-1 range should raise validation error."""
        with pytest.raises(ValueError):
            Evidence(
                evidence_id="EV-007",
                source="https://example.com",
                summary="Test",
                relevance_score=1.5,
            )

        with pytest.raises(ValueError):
            Evidence(
                evidence_id="EV-008",
                source="https://example.com",
                summary="Test",
                relevance_score=-0.1,
            )

    def test_evidence_to_dict(self) -> None:
        """to_dict should return dict without None values."""
        evidence = Evidence(
            evidence_id="EV-009",
            source="https://example.com",
            summary="Test summary",
            source_type="web_search",
        )
        data = evidence.to_dict()

        assert data["evidence_id"] == "EV-009"
        assert data["source"] == "https://example.com"
        assert data["summary"] == "Test summary"
        assert data["source_type"] == "web_search"
        # None values should be excluded
        assert "chunk_id" not in data
        assert "insight_id" not in data
        assert "relevance_score" not in data


class TestGenerateEvidenceList:
    """Tests for generate_evidence_list function."""

    def test_generate_evidence_list_normal(self) -> None:
        """Normal case: 5 sources should produce 5 Evidence objects."""
        state: AgentState = {
            "sources_found": [
                {
                    "url": "https://example.com/article1",
                    "title": "Article One",
                    "snippet": "First article with detailed content about the research topic.",
                },
                {
                    "url": "https://example.com/article2",
                    "title": "Article Two",
                    "snippet": "Second article covering different aspects of the subject.",
                },
                {
                    "url": "https://example.org/resource",
                    "title": "External Resource",
                    "snippet": "Third party resource with complementary information.",
                },
                {
                    "url": "https://docs.example.com/guide",
                    "title": "Documentation Guide",
                    "snippet": "Official documentation providing technical details.",
                },
                {
                    "url": "https://blog.example.io/post",
                    "title": "Blog Post",
                    "snippet": "Blog post discussing practical applications.",
                },
            ],
        }

        evidence = generate_evidence_list(state)

        assert len(evidence) == 5
        # Verify evidence_ids are sequential
        for i, ev in enumerate(evidence, start=1):
            assert ev.evidence_id == f"EV-{i:03d}"

        # Verify all have required fields
        for ev in evidence:
            assert ev.source
            assert ev.summary
            assert ev.source_type == "web_search"

    def test_generate_evidence_list_empty(self) -> None:
        """Empty sources_found should return empty list, not error."""
        state: AgentState = {"sources_found": []}
        evidence = generate_evidence_list(state)
        assert evidence == []

    def test_generate_evidence_list_none_sources(self) -> None:
        """None sources_found should return empty list."""
        state: AgentState = {"sources_found": None}
        evidence = generate_evidence_list(state)
        assert evidence == []

    def test_generate_evidence_list_missing_sources_key(self) -> None:
        """Missing sources_found key should return empty list."""
        state: AgentState = {}
        evidence = generate_evidence_list(state)
        assert evidence == []

    def test_generate_evidence_list_deduplication(self) -> None:
        """Duplicate URLs should be deduplicated, keeping first occurrence."""
        state: AgentState = {
            "sources_found": [
                {
                    "url": "https://example.com/article",
                    "title": "First Occurrence",
                    "snippet": "This is the first time we see this URL.",
                },
                {
                    "url": "https://example.com/article",
                    "title": "Duplicate",
                    "snippet": "This duplicate should be skipped.",
                },
                {
                    "url": "https://example.com/other",
                    "title": "Other Article",
                    "snippet": "This is a different URL, should be included.",
                },
            ],
        }

        evidence = generate_evidence_list(state)

        assert len(evidence) == 2
        assert evidence[0].evidence_id == "EV-001"
        assert evidence[1].evidence_id == "EV-002"
        # First occurrence title should be kept
        assert "First Occurrence" in evidence[0].summary or "first time" in evidence[0].summary

    def test_generate_evidence_list_sequential_ids_after_dedup(self) -> None:
        """Evidence IDs should be sequential even after deduplication (no gaps)."""
        state: AgentState = {
            "sources_found": [
                {"url": "https://a.com", "title": "A", "snippet": "Source A content here."},
                {"url": "https://a.com", "title": "A dup", "snippet": "Duplicate of A."},
                {"url": "https://b.com", "title": "B", "snippet": "Source B content here."},
                {"url": "https://b.com", "title": "B dup", "snippet": "Duplicate of B."},
                {"url": "https://c.com", "title": "C", "snippet": "Source C content here."},
            ],
        }

        evidence = generate_evidence_list(state)

        assert len(evidence) == 3
        ids = [ev.evidence_id for ev in evidence]
        assert ids == ["EV-001", "EV-002", "EV-003"]

    def test_generate_evidence_list_missing_snippet(self) -> None:
        """Source missing snippet should fall back to title."""
        state: AgentState = {
            "sources_found": [
                {
                    "url": "https://example.com/article",
                    "title": "Article Without Snippet",
                    # No snippet field
                },
            ],
        }

        evidence = generate_evidence_list(state)

        assert len(evidence) == 1
        assert "Source: Article Without Snippet" == evidence[0].summary

    def test_generate_evidence_list_missing_snippet_and_title(self) -> None:
        """Source missing both snippet and title should use default summary."""
        state: AgentState = {
            "sources_found": [
                {
                    "url": "https://example.com/unknown",
                    # No snippet, no title
                },
            ],
        }

        evidence = generate_evidence_list(state)

        assert len(evidence) == 1
        assert evidence[0].summary == "Web search result"

    def test_generate_evidence_list_empty_snippet(self) -> None:
        """Empty snippet string should fall back to title."""
        state: AgentState = {
            "sources_found": [
                {
                    "url": "https://example.com/article",
                    "title": "Article Title",
                    "snippet": "",  # Empty string
                },
            ],
        }

        evidence = generate_evidence_list(state)

        assert len(evidence) == 1
        assert "Source: Article Title" == evidence[0].summary

    def test_generate_evidence_list_short_snippet(self) -> None:
        """Snippet shorter than 10 chars should fall back to title."""
        state: AgentState = {
            "sources_found": [
                {
                    "url": "https://example.com/article",
                    "title": "Article Title",
                    "snippet": "Short",  # Less than 10 chars
                },
            ],
        }

        evidence = generate_evidence_list(state)

        assert len(evidence) == 1
        assert "Source: Article Title" == evidence[0].summary

    def test_generate_evidence_list_missing_url(self) -> None:
        """Source without URL should use title as source identifier."""
        state: AgentState = {
            "sources_found": [
                {
                    "title": "Source Without URL",
                    "snippet": "This source has no URL but has content.",
                },
            ],
        }

        evidence = generate_evidence_list(state)

        assert len(evidence) == 1
        assert evidence[0].source == "Source Without URL"

    def test_generate_evidence_list_source_type_is_web_search(self) -> None:
        """All generated Evidence should have source_type='web_search'."""
        state: AgentState = {
            "sources_found": [
                {"url": "https://example.com", "title": "Test", "snippet": "Test content here."},
            ],
        }

        evidence = generate_evidence_list(state)

        assert len(evidence) == 1
        assert evidence[0].source_type == "web_search"

    def test_generate_evidence_list_includes_domain_tag(self) -> None:
        """Evidence should include domain as tag (without www.)."""
        state: AgentState = {
            "sources_found": [
                {
                    "url": "https://www.example.com/article",
                    "title": "Test",
                    "snippet": "Content for testing domain tag extraction.",
                },
            ],
        }

        evidence = generate_evidence_list(state)

        assert len(evidence) == 1
        assert "research" in evidence[0].tags
        assert "example.com" in evidence[0].tags

    def test_generate_evidence_list_ignores_non_dict_sources(self) -> None:
        """Non-dict entries in sources_found should be skipped."""
        state: AgentState = {
            "sources_found": [
                {"url": "https://valid.com", "title": "Valid", "snippet": "Valid source content."},
                "invalid string entry",
                123,
                None,
                {"url": "https://also-valid.com", "title": "Also Valid", "snippet": "Also valid content."},
            ],
        }

        evidence = generate_evidence_list(state)

        assert len(evidence) == 2
        assert evidence[0].source == "https://valid.com"
        assert evidence[1].source == "https://also-valid.com"


class TestEvidenceIdFormat:
    """Tests for evidence_id format (EV-{index:03d})."""

    def test_evidence_id_format_single_digit(self) -> None:
        """Evidence ID should be zero-padded to 3 digits."""
        state: AgentState = {
            "sources_found": [
                {"url": "https://example.com", "title": "Test", "snippet": "Test content."},
            ],
        }

        evidence = generate_evidence_list(state)

        assert evidence[0].evidence_id == "EV-001"

    def test_evidence_id_format_double_digit(self) -> None:
        """Evidence ID should handle double digits correctly."""
        sources = [
            {"url": f"https://example{i}.com", "title": f"Test {i}", "snippet": f"Content {i}"}
            for i in range(15)
        ]
        state: AgentState = {"sources_found": sources}

        evidence = generate_evidence_list(state)

        assert len(evidence) == 15
        assert evidence[9].evidence_id == "EV-010"
        assert evidence[14].evidence_id == "EV-015"

    def test_evidence_id_format_triple_digit(self) -> None:
        """Evidence ID should handle triple digits correctly."""
        sources = [
            {"url": f"https://example{i}.com", "title": f"Test {i}", "snippet": f"Content {i}"}
            for i in range(105)
        ]
        state: AgentState = {"sources_found": sources}

        evidence = generate_evidence_list(state)

        assert len(evidence) == 105
        assert evidence[99].evidence_id == "EV-100"
        assert evidence[104].evidence_id == "EV-105"


class TestBuildSummary:
    """Tests for _build_summary helper function."""

    def test_build_summary_uses_snippet(self) -> None:
        """Should use snippet if present and >= 10 chars."""
        source = {"snippet": "This is a valid snippet with enough content."}
        summary = _build_summary(source)
        assert summary == "This is a valid snippet with enough content."

    def test_build_summary_truncates_long_snippet(self) -> None:
        """Should truncate snippets longer than 500 chars."""
        long_snippet = "A" * 600
        source = {"snippet": long_snippet}
        summary = _build_summary(source)
        assert len(summary) == 500
        assert summary == "A" * 500

    def test_build_summary_falls_back_to_title(self) -> None:
        """Should use title if snippet is missing or too short."""
        source = {"title": "Article Title", "snippet": "Short"}
        summary = _build_summary(source)
        assert summary == "Source: Article Title"

    def test_build_summary_default(self) -> None:
        """Should return default if no snippet or title."""
        source = {}
        summary = _build_summary(source)
        assert summary == "Web search result"

    def test_build_summary_strips_whitespace(self) -> None:
        """Should strip whitespace from snippet."""
        source = {"snippet": "   Content with whitespace   "}
        summary = _build_summary(source)
        assert summary == "Content with whitespace"


class TestExtractTags:
    """Tests for _extract_tags helper function."""

    def test_extract_tags_always_includes_research(self) -> None:
        """Should always include 'research' tag."""
        source = {}
        tags = _extract_tags(source)
        assert "research" in tags

    def test_extract_tags_includes_domain(self) -> None:
        """Should include domain from URL."""
        source = {"url": "https://docs.python.org/tutorial"}
        tags = _extract_tags(source)
        assert "research" in tags
        assert "docs.python.org" in tags

    def test_extract_tags_removes_www(self) -> None:
        """Should remove www. prefix from domain."""
        source = {"url": "https://www.example.com/page"}
        tags = _extract_tags(source)
        assert "example.com" in tags
        assert "www.example.com" not in tags

    def test_extract_tags_no_url(self) -> None:
        """Should only include 'research' if no URL."""
        source = {"title": "No URL"}
        tags = _extract_tags(source)
        assert tags == ["research"]

    def test_extract_tags_empty_url(self) -> None:
        """Should only include 'research' if URL is empty."""
        source = {"url": ""}
        tags = _extract_tags(source)
        assert tags == ["research"]

    def test_extract_tags_include_target_slot_provenance(self) -> None:
        """Target slots should be preserved as protocol-ready evidence tags."""
        source = {
            "url": "https://example.com/page",
            "target_slots": ["slot-objective-1", "slot-entity-openai"],
        }
        tags = _extract_tags(source)
        assert "slot:slot-objective-1" in tags
        assert "slot:slot-entity-openai" in tags


class TestDeduplicateByUrl:
    """Tests for deduplicate_by_url utility function."""

    def test_deduplicate_keeps_first(self) -> None:
        """Should keep first occurrence of duplicate URLs."""
        sources = [
            {"url": "https://a.com", "title": "First A"},
            {"url": "https://a.com", "title": "Second A"},
            {"url": "https://b.com", "title": "First B"},
        ]

        result = deduplicate_by_url(sources)

        assert len(result) == 2
        assert result[0]["title"] == "First A"
        assert result[1]["title"] == "First B"

    def test_deduplicate_preserves_order(self) -> None:
        """Should preserve original order."""
        sources = [
            {"url": "https://c.com", "title": "C"},
            {"url": "https://a.com", "title": "A"},
            {"url": "https://b.com", "title": "B"},
        ]

        result = deduplicate_by_url(sources)

        assert [s["title"] for s in result] == ["C", "A", "B"]

    def test_deduplicate_empty_list(self) -> None:
        """Should handle empty list."""
        result = deduplicate_by_url([])
        assert result == []

    def test_deduplicate_no_duplicates(self) -> None:
        """Should return same list if no duplicates."""
        sources = [
            {"url": "https://a.com", "title": "A"},
            {"url": "https://b.com", "title": "B"},
        ]

        result = deduplicate_by_url(sources)

        assert len(result) == 2

    def test_deduplicate_handles_missing_url(self) -> None:
        """Should include sources without URL (can't dedupe them)."""
        sources = [
            {"title": "No URL 1"},
            {"title": "No URL 2"},
            {"url": "https://a.com", "title": "With URL"},
        ]

        result = deduplicate_by_url(sources)

        assert len(result) == 3


class TestSchemaValidation:
    """Tests for TraceLab schema compatibility."""

    def test_evidence_model_validates(self) -> None:
        """Evidence should pass Pydantic validation."""
        evidence = Evidence(
            evidence_id="EV-001",
            source="https://example.com",
            summary="Test summary",
        )
        # Re-validate from dict
        validated = Evidence.model_validate(evidence.model_dump())
        assert validated.evidence_id == "EV-001"

    def test_generated_evidence_validates(self) -> None:
        """Generated Evidence objects should pass schema validation."""
        state: AgentState = {
            "sources_found": [
                {
                    "url": "https://example.com/article",
                    "title": "Article",
                    "snippet": "Article content for validation testing.",
                },
            ],
        }

        evidence = generate_evidence_list(state)

        for ev in evidence:
            # Should not raise
            Evidence.model_validate(ev.model_dump())

    def test_evidence_non_empty_summary(self) -> None:
        """Every generated Evidence should have non-empty summary."""
        state: AgentState = {
            "sources_found": [
                {"url": "https://a.com"},  # No snippet, no title
                {"url": "https://b.com", "snippet": ""},  # Empty snippet
                {"url": "https://c.com", "title": ""},  # Empty title
                {"url": "https://d.com", "snippet": "Short"},  # Too short
            ],
        }

        evidence = generate_evidence_list(state)

        for ev in evidence:
            assert ev.summary, f"Evidence {ev.evidence_id} has empty summary"
            assert len(ev.summary) > 0
