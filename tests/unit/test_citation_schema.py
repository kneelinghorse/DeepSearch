"""Schema validation tests for ``CitationAnnotation`` (S67 plan-doc decision #520).

Locks the OpenAI ``url_citation`` shape verbatim: 5 fields, type discriminator,
codepoint-indexed start/end, non-empty URL, nullable title. ``to_dict()``
excludes ``None`` title from the dump.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepsearch.report.citation import CitationAnnotation
from deepsearch.report.protocol import MissionProtocolComplete
from deepsearch.report.synthesis import ResearchStatement, Synthesis


def _build_minimal_protocol(**overrides) -> MissionProtocolComplete:
    research_statement = ResearchStatement(
        topic="t",
        objective="o",
        scope="s",
    )
    synthesis = Synthesis(
        key_insights=["k"],
        recommendations=["r"],
        next_steps=["n"],
    )
    return MissionProtocolComplete(
        mission_id="M-1",
        title="T",
        research_statement=research_statement,
        synthesis=synthesis,
        **overrides,
    )


class TestCitationAnnotationSchema:
    def test_minimal_valid_annotation(self):
        ann = CitationAnnotation(start_index=0, end_index=5, url="https://example.com")
        assert ann.type == "url_citation"
        assert ann.start_index == 0
        assert ann.end_index == 5
        assert ann.url == "https://example.com"
        assert ann.title is None

    def test_type_discriminator_default(self):
        ann = CitationAnnotation(start_index=0, end_index=1, url="u")
        assert ann.type == "url_citation"

    def test_type_discriminator_rejects_other_value(self):
        with pytest.raises(ValidationError):
            CitationAnnotation(
                type="text_citation",
                start_index=0,
                end_index=1,
                url="u",
            )

    def test_start_index_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            CitationAnnotation(start_index=-1, end_index=0, url="u")

    def test_end_index_can_equal_start_index(self):
        ann = CitationAnnotation(start_index=4, end_index=4, url="u")
        assert ann.end_index == ann.start_index

    def test_end_index_must_be_at_least_start_index(self):
        with pytest.raises(ValidationError):
            CitationAnnotation(start_index=10, end_index=5, url="u")

    def test_url_must_not_be_empty(self):
        with pytest.raises(ValidationError):
            CitationAnnotation(start_index=0, end_index=1, url="")

    def test_title_optional(self):
        ann = CitationAnnotation(start_index=0, end_index=1, url="u", title=None)
        assert ann.title is None
        ann2 = CitationAnnotation(start_index=0, end_index=1, url="u", title="Doc")
        assert ann2.title == "Doc"

    def test_to_dict_excludes_none_title(self):
        ann = CitationAnnotation(start_index=0, end_index=5, url="u")
        data = ann.to_dict()
        assert "title" not in data
        assert data == {
            "type": "url_citation",
            "start_index": 0,
            "end_index": 5,
            "url": "u",
        }

    def test_to_dict_includes_present_title(self):
        ann = CitationAnnotation(start_index=0, end_index=5, url="u", title="Doc")
        data = ann.to_dict()
        assert data["title"] == "Doc"

    def test_field_count_locked_at_six(self):
        # Decision #520 (S67) locked the schema at 5 fields; S68 PD9 adds the
        # additive `live` field for URL-liveness gating. Any further added
        # field would silently expand the protocol — fail loudly here so
        # future sprints get a deliberate review.
        assert set(CitationAnnotation.model_fields.keys()) == {
            "type",
            "start_index",
            "end_index",
            "url",
            "title",
            "live",
        }

    def test_live_field_defaults_to_none(self):
        ann = CitationAnnotation(start_index=0, end_index=5, url="https://a")
        assert ann.live is None

    def test_live_field_accepts_true_false_none(self):
        for value in (None, True, False):
            ann = CitationAnnotation(
                start_index=0, end_index=5, url="https://a", live=value
            )
            assert ann.live is value

    def test_to_dict_excludes_none_live(self):
        ann = CitationAnnotation(start_index=0, end_index=5, url="https://a")
        data = ann.to_dict()
        assert "live" not in data

    def test_to_dict_includes_live_true(self):
        ann = CitationAnnotation(start_index=0, end_index=5, url="https://a", live=True)
        data = ann.to_dict()
        assert data.get("live") is True

    def test_to_dict_includes_live_false(self):
        ann = CitationAnnotation(start_index=0, end_index=5, url="https://a", live=False)
        data = ann.to_dict()
        assert data.get("live") is False


class TestProtocolCitationsField:
    def test_default_empty_list(self):
        protocol = _build_minimal_protocol()
        assert protocol.citations == []

    def test_accepts_citation_list(self):
        annotations = [
            CitationAnnotation(start_index=0, end_index=5, url="https://a"),
            CitationAnnotation(start_index=10, end_index=20, url="https://b", title="B"),
        ]
        protocol = _build_minimal_protocol(citations=annotations)
        assert len(protocol.citations) == 2
        assert protocol.citations[0].url == "https://a"

    def test_to_dict_includes_citations_key(self):
        protocol = _build_minimal_protocol(
            citations=[
                CitationAnnotation(start_index=0, end_index=5, url="https://a"),
            ]
        )
        data = protocol.to_dict()
        assert "citations" in data
        assert data["citations"] == [
            {"type": "url_citation", "start_index": 0, "end_index": 5, "url": "https://a"}
        ]

    def test_to_dict_citations_key_present_when_empty(self):
        protocol = _build_minimal_protocol()
        data = protocol.to_dict()
        # citations is materialised even when empty so downstream consumers
        # don't have to special-case missing-vs-empty.
        assert data.get("citations") == []
