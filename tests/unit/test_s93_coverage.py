"""Format repairs must save paid recovery without concealing real content gaps."""

from __future__ import annotations


import pytest

from deepsearch.agent.critique import CritiqueResult
from deepsearch.agent.quality_series import build_quality_record
from deepsearch.agent.recovery import decide_recovery
from deepsearch.agent.tools.completeness_check import completeness_check

def _healthy_critique() -> CritiqueResult:
    return CritiqueResult(
        output_markdown="report", claim_count=1, gap_notes_inserted=0,
        runtime_seconds=0, token_usage={},
    )


CITED = "A documented finding supported by https://example.org/evidence."




@pytest.mark.parametrize("markdown", ["", "# Markdown report\n", "# Report\nTODO",
    "# Report\n\n## References\n[1] https://example.org/"])
def test_format_label_cannot_hide_an_empty_artifact(markdown: str) -> None:
    coverage = completeness_check(markdown, ["Markdown report"])
    assert coverage["missing"] == ["Markdown report"]
    assert coverage["shortfall_score"] == 1.0
    assert decide_recovery(_healthy_critique(), coverage).triggered


@pytest.mark.parametrize("tail", ["with an options comparison table", "including risk analysis"])
def test_attached_content_remains_required(tail: str) -> None:
    coverage = completeness_check("## Findings\n" + CITED, ["Markdown report " + tail])
    assert coverage["artifact_requirements"][0]["satisfied"] is True
    assert len(coverage["missing"]) == 1
    assert coverage["shortfall_score"] == 0.5
    assert decide_recovery(_healthy_critique(), coverage).triggered


def test_absent_evidence_is_a_real_gap_but_presence_is_not_claim_validation() -> None:
    spec = ["Markdown research report with citations resolving to evidence"]
    prose = "## Findings\n" + "A meaningful finding requires its own evidence. " * 10
    missing = completeness_check(prose, spec)
    assert missing["missing"] == ["citations resolving to evidence"]
    assert decide_recovery(_healthy_critique(), missing).triggered
    present = completeness_check(
        prose + "Supported claim [1].\n\n## References\n[1] https://example.org", spec
    )
    assert present["missing"] == []
    assert present["content_requirements"][0]["resolution_status"] == "unassessed"


def test_metadata_and_footer_cannot_supply_content_or_evidence() -> None:
    wrapped = (
        "# Research Report: M - title\n\n**Sources collected:** 99\n"
        "**Verified references:** 99\n**Agent steps:** 30 / 40\n**Runtime:** 10m\n\n"
        "## Findings\nTODO\n\n---\n\n**Mission Objectives:**\n"
        "## Recommendations\n" + CITED
    )
    coverage = completeness_check(wrapped, ["Markdown report with citations", "Recommendations"])
    assert coverage["shortfall_score"] == 1.0


@pytest.mark.parametrize("spec", [None, [], [None, 42], ["", "   "], "Markdown report"])
def test_unassessed_is_not_full_coverage_or_a_paid_repair_target(spec: object) -> None:
    coverage = completeness_check("## Findings\n" + CITED, spec)  # type: ignore[arg-type]
    assert coverage["assessment_status"] == "unassessed"
    assert coverage["shortfall_score"] is None
    assert not decide_recovery(_healthy_critique(), coverage).triggered
    record = build_quality_record({"deliverable_coverage": coverage}, 100)
    assert record["schema_version"] == 6
    assert record["coverage"]["assessment_status"] == "unassessed"
    assert record["coverage"]["shortfall_score"] is None


def test_success_criteria_and_content_topics_still_use_substantive_sections() -> None:
    coverage = completeness_check("## Findings\n" + CITED,
        ["Risk analysis covering operational failures", "Report generation risks"])
    assert len(coverage["missing"]) == 2
    assert coverage["artifact_requirements"] == []
    assert coverage["shortfall_score"] == 1.0
