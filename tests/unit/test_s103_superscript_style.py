"""S103-m02: a superscript is a citation only where the report cites that way.

The agent is asked for ``[1]``-style inline numbers and ``[N] title — url`` entries
(contract_to_prompt.py). A report that labels its References entries that way never has
a superscript read as a reader marker unless superscripts are its only citations, so
``5×10⁻⁶``, ``p<10⁻³³`` and ``χ²`` stay maths: numbered fallback no longer turns them into
citations of the wrong source, and the References audit no longer counts them as dangling
reader markers. A report that labels entries with superscripts (``¹ Title — URL``) reads
them as before. In every report a digit run after ``⁺`` or ``⁻`` is an exponent.

One helper, ``reads_superscripts``, decides. Finalize asks it once, on the text the
validation pass extracts, and carries the answer to both References audits and the fates
count, whose texts have no ``[cite:URL]`` marker left to decide on. ``_claim_end`` now
walks back over the trailing marker cluster instead of searching the whole prefix; it
equals the old regex search, kept here as an oracle with the sign rule added.
"""

from __future__ import annotations

import asyncio
import random
import re
from unittest.mock import patch

import pytest

from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.citation_extraction import (
    _claim_end,
    extract_citations,
    is_cited,
    reads_superscripts,
)
from deepsearch.agent.citation_liveness import LivenessTelemetry
from deepsearch.agent.codeagent_runner import SynthesisResult, _recover_stranded_draft
from deepsearch.agent.critique import CritiqueResult
from deepsearch.agent.quality_series import _citation_surface
from deepsearch.report.citation import CitationAnnotation

URL_A = "https://a.example/one"
URL_B = "https://b.example/two"
URL_C = "https://c.example/three"
URL_D = "https://d.example/four"
BRACKET_REFERENCES = f"\n## References\n\n[1] Alpha — {URL_A}\n[2] Beta — {URL_B}\n"
SUPERSCRIPT_REFERENCES = (
    "\n## References\n\n"
    "¹ First — https://s.example/1\n² Second — https://s.example/2\n"
    "⁶ Sixth — https://s.example/6\n¹⁰ Tenth — https://s.example/10\n"
)
EXPONENTS = ("χ²", "σ²", "x²", "km²", "R²", "10⁶", "5×10⁻⁶", "p<10⁻³³")

SUPER = "⁰¹²³⁴⁵⁶⁷⁸⁹"
# Today's _claim_end regex, kept as the oracle, with the sign rule added.
ORACLE_READ = re.compile(r"(?:\s*(?:\[\d+\]|(?<![A-Z" + SUPER + r"⁺⁻])[" + SUPER + r"]+))+$")
ORACLE_UNREAD = re.compile(r"(?:\s*\[\d+\])+$")


def audit(text: str, annotations: list[CitationAnnotation],
          superscripts: bool | None = None) -> dict:
    return fnode._annotate_references_and_check_consistency(
        text, kept_annotations=annotations, downgraded_dead=[], quarantined=[],
        citation_reference_markdown=text, require_number_mapping=True,
        superscripts=superscripts,
    )[1]


def oracle_claim_end(markdown: str, end: int, *, superscripts: bool) -> int:
    match = (ORACLE_READ if superscripts else ORACLE_UNREAD).search(markdown[:end])
    return len(markdown[:match.start() if match else end].rstrip())


@pytest.mark.parametrize("cited", ["[1]", f"[cite:{URL_A}]"])
@pytest.mark.parametrize("exponent", EXPONENTS)
def test_exponents_in_a_bracket_style_report_are_maths(exponent: str, cited: str) -> None:
    text = f"The fit gives {exponent} overall {cited}.\n" + BRACKET_REFERENCES
    assert not reads_superscripts(text)
    counts: dict = {}
    stripped, annotations = extract_citations(text, telemetry=counts)
    # Only the claim's own citation; no number is read from the exponent.
    assert [a.url for a in annotations] == [URL_A]
    assert counts["unresolved_numbered_markers"] == 0
    consistency = audit(stripped, annotations, superscripts=reads_superscripts(text))
    assert consistency["reader_markers"] == stripped.split("## References")[0].count("[1]")
    assert consistency["dangling_reader_numbers"] == []
    assert (consistency["superscripts_read"], consistency["superscript_runs_ignored"]) == (
        False, 1)


def test_a_report_cited_only_by_superscripts_keeps_its_citations(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))
    draft = ("# Report\n\nPKCE adds a code challenge¹. The verifier is hashed². "
             "The server compares them³.\n\n## References\n\n"
             f"[1] RFC 7636 — {URL_A}\n[2] Hashing — {URL_B}\n[3] Compare — {URL_C}\n")
    assert reads_superscripts(draft)
    _, annotations = extract_citations(draft)
    assert [(a.start_index, a.end_index, a.url, a._extraction_mode) for a in annotations] == [
        (10, 36, URL_A, "numbered"), (39, 61, URL_B, "numbered"), (64, 88, URL_C, "numbered")]
    # Generation recovery and stranded-draft salvage accept it as cited.
    assert is_cited(draft)
    assert _recover_stranded_draft({"report_draft": draft}) == draft
    result, _ = finalize(draft, [URL_A, URL_B, URL_C])
    telemetry = result["synthesis_telemetry"]
    assert telemetry["citation_telemetry"]["citations_extracted"] == 3
    # The gate sees three citations, not zero.
    assert result["delivery_quality"] == "usable_with_warning"
    assert not [f for f in result["quality_gate_failures"] if f.startswith("zero_citations")]
    consistency = telemetry["citation_consistency"]
    assert (consistency["superscripts_read"], consistency["reader_markers"],
            consistency["superscript_runs_ignored"], consistency["consistent"]) == (
        True, 3, 0, True)


@pytest.mark.parametrize(("claim", "numbers"), [
    ("A claim¹.", [1]), ("A claim.¹", [1]), ("A claim¹ ².", [1, 2]), ("A claim¹⁰.", [10]),
    ("A rate of 10⁻⁶.", []), ("Growth of e⁺².", []), ("The variable T¹⁰.", []),
])
def test_a_superscript_style_report_reads_them_as_before(claim: str, numbers: list[int]) -> None:
    text = f"# Report\n\n{claim}\n" + SUPERSCRIPT_REFERENCES
    assert reads_superscripts(text)
    stripped, annotations = extract_citations(text)
    assert sorted(a.url for a in annotations) == [f"https://s.example/{n}" for n in numbers]
    consistency = audit(stripped, annotations)
    assert consistency["superscripts_read"] is True
    assert consistency["reader_markers"] == len(numbers)
    assert consistency["superscript_runs_ignored"] == (0 if numbers else 1)


def test_only_a_label_outside_code_switches_superscripts_on() -> None:
    body = "The fit gives χ² overall [1].\n"
    fenced = body + "\n```text\n² Beta — https://b.example/two\n```\n" + BRACKET_REFERENCES
    assert not reads_superscripts(fenced)
    assert [a.url for a in extract_citations(fenced)[1]] == [URL_A]
    labelled = body + BRACKET_REFERENCES + f"² Beta again — {URL_B}\n"
    assert reads_superscripts(labelled)
    assert sorted(a.url for a in extract_citations(labelled)[1]) == [URL_A, URL_B]


def test_the_audit_uses_the_carried_reading_not_its_stripped_text() -> None:
    # A marked draft cited only by [cite:URL] markers, with ``CMH χ²`` at a
    # citation boundary and an entry [2] naming another URL.
    marked = (f"The test gives CMH χ²[cite:{URL_A}]. Another claim holds[cite:{URL_A}].\n"
              + BRACKET_REFERENCES)
    carried = reads_superscripts(marked)
    assert carried is False
    stripped, annotations = extract_citations(marked)
    chi = annotations[0]
    assert stripped[chi.end_index - 2:chi.end_index] == "χ²"
    # The stripped text has no [cite:URL] left, so deciding there reads ² and
    # binds the citation to entry [2].
    assert reads_superscripts(stripped)
    assert fnode._boundary_numbers(stripped, chi, superscripts=True) == {2}
    assert fnode._boundary_numbers(stripped, chi, superscripts=False) == set()
    kept = audit(stripped, annotations, superscripts=carried)
    assert (kept["superscripts_read"], kept["reader_markers"],
            kept["superscript_runs_ignored"]) == (False, 0, 1)
    # Direct callers that pass no reading decide on the audited text, as before.
    decided = audit(stripped, annotations)
    assert (decided["superscripts_read"], decided["reader_markers"],
            decided["superscript_runs_ignored"]) == (True, 1, 0)


def test_a_report_with_no_references_stays_inconsistent_through_protocol_only() -> None:
    text = f"A claim¹[cite:{URL_A}] stands.\n"
    assert not reads_superscripts(text)
    stripped, annotations = extract_citations(text)
    consistency = audit(stripped, annotations, superscripts=False)
    assert (consistency["protocol_only"], consistency["reader_markers"],
            consistency["superscript_runs_ignored"], consistency["consistent"]) == (1, 0, 1, False)
    assert _citation_surface({"citation_consistency": consistency})["actionable"]


def test_a_citation_ending_in_an_exponent_gets_its_reader_number() -> None:
    draft = f"The test gives CMH χ²[cite:{URL_A}] and more.\n" + BRACKET_REFERENCES
    rendered, telemetry, keys = fnode._render_reader_numbers(draft)
    # Before m02 the ² bound the citation to entry [2], so it was skipped as a
    # conflicting number; now the number goes after the exponent.
    assert f"CMH χ²[1][cite:{URL_A}] and more." in rendered
    assert (telemetry["rendered"], telemetry["skipped_conflicting_number"]) == (1, 0)
    assert list(keys.values()) == [1]


def test_claim_end_equals_the_oracle_on_generated_clusters() -> None:
    alphabet = list("[]0123456789 \n\tabcAZT.") + list(SUPER) + ["⁺", "⁻", " ", "٣", "χ"]
    generator = random.Random(103)
    for _ in range(20000):
        text = "".join(generator.choice(alphabet) for _ in range(generator.randint(0, 14)))
        for end in range(len(text) + 1):
            for superscripts in (True, False):
                assert _claim_end(text, end, superscripts=superscripts) == oracle_claim_end(
                    text, end, superscripts=superscripts), (text, end, superscripts)


def test_claim_end_without_superscripts_is_the_old_regex_on_text_with_none() -> None:
    old = re.compile(r"(?:\s*(?:\[\d+\]|(?<![A-Z" + SUPER + r"])[" + SUPER + r"]+))+$")
    generator = random.Random(1031)
    alphabet = list("[]0123456789 \n\tabcAZT.χ")
    for _ in range(5000):
        text = "".join(generator.choice(alphabet) for _ in range(generator.randint(0, 16)))
        for end in range(len(text) + 1):
            match = old.search(text[:end])
            assert _claim_end(text, end, superscripts=False) == len(
                text[:match.start() if match else end].rstrip())


# A superscript-style report through finalize: extraction, both audits, rendering and
# fates read its superscripts the same way.
STYLE_DRAFT = (
    "# Report\n\nFirst claim¹[cite:https://s.example/1]. Second claim stands². "
    f"Third claim holds [cite:{URL_C}]. Fourth claim² [cite:{URL_D}] too. "
    "A rate of 10⁻⁶ holds.\n\n## References\n\n"
    f"¹ First — https://s.example/1\n² Second — https://s.example/2\n³ Third — {URL_C}\n"
    f"⁴ Fourth — {URL_D}\n"
)


def finalize(draft: str, urls: list[str]) -> tuple[dict, dict]:
    """Run finalize on one draft with an identity critique; capture each audit's reading."""

    captured: dict = {"audits": [], "fates": []}
    audit_pass = fnode._annotate_references_and_check_consistency
    fates_count = fnode._reader_number_fates

    def recorded_audit(*args, **kwargs):
        result = audit_pass(*args, **kwargs)
        captured["audits"].append((kwargs.get("superscripts"), result[1]))
        return result

    def recorded_fates(**kwargs):
        captured["fates"].append(kwargs.get("superscripts"))
        return fates_count(**kwargs)

    def critique(draft_markdown, sources_collected, mission_context, *, citations=None,
                 scope_admission=None) -> CritiqueResult:
        return CritiqueResult(output_markdown=draft_markdown, claim_count=0,
                              gap_notes_inserted=0, runtime_seconds=0.0)

    async def liveness(citations, sources_collected, budget_seconds=45.0, scope_admission=None):
        for annotation in citations:
            annotation.live = True
        return list(citations), LivenessTelemetry(
            checked=len(citations), live=len(citations), dead=0, orphan=0, unknown=0,
            hallucinated_url_count=0, distinct_urls_checked=len({a.url for a in citations}),
        )

    synthesis = SynthesisResult(
        output_markdown=draft, step_count=5, self_terminated=True, hit_max_steps=False,
        runtime_seconds=10.0, token_usage={"input": 100, "output": 100, "total": 200},
        tool_call_summary={"total_calls": 3},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0}, pool_size=0,
        step_trace=[{"step_index": 0, "type": "ActionStep"}],
        sources_collected=[{"url": url, "title": url, "snippet": "x"} for url in urls],
        error=None,
    )
    state = {
        "mission_id": "S103-M02-STYLE", "mission_objectives": ["c1"],
        "mission_context": {"title": "Style", "objective": "o", "deliverables": []},
        "sources_found": [], "telemetry": {},
    }

    async def run():
        with patch.object(fnode, "run_codeagent_synthesis", return_value=synthesis), \
                patch.object(fnode, "run_critique", side_effect=critique), \
                patch.object(fnode, "check_citations_liveness", new=liveness), \
                patch.object(fnode, "_annotate_references_and_check_consistency",
                             side_effect=recorded_audit), \
                patch.object(fnode, "_reader_number_fates", side_effect=recorded_fates):
            return await fnode.codeagent_finalize_node(state)

    return asyncio.run(run()), captured


def test_a_superscript_style_report_is_read_one_way_through_finalize(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))
    urls = ["https://s.example/1", "https://s.example/2", URL_C, URL_D]
    result, captured = finalize(STYLE_DRAFT, urls)
    telemetry = result["synthesis_telemetry"]
    # Both audits got the carried reading, and so did the fates count.
    assert [reading for reading, _ in captured["audits"]] == [True, True]
    assert [result["superscripts_read"] for _, result in captured["audits"]] == [True, True]
    assert captured["fates"] == [True]
    # Rendering read the ² before URL_D as that claim's number, a conflicting one,
    # and gave the unnumbered third claim its entry's number.
    rendering = telemetry["reader_number_rendering"]
    assert (rendering["rendered"], rendering["skipped_conflicting_number"]) == (1, 1)
    assert rendering["fates"]["mapped"] == 1
    # Both audits counted ¹, both ² and the rendered [3], but not the ⁶ after ⁻.
    for _, consistency in captured["audits"]:
        assert (consistency["reader_markers"], consistency["superscript_runs_ignored"]) == (4, 1)
    # Extraction cited each bare ² as entry ²; the three [cite:URL] markers stay primary.
    citation_telemetry = telemetry["citation_telemetry"]
    assert (citation_telemetry["citations_extracted"],
            citation_telemetry["persisted_provenance"]) == (5, {"primary": 3, "numbered": 2})
