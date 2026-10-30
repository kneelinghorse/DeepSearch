"""S101-m03: a dead link a collected source names is dead, not fabricated.

A cited URL that returns a dead probe and was never collected was classified
HALLUCINATED, so its citations were quarantined and the critique was told the
URL was invented. When the exact cited string occurs at a URL boundary in the
body of a source the run collected and read, the URL is a real, dead link: the
paper names it. Such a citation is now DEAD_UNCOLLECTED. It survives, flagged
unverified like any dead citation, and is still an orphan and a missing source.
No URL is normalized, so miscopies and www/apex twins stay HALLUCINATED.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from deepsearch.agent import citation_liveness, critique
from deepsearch.agent import codeagent_finalize_node as fnode
from deepsearch.agent.citation_liveness import (
    CLASSIFICATION_DEAD,
    CLASSIFICATION_DEAD_UNCOLLECTED,
    CLASSIFICATION_HALLUCINATED,
    CLASSIFICATION_UNKNOWN,
    _named_at_url_boundary,
    build_source_url_set,
    check_citations_liveness,
    classify_citation,
    quarantine_dead_citations,
    source_named_dead_urls,
)
from deepsearch.agent.codeagent_runner import SynthesisResult
from deepsearch.agent.critique import CritiqueResult
from deepsearch.report.citation import CitationAnnotation

ROOT = Path(__file__).resolve().parents[2]
NAMED = "https://github.com/example/artifact-repo"
INVENTED = "https://invented.example/fake-page"
PAPER = "https://paper.example/html/2606.06240"


def citation(url: str, live: bool | None = False) -> CitationAnnotation:
    return CitationAnnotation(start_index=0, end_index=5, url=url, live=live)


def readable(url: str, body: str, **fields) -> dict:
    return {"url": url, "body": body, "content_quality": "readable_body",
            "body_source": "source_fetch", **fields}


@pytest.mark.parametrize(("body", "named"), [
    (f"Code, data and artifact:\n{NAMED}\n1. Introduction", True),
    (f"see ({NAMED}).", True),
    (f"**{NAMED}**", True),
    (f'"{NAMED}", the repository', True),
    (NAMED, True),
    (f"{NAMED}s are listed", False),  # followed by a letter: a safe miss
    (f"x{NAMED} y", False),
    (f"{NAMED}/tree/main", False),
    (f"{NAMED}.git", False),
])
def test_the_url_boundary(body: str, named: bool) -> None:
    assert _named_at_url_boundary(body, NAMED) is named


def test_a_url_with_an_open_parenthesis_is_not_closed_by_one() -> None:
    # M1's shape: a URL truncated before its closing ")" is a prefix of the
    # collected URL, never a boundary occurrence of itself.
    cut = "http://repo.example/publications/video-art-19681990(b9b0c243"
    assert not _named_at_url_boundary(f"see {cut}).html now", cut)
    assert not _named_at_url_boundary("https://w.example/Foo_(bar) ", "https://w.example/Foo_(bar")
    assert _named_at_url_boundary("https://w.example/Foo_(bar) ", "https://w.example/Foo_(bar)")


def test_a_named_dead_uncollected_url_is_dead_uncollected() -> None:
    sources = [readable(PAPER, f"Artifact: {NAMED}\nMore text.")]
    citations = [citation(NAMED), citation(INVENTED)]
    named = source_named_dead_urls(citations, sources)
    assert named == {NAMED}
    source_set = build_source_url_set(sources)
    assert classify_citation(live=False, url=NAMED, source_url_set=source_set,
                             source_named_urls=named) == CLASSIFICATION_DEAD_UNCOLLECTED
    assert classify_citation(live=False, url=INVENTED, source_url_set=source_set,
                             source_named_urls=named) == CLASSIFICATION_HALLUCINATED
    # Without the set, today's class.
    assert classify_citation(live=False, url=NAMED, source_url_set=source_set) == (
        CLASSIFICATION_HALLUCINATED)
    kept, quarantined = quarantine_dead_citations(citations, sources)
    assert ([a.url for a in kept], [a.url for a in quarantined]) == ([NAMED], [INVENTED])


@pytest.mark.parametrize(("citations", "sources", "expected"), [
    # A search snippet does not qualify.
    ([citation(NAMED)], [{"url": PAPER, "snippet": f"Artifact: {NAMED}"}],
     CLASSIFICATION_HALLUCINATED),
    # Nor does a body that is not a readable fetched body.
    ([citation(NAMED)], [readable(PAPER, f"Artifact: {NAMED}", content_quality="snippet")],
     CLASSIFICATION_HALLUCINATED),
    ([citation(NAMED)], [readable(PAPER, f"Artifact: {NAMED}", fetched_gone=True)],
     CLASSIFICATION_HALLUCINATED),
    # The body names the www twin; apex and www are distinct origins.
    ([citation(NAMED)], [readable(PAPER, f"Artifact: {NAMED.replace('://', '://www.')}")],
     CLASSIFICATION_HALLUCINATED),
    # The body names a longer URL the cited one is a prefix of.
    ([citation(NAMED)], [readable(PAPER, f"Artifact: {NAMED}-v2 here")],
     CLASSIFICATION_HALLUCINATED),
    # An unchecked citation is unknown, named or not.
    ([citation(NAMED, live=None)], [readable(PAPER, f"Artifact: {NAMED}")], CLASSIFICATION_UNKNOWN),
    # A dead URL that was collected is dead, as before.
    ([citation(PAPER)], [readable(PAPER, f"Its own URL: {PAPER}")], CLASSIFICATION_DEAD),
], ids=["snippet", "not readable", "fetched gone", "www twin", "prefix", "live None",
        "collected"])
def test_negative_controls_keep_todays_class(citations, sources, expected) -> None:
    named = source_named_dead_urls(citations, sources)
    annotation = citations[0]
    assert classify_citation(live=annotation.live, url=annotation.url,
                             source_url_set=build_source_url_set(sources),
                             source_named_urls=named) == expected
    assert NAMED not in named


def test_the_url_as_the_agent_wrote_it_is_matched() -> None:
    # Canonicalization adds a slash to a bare host; the paper prints it without.
    _, extracted = fnode._extract_and_resolve_citations(
        "Project page [cite:https://toki-memory.github.io] holds.",
        sources_collected=[], synthesis_telemetry={},
    )
    (written,) = extracted
    assert (written.url, written._cited_url) == (
        "https://toki-memory.github.io/", "https://toki-memory.github.io")
    written.live = False
    sources = [readable(PAPER, "Project page: https://toki-memory.github.io\nMore.")]
    assert source_named_dead_urls([written], sources) == {"https://toki-memory.github.io/"}
    canonical_only = citation("https://toki-memory.github.io/")
    assert source_named_dead_urls([canonical_only], sources) == set()


@pytest.mark.parametrize(("cited", "body", "expected"), [
    ("https://x.org/a)", "see (https://x.org/a). Next", CLASSIFICATION_HALLUCINATED),
    ("https://x.org/a,", "see https://x.org/a, and", CLASSIFICATION_HALLUCINATED),
    ("https://x.org/a.", "see https://x.org/a. Next", CLASSIFICATION_HALLUCINATED),
    ("https://x.org/a_", "see https://x.org/a_ end", CLASSIFICATION_HALLUCINATED),
    ("https://w.example/Foo_(bar)", "see https://w.example/Foo_(bar) now",
     CLASSIFICATION_DEAD_UNCOLLECTED),
], ids=["stray paren", "comma", "period", "underscore", "balanced paren"])
def test_a_cited_url_ending_in_a_stray_closer_is_a_miscopy(cited, body, expected) -> None:
    sources = [readable(PAPER, body)]
    named = source_named_dead_urls([citation(cited)], sources)
    assert classify_citation(live=False, url=cited, source_url_set=build_source_url_set(sources),
                             source_named_urls=named) == expected


def test_a_named_urls_fragment_and_tracking_twins_share_its_class() -> None:
    twins = [f"{NAMED}#readme", f"{NAMED}?utm_source=feed", NAMED.replace("https", "HTTPS")]
    sources = [readable(PAPER, f"Artifact: {NAMED}\nMore text.")]
    citations = [citation(url) for url in (*twins, NAMED)]
    assert source_named_dead_urls(citations, sources) == {*twins, NAMED}
    # Only twins of a named URL: an unnamed URL's twin stays hallucinated.
    assert source_named_dead_urls([citation(f"{INVENTED}#x")], sources) == set()
    # And www is no twin of the apex.
    www = NAMED.replace("://", "://www.")
    assert www not in source_named_dead_urls([citation(www), citation(NAMED)], sources)


def test_the_critique_hears_dead_not_fabricated_and_rule_7_is_unchanged() -> None:
    sources = [readable(PAPER, f"Artifact: {NAMED}\nMore text.")]
    citations = [citation(NAMED), citation(INVENTED)]
    block = critique._format_citation_liveness_block(
        citations, build_source_url_set(sources), source_named_dead_urls(citations, sources))
    named_line, invented_line = block.split("\n\n")[2].split("\n")
    assert named_line == (
        f"- [cite:{NAMED}]  →  dead (not in sources_collected, but a collected source names "
        "this exact URL; GET-confirmed gone — not fabricated)")
    assert invented_line.endswith("hallucinated (not in sources AND GET-confirmed unreachable "
                                  "— fabricated)")
    assert block.endswith("Apply rule 7 to each hallucinated citation; preserve the rest.")
    assert "keep dead, orphan, live, and unknown citations' claims as-is" in block


def test_its_claims_keep_the_incomplete_grounding_safeguard() -> None:
    sources = [readable(PAPER, f"Artifact: {NAMED}\nMore text.")]
    citations = [citation(NAMED), citation(INVENTED)]
    assert critique._hallucinated_citation_urls(citations, sources) == {INVENTED}
    # Uncollected, so never a dead-with-empty-body URL either.
    assert critique._dead_empty_body_citation_urls(
        citations, [*sources, {"url": "https://empty.example/x"}]) == set()


def test_the_gate_counts_it_an_orphan_not_hallucinated_or_dead(monkeypatch) -> None:
    def probe(url: str, timeout_s: float = 5.0) -> dict:
        return {"alive": url == PAPER, "status_code": 200 if url == PAPER else 404,
                "final_url": url}

    monkeypatch.setattr(citation_liveness, "url_liveness", probe)
    monkeypatch.setattr(citation_liveness, "_RETRY_BACKOFF_S", 0.0)
    sources = [readable(PAPER, f"Artifact: {NAMED}\n" + "Body text. " * 40)]
    citations = [citation(NAMED, live=None), citation(INVENTED, live=None),
                 citation(PAPER, live=None)]
    _, telemetry = asyncio.run(check_citations_liveness(citations, sources))
    assert (telemetry.orphan, telemetry.dead, telemetry.hallucinated_url_count) == (2, 0, 1)


def test_every_production_classifier_call_passes_the_named_set() -> None:
    calls = []
    for path in sorted((ROOT / "deepsearch").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(
                    node.func, "id", getattr(node.func, "attr", None)) == "classify_citation":
                keywords = {keyword.arg for keyword in node.keywords}
                calls.append((path.relative_to(ROOT).as_posix(), node.lineno,
                              "source_named_urls" in keywords))
    assert len(calls) == 7, calls
    assert [call for call in calls if not call[2]] == []


def _synthesis(paper_body: str, extra: str = "") -> SynthesisResult:
    body = (
        f"The paper's artifact repository returns 404 [cite:{NAMED}] and more words here.\n"
        f"An invented page backs this claim [cite:{INVENTED}] and more words here.\n"
        f"The collected paper backs this one [cite:{PAPER}] and more words here.\n{extra}"
    )
    references = f"[1] Artifact — {NAMED}\n[2] Invented — {INVENTED}\n[3] Paper — {PAPER}\n"
    return SynthesisResult(
        output_markdown=f"# Report\n\n{body}\n## References\n\n{references}",
        step_count=5, self_terminated=True, hit_max_steps=False, runtime_seconds=10.0,
        token_usage={"input": 100, "output": 100, "total": 200},
        tool_call_summary={"total_calls": 3},
        authority_tier_distribution={"high": 0, "medium": 0, "low": 0},
        pool_size=0, step_trace=[{"step_index": 0, "type": "ActionStep"}],
        sources_collected=[{**readable(PAPER, paper_body), "title": "Paper", "snippet": "s"}],
        error=None,
    )


def _finalize(tmp_path: Path, monkeypatch, paper_body: str,
              synthesis: SynthesisResult | None = None) -> tuple[dict, dict]:
    monkeypatch.setenv("DEEPSEARCH_REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(citation_liveness, "_RETRY_BACKOFF_S", 0.0)
    captured: dict = {}

    def probe(url: str, **kwargs) -> dict:
        return {"alive": False, "status_code": 404, "final_url": url}

    def capture_protocol(protocol, **kwargs):
        captured["citations"] = list(protocol.citations)
        return str(tmp_path / "protocol.json")

    def capture_report(content, **kwargs):
        captured["report"] = content
        return str(tmp_path / "report.md")

    passthrough = CritiqueResult(output_markdown="", claim_count=3, gap_notes_inserted=0,
                                 runtime_seconds=0.0, token_usage_complete=True)
    state = {
        "mission_id": "S101-M03", "mission_objectives": ["c1"],
        "mission_context": {"title": "Named dead", "objective": "o", "deliverables": []},
        "sources_found": [], "telemetry": {},
    }

    async def run():
        with patch.object(fnode, "run_codeagent_synthesis",
                          return_value=synthesis or _synthesis(paper_body)), \
                patch.object(fnode, "run_critique", return_value=passthrough), \
                patch.object(citation_liveness, "url_liveness", new=probe), \
                patch.object(fnode, "save_protocol", side_effect=capture_protocol), \
                patch.object(fnode, "save_report", side_effect=capture_report):
            return await fnode.codeagent_finalize_node(state)

    return asyncio.run(run()), captured


def test_finalize_keeps_a_source_named_dead_citation_flagged_unverified(
    tmp_path, monkeypatch,
) -> None:
    filler = "Supporting body text. " * 40
    baseline, _ = _finalize(tmp_path / "before", monkeypatch, f"No repository link.\n{filler}")
    result, captured = _finalize(tmp_path / "after", monkeypatch,
                                 f"Code and data: {NAMED}\n{filler}")
    before, after = baseline["synthesis_telemetry"], result["synthesis_telemetry"]
    # Today, unnamed: quarantined as fabricated with the invented control.
    assert (before["hallucinated_quarantined"], before["markers_quarantined"]) == (2, 0)
    # Named by the collected paper: kept, live=False, flagged unverified.
    shipped = {c.url: c.live for c in captured["citations"]}
    assert shipped == {NAMED: False, PAPER: True}
    report = captured["report"]
    lines = report.split("\n")
    assert f"[1] Artifact — {NAMED}{fnode._UNVERIFIED_REF_SUFFIX}" in lines
    assert f"[2] Invented — {INVENTED}{fnode._WITHDRAWN_REF_SUFFIX}" in lines
    assert (after["markers_quarantined"], after["source_named_dead"],
            after["hallucinated_quarantined"]) == (1, 1, 1)
    # Still an orphan and still not a dead collected source.
    for key in ("orphan", "dead", "checked"):
        assert after["liveness_telemetry"][key] == before["liveness_telemetry"][key], key
    assert (after["liveness_telemetry"]["hallucinated_url_count"],
            before["liveness_telemetry"]["hallucinated_url_count"]) == (1, 2)
    # The outcome gate's dead_rate is dead / checked.
    rates = [telemetry["liveness_telemetry"]["dead"] / telemetry["liveness_telemetry"]["checked"]
             for telemetry in (before, after)]
    assert rates[0] == rates[1]
    # It now counts as a missing, unread source: two new, true warnings.
    gained = set(result["quality_gate_failures"]) - set(baseline["quality_gate_failures"])
    lost = set(baseline["quality_gate_failures"]) - set(result["quality_gate_failures"])
    assert gained == {"missing_cited_sources_1", "cited_without_readable_body_1",
                      "markers_quarantined_1", "hallucinated_quarantined_1"}
    assert lost == {"hallucinated_quarantined_2"}
    assert result["delivery_quality"] == baseline["delivery_quality"]
    # With m02, the kept citation also shows and maps its reader number.
    assert "returns 404[1] and more words here." in report
    assert after["reader_number_rendering"]["fates"]["mapped"] == 2


@pytest.mark.parametrize("twin_first", [True, False], ids=["twin first", "plain first"])
def test_a_named_url_and_its_twin_ship_one_class_end_to_end(
    tmp_path, monkeypatch, twin_first: bool,
) -> None:
    twin = f"{NAMED}#readme"
    extra = f"Its README shows the same 404 [cite:{twin}] and more words here.\n"
    paper = f"Code and data: {NAMED}\n" + "Supporting body text. " * 40
    synthesis = _synthesis(paper, extra)
    if twin_first:
        lines = synthesis.output_markdown.split("\n")
        lines.insert(2, lines.pop(lines.index(extra.rstrip("\n"))))
        synthesis.output_markdown = "\n".join(lines)
    claims = synthesis.output_markdown.split("\n")
    assert claims[2].startswith("Its README" if twin_first else "The paper's")
    result, captured = _finalize(tmp_path, monkeypatch, paper, synthesis)
    telemetry = result["synthesis_telemetry"]
    assert {c.url for c in captured["citations"]} >= {NAMED, twin}
    assert f"[1] Artifact — {NAMED}{fnode._UNVERIFIED_REF_SUFFIX}" in captured["report"].split("\n")
    assert (telemetry["hallucinated_quarantined"], telemetry["source_named_dead"],
            telemetry["liveness_telemetry"]["hallucinated_url_count"]) == (1, 2, 1)
