"""S98-m02: one arXiv paper identity for collection membership, and nowhere else.

A run that collected ``doi.org/10.48550/arXiv.<id>`` and cited ``arxiv.org/abs/<id>``
read the same paper; the census counted 8 such citations as orphans (decision 1002).
Membership answers "did the run collect this paper". Readability, fetch failures,
the probe cache and the References audit keep exact URLs: an abstract page is not
the PDF's full text, and missing evidence is never upgraded (decision 967).
"""

from __future__ import annotations

from typing import Any

import pytest

from deepsearch.agent import citation_liveness
from deepsearch.agent.citation_liveness import (
    CLASSIFICATION_DEAD,
    CLASSIFICATION_HALLUCINATED,
    CLASSIFICATION_LIVE,
    CLASSIFICATION_ORPHAN,
    _build_substantive_body_url_set,
    _normalise_url_for_cache,
    build_source_url_set,
    check_citations_liveness,
    classify_citation,
    strip_dead_cite_markers,
)
from deepsearch.agent.codeagent_finalize_node import _reference_url_identity
from deepsearch.agent.quality_series import build_evidence_dependencies
from deepsearch.agent.url_utils import arxiv_identity
from deepsearch.report.citation import CitationAnnotation

ABS = "https://arxiv.org/abs/2605.29313"
DOI = "https://doi.org/10.48550/arxiv.2605.29313"


@pytest.mark.parametrize(("url", "identity"), [
    (ABS, "arxiv:2605.29313"),
    ("https://arxiv.org/abs/2605.29313v2", "arxiv:2605.29313"),
    ("http://www.arxiv.org/abs/2605.29313v1/", "arxiv:2605.29313"),
    ("https://export.arxiv.org/abs/2605.29313", "arxiv:2605.29313"),
    ("https://arxiv.org/pdf/2605.29313", "arxiv:2605.29313"),
    ("https://arxiv.org/pdf/2605.29313v3.pdf", "arxiv:2605.29313"),
    ("https://arxiv.org/html/2510.25779v1", "arxiv:2510.25779"),
    ("https://arxiv.org/html/2510.25779v1#S2", "arxiv:2510.25779"),
    ("https://arxiv.org/abs/2510.25779?context=cs.AI", "arxiv:2510.25779"),
    (DOI, "arxiv:2605.29313"),
    ("https://doi.org/10.48550/arXiv.2605.29313", "arxiv:2605.29313"),
    ("HTTPS://DOI.ORG/10.48550/ARXIV.2605.29313", "arxiv:2605.29313"),
    ("https://dx.doi.org/10.48550/arXiv.2605.29313", "arxiv:2605.29313"),
    ("https://arxiv.org/abs/0704.0001", "arxiv:0704.0001"),
    ("https://arxiv.org/abs/hep-th/9901001", "arxiv:hep-th/9901001"),
    ("https://arxiv.org/abs/hep-th/9901001v2", "arxiv:hep-th/9901001"),
    ("https://arxiv.org/abs/math/0309136", "arxiv:math/0309136"),
    ("https://arxiv.org/pdf/math.GT/0309136v1.pdf", "arxiv:math/0309136"),
    ("https://doi.org/10.48550/arXiv.hep-th/9901001", "arxiv:hep-th/9901001"),
])
def test_every_arxiv_form_names_one_versionless_paper(url: str, identity: str) -> None:
    assert arxiv_identity(url) == identity


@pytest.mark.parametrize("url", [
    "https://doi.org/10.1162/isal.a.1035",  # a non-arXiv DOI STIG-001 collected
    "https://doi.org/10.48550",
    "https://doi.org/10.485501/arXiv.2605.29313",
    "https://doi.org/10.48550/zenodo.2605.29313",
    "https://arxiv.org/list/cs.AI/recent",
    "https://arxiv.org/abs/",
    "https://arxiv.org/abs/2605",
    "https://arxiv.org/abs/2605.29313v",
    "https://arxiv.org.example.com/abs/2605.29313",
    "https://notarxiv.org/abs/2605.29313",
    "https://example.org/abs/2605.29313",
    "ftp://arxiv.org/abs/2605.29313",
    "not a url",
    "",
])
def test_non_arxiv_or_malformed_urls_have_no_paper_identity(url: str) -> None:
    assert arxiv_identity(url) is None


def test_membership_matches_a_collected_paper_across_url_forms() -> None:
    sources = build_source_url_set([{"url": DOI}, {"url": "https://arxiv.org/abs/2510.25779"}])
    assert classify_citation(live=True, url=ABS, source_url_set=sources) == CLASSIFICATION_LIVE
    assert classify_citation(
        live=True, url="https://arxiv.org/html/2510.25779v1", source_url_set=sources,
    ) == CLASSIFICATION_LIVE
    assert classify_citation(live=True, url=DOI, source_url_set=sources) == CLASSIFICATION_LIVE
    # A failed probe of a collected paper is dead-with-source: kept and flagged
    # unverified, never quarantined as fabricated.
    assert classify_citation(live=False, url=ABS, source_url_set=sources) == CLASSIFICATION_DEAD
    other = "https://arxiv.org/abs/2607.10487"
    assert classify_citation(live=True, url=other, source_url_set=sources) == CLASSIFICATION_ORPHAN
    assert classify_citation(
        live=False, url=other, source_url_set=sources,
    ) == CLASSIFICATION_HALLUCINATED


NON_ARXIV = [
    "https://Example.ORG/Docs/Page?utm_source=x&b=2#frag",
    "https://example.org/docs/page/",
    "http://www.example.org/docs/page?b=2",
    "https://doi.org/10.1162/isal.a.1035",
    "https://example.org/report.pdf?fbclid=abc",
]


def test_non_arxiv_membership_is_unchanged() -> None:
    sources = build_source_url_set([{"url": url} for url in NON_ARXIV])
    expected = {_normalise_url_for_cache(u) for u in NON_ARXIV}
    assert sources == expected
    for cited in NON_ARXIV + ["https://example.org/docs/page", "https://other.example/x"]:
        assert classify_citation(live=True, url=cited, source_url_set=sources) == (
            CLASSIFICATION_LIVE if _normalise_url_for_cache(cited) in expected
            else CLASSIFICATION_ORPHAN
        )


def _source(url: str, body: str = "x" * 200) -> dict[str, Any]:
    return {"url": url, "body": body, "content_quality": "readable_body",
            "body_source": "source_fetch"}


def test_missing_cited_sources_counts_papers_but_readability_stays_exact() -> None:
    outcomes = {"records": [], "records_truncated": 0}
    deps = build_evidence_dependencies([ABS, DOI], [_source(DOI)], outcomes)
    assert deps["version"] == 1
    assert deps["cited_urls"] == 2  # counted per cited URL, not per paper
    assert deps["missing_cited_sources"] == 0
    # Only the DOI page's body was read; the abstract URL stays unresolved.
    assert deps["readable_cited_urls"] == 1
    assert deps["cited_without_readable_body"] == 1


def test_non_arxiv_evidence_comparison_stays_exact_strings() -> None:
    outcomes = {"records": [], "records_truncated": 0}
    cited = ["https://example.org/a#frag", "https://Example.org/b", "https://example.org/c"]
    collected = [_source("https://example.org/a"), _source("https://example.org/b"),
                 _source("https://example.org/c")]
    deps = build_evidence_dependencies(cited, collected, outcomes)
    assert deps["missing_cited_sources"] == 2
    assert deps["cited_without_readable_body"] == 2


def test_probe_cache_body_trust_and_reference_identity_keep_url_forms_distinct() -> None:
    assert _normalise_url_for_cache(ABS) != _normalise_url_for_cache(DOI)
    assert _reference_url_identity(ABS) != _reference_url_identity(DOI)
    trusted = _build_substantive_body_url_set([_source(DOI, "x" * 2000)])
    assert _normalise_url_for_cache(DOI) in trusted
    assert _normalise_url_for_cache(ABS) not in trusted
    assert "arxiv:2605.29313" not in trusted


@pytest.mark.asyncio
async def test_liveness_gate_counts_each_cited_url_and_probes_untrusted_forms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed: list[str] = []

    def fake(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
        probed.append(url)
        return {"alive": True, "status_code": 200, "final_url": url}

    monkeypatch.setattr(citation_liveness, "url_liveness", fake)
    citations = [
        CitationAnnotation(start_index=0, end_index=5, url=ABS),
        CitationAnnotation(start_index=6, end_index=9, url=DOI),
    ]
    annotations, telemetry = await check_citations_liveness(citations, [_source(DOI, "x" * 2000)])
    assert [a.live for a in annotations] == [True, True]
    assert (telemetry.checked, telemetry.live, telemetry.orphan) == (2, 2, 0)
    assert telemetry.distinct_urls_checked == 2
    # The DOI body is trusted by its exact URL; the abstract form is still probed.
    assert probed == [ABS]


def test_gap_notes_never_fire_for_a_collected_paper() -> None:
    markdown = f"Claim [cite:{ABS}]. Other [cite:https://arxiv.org/abs/2607.10487]."
    quarantined = [
        CitationAnnotation(start_index=0, end_index=5, url=ABS),
        CitationAnnotation(start_index=0, end_index=5, url="https://arxiv.org/abs/2607.10487"),
    ]
    stripped, acted = strip_dead_cite_markers(markdown, quarantined, [{"url": DOI}])
    assert acted == 2
    assert stripped == "Claim . Other [Data unavailable in primary sources]."
