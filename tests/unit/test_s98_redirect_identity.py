"""S98-m06: a URL whose own successful fetch redirected to a collected page is collected.

AGENTS.md accepts canonical redirect provenance as URL equivalence. The census found
7 orphans that were the requested URLs of pages the run fetched and stored under
their post-redirect ``final_url`` (decision 1011). Like the arXiv identity (m02),
the alias serves collection membership; body trust and the References audit stay
keyed by the exact collected URL. Since S100 m04 a same-site alias of a readable
source also counts as read, because its stored body is the page the alias serves.
"""

from __future__ import annotations

from typing import Any

import pytest

from deepsearch.agent import codeagent_runner as runner
from deepsearch.agent.citation_liveness import (
    CLASSIFICATION_LIVE,
    CLASSIFICATION_ORPHAN,
    _build_substantive_body_url_set,
    _normalise_url_for_cache,
    build_source_url_set,
    classify_citation,
)
from deepsearch.agent.codeagent_finalize_node import _reference_url_identity
from deepsearch.agent.quality_series import build_evidence_dependencies, readable_source_urls
from tests.unit.test_codeagent_runner import _install_fake_smolagents

OLD = "https://docs.example.org/old-page"
NEW = "https://docs.example.org/new-page"
BODY = "A moved documentation page with its full readable text. " * 20


def test_only_a_successful_observed_redirect_records_an_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_smolagents(monkeypatch)
    import deepsearch.agent.tools.source_fetch as sf_mod

    def fake_fetch(url: str, timeout_s: float = 15.0) -> dict[str, Any]:
        if "broken" in url:
            return {"text": "", "status": "error", "status_code": 0,
                    "fetched_at": "now", "final_url": url, "error": "connect"}
        final = NEW if url.startswith(OLD) else url
        return {"text": BODY, "status": "ok", "status_code": 200,
                "fetched_at": "now", "final_url": final}

    monkeypatch.setattr(sf_mod, "source_fetch", fake_fetch)
    bundle = runner.build_codeagent_tools()
    fetch = bundle.tools[1]
    fetch(f"{OLD}#section", 10.0)
    fetch(NEW, 10.0)
    fetch("https://docs.example.org/direct", 10.0)
    fetch("https://docs.example.org/broken", 10.0)
    by_url = {s["url"]: s for s in bundle.sources_collected}
    assert by_url[NEW]["redirected_from"] == [f"{OLD}#section"]
    assert "redirected_from" not in by_url["https://docs.example.org/direct"]
    assert "https://docs.example.org/broken" not in by_url


def _sources() -> list[dict[str, Any]]:
    return [{"url": NEW, "redirected_from": [f"{OLD}#section"], "body": BODY,
             "content_quality": "readable_body", "body_source": "source_fetch"}]


def test_membership_and_readability_count_the_alias_but_body_trust_stays_exact() -> None:
    sources = _sources()
    source_set = build_source_url_set(sources)
    assert classify_citation(live=True, url=OLD, source_url_set=source_set) == CLASSIFICATION_LIVE
    other = "https://docs.example.org/other"
    assert classify_citation(live=True, url=other, source_url_set=source_set) == (
        CLASSIFICATION_ORPHAN)
    # S100 m04 (superseding decision 1013's readability clause): the same-site
    # alias, exactly as fetched, reads the stored page.
    assert readable_source_urls(sources) == {NEW, f"{OLD}#section"}
    trusted = _build_substantive_body_url_set(sources)
    assert _normalise_url_for_cache(NEW) in trusted and _normalise_url_for_cache(OLD) not in trusted
    assert _reference_url_identity(OLD) != _reference_url_identity(NEW)
    outcomes = {"records": [], "records_truncated": 0}
    deps = build_evidence_dependencies([f"{OLD}#section", NEW], sources, outcomes)
    assert deps["missing_cited_sources"] == 0
    assert deps["readable_cited_urls"] == 2 and deps["cited_without_readable_body"] == 0


def test_sources_without_aliases_give_identical_membership() -> None:
    plain = [{"url": NEW}, {"url": "https://example.org/x?utm_source=y#z"}]
    assert build_source_url_set(plain) == {
        _normalise_url_for_cache(NEW), _normalise_url_for_cache("https://example.org/x?utm_source=y#z"),
    }
    assert build_source_url_set([{"url": NEW, "redirected_from": "not-a-list"}]) == {
        _normalise_url_for_cache(NEW)}


def test_the_alias_field_passes_the_strict_source_projection() -> None:
    projected = runner._canonicalize_sources_collected(
        [{"url": NEW, "title": "Moved page", "redirected_from": [OLD]}]
    )
    assert projected == [{"url": NEW, "title": "Moved page", "redirected_from": [OLD]}]
