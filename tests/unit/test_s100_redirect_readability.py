"""S100-m04: a page read through a recorded same-site redirect counts as read.

A source's ``redirected_from`` aliases exist only when that exact URL's own
successful fetch resolved to the source's final URL (decision 1013), so the
stored body is what the cited alias serves. Collection membership already
counted aliases; readability now does too, in both the evidence dependencies and
the unread-source nudge, which share ``readable_source_urls``. A redirect that
changed host or landed on the site root (a login or a homepage) keeps its alias
unread, as does any alias of a source without a readable body.
"""

from __future__ import annotations

from typing import Any

from deepsearch.agent.quality_series import build_evidence_dependencies, readable_source_urls

BLOG = "https://mem0.ai/blog/memory-eviction-and-forgetting-in-ai-agents"
LIBRARY = "https://mem0.ai/library/agent-memory/memory-eviction-and-forgetting-in-ai-agents"


def source(url: str, *aliases: str, readable: bool = True,
           truncated: bool = False) -> dict[str, Any]:
    return {
        "url": url, "redirected_from": list(aliases), "body_source": "source_fetch",
        "content_quality": "readable_body" if readable else "snippet_only",
        "body_chars": 19210 if readable else 0, "body_truncated": truncated,
    }


def dependencies(cited: list[str], sources: list[dict[str, Any]]) -> dict[str, Any]:
    return build_evidence_dependencies(cited, sources, {"records": [], "records_truncated": 0})


def test_a_same_site_alias_of_a_readable_source_is_read() -> None:
    sources = [source(LIBRARY, BLOG)]
    assert readable_source_urls(sources) == {LIBRARY, BLOG}
    found = dependencies([BLOG], sources)
    assert (found["readable_cited_urls"], found["cited_without_readable_body"],
            found["missing_cited_sources"], found["truncated_cited_bodies"]) == (1, 0, 0, 0)


def test_an_alias_of_a_truncated_body_counts_as_truncated() -> None:
    found = dependencies([BLOG], [source(LIBRARY, BLOG, truncated=True)])
    assert (found["readable_cited_urls"], found["truncated_cited_bodies"]) == (1, 1)


def test_an_alias_of_a_source_without_a_readable_body_stays_unread() -> None:
    sources = [source(LIBRARY, BLOG, readable=False)]
    assert readable_source_urls(sources) == set()
    found = dependencies([BLOG], sources)
    # Collected through the redirect, as before, but not read.
    assert (found["readable_cited_urls"], found["cited_without_readable_body"],
            found["missing_cited_sources"]) == (0, 1, 0)


def test_an_alias_whose_redirect_changed_host_stays_unread() -> None:
    # An apex-to-www redirect and a cross-site login redirect both change host.
    for alias, final in (
        ("https://alphaxiv.org/abs/2608.02619", "https://www.alphaxiv.org/abs/2608.02619"),
        ("https://example.org/paper", "https://login.example.com/paper"),
    ):
        sources = [source(final, alias)]
        assert readable_source_urls(sources) == {final}
        found = dependencies([alias], sources)
        assert (found["readable_cited_urls"], found["missing_cited_sources"]) == (0, 0)


def test_an_alias_whose_redirect_landed_on_the_site_root_stays_unread() -> None:
    old = "https://example.org/2024/old-article"
    for final in ("https://example.org/", "https://example.org", "https://example.org/?moved=1"):
        sources = [source(final, old)]
        assert readable_source_urls(sources) == {final}
        assert dependencies([old], sources)["readable_cited_urls"] == 0
