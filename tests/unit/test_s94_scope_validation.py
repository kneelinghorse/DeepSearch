"""Word/source limits describe the persisted artifact, not citation presentation."""
from __future__ import annotations

import hashlib

import pytest

from deepsearch.agent.outcome import MissionOutcome, evaluate_outcome_gate
from deepsearch.mission.scope import AuthoredScope, ScopeAdmission
from deepsearch.report.scope_validation import (
    WORD_COUNTER_VERSION,
    count_report_words,
    measured_text,
    validate_persisted_scope,
)

URL1 = "https://www.postgresql.org/docs/current/functions-datetime.html"
URL2 = "https://www.postgresql.org/docs/current/explicit-locking.html"


def admission() -> ScopeAdmission:
    return ScopeAdmission(AuthoredScope(
        restriction="exact_pages", allowed_urls=[URL1, URL2],
        min_words=300, max_words=500, max_sources=2,
    ))


@pytest.mark.parametrize("words,violation", [
    (299, "scope_words_below_minimum"), (300, None), (500, None),
    (501, "scope_words_above_maximum"),
])
def test_inclusive_word_limits_count_exact_persisted_input(words: int, violation: str) -> None:
    text = " ".join(["word"] * words)
    policy = admission()
    policy.admit(URL1, "source_fetch")
    policy.admit(URL2, "source_fetch")
    result = validate_persisted_scope(
        admission=policy, pre_render="earlier draft", persisted=text,
        observed_sources=[], cited_urls=[URL1, URL2],
    )
    assert result["violations"] == ([violation] if violation else [])
    assert result["counter_version"] == WORD_COUNTER_VERSION
    assert result["persisted"]["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert result["persisted"]["words"] == words
    assert result["pre_render"]["words"] == 2
    assert result["controlling_surface"] == "persisted_report"


def test_wrapper_expansion_cannot_hide_behind_shorter_draft_count() -> None:
    draft = " ".join(["body"] * 500)
    persisted = "# Title\n**Sources:** 2\n" + draft + "\n## Objective\nExplain timing.\n"
    result = validate_persisted_scope(
        admission=admission(), pre_render=draft, persisted=persisted,
        observed_sources=[], cited_urls=[],
    )
    assert result["pre_render"]["words"] == 500
    assert result["persisted"]["words"] == 508
    assert "scope_words_above_maximum" in result["violations"]
    assert draft == " ".join(["body"] * 500)  # Validation never trims the paid artifact.


def test_counter_includes_markdown_citations_tables_urls_and_unicode_whitespace() -> None:
    text = "# Title\n**Sources:** 2\n| A | B |\ntext[cite:https://a.test/] [1]\n[x](https://x/)"
    assert count_report_words(text) == 12
    assert count_report_words("one\u001ctwo\u00a0three\u202ffour\u3000five") == 5
    assert count_report_words("one\u200btwo") == 1  # Zero-width space is not whitespace.
    assert count_report_words(" \n\t") == 0
    assert measured_text("a\r\nb")["sha256"] != measured_text("a\nb")["sha256"]


def test_two_citations_cannot_mask_22_consulted_pages_or_nonofficial_critique() -> None:
    urls = [URL1, URL2] + [f"https://postgrespro.com/page/{n}" for n in range(20)]
    policy = admission()
    # Simulate an upstream admission defect; the critic filters evidence, while
    # final validation still accounts for the actor's broader collected corpus.
    corpus = [{"url": u, "snippet": "Evidence"} for u in urls]
    assert len(policy.filter_sources(corpus, "pre_critique")) == 2
    text = "No third source was consulted. " + "word " * 310
    result = validate_persisted_scope(
        admission=policy, pre_render=text, persisted=text,
        observed_sources=corpus, cited_urls=[URL1, URL2],
    )
    assert result["consulted_count"] == 22
    assert len(result["prohibited_consulted_urls"]) == 20
    assert result["violations"] == [
        "scope_consulted_source_limit_exceeded", "scope_consulted_source_not_allowed",
    ]
    gate = evaluate_outcome_gate({"authored_scope_validation": result}, report_word_count=316)
    assert gate.outcome == MissionOutcome.VALIDATION_FAILED


def test_empty_unavailable_sources_are_explicit_partial_not_fake_consultation() -> None:
    policy = admission()
    policy.unavailable[URL1] = "timeout"
    text = "word " * 300
    result = validate_persisted_scope(
        admission=policy, pre_render=text, persisted=text,
        observed_sources=[], cited_urls=[],
    )
    assert result["verdict"] == "partial"
    assert result["consulted_count"] == 0
    assert result["warnings"] == ["scope_sources_unavailable", "scope_no_consulted_evidence"]
    assert result["unavailable"] == [
        {"url": URL1, "reason": "timeout"}, {"url": URL2, "reason": "not_retrieved"},
    ]


def test_rejected_unseen_search_result_is_not_counted_as_consultation() -> None:
    policy = admission()
    policy.admit(URL1, "source_fetch")
    policy.reject("https://outside.test/", "web_search", "scope_source_not_allowed")
    text = "word " * 300
    result = validate_persisted_scope(
        admission=policy, pre_render=text, persisted=text,
        observed_sources=[], cited_urls=[URL1],
    )
    assert result["consulted_count"] == 1
    assert not result["violations"]
