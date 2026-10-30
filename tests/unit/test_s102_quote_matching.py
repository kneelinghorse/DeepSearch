"""S102-m03: a verdict is not lost because its quote differs only in spacing or typography.

After an exact match fails, a source-passage quote is matched under an explicit
fold table: quotes, apostrophes and primes, hyphens and dashes, and the
ellipsis. Each run of whitespace becomes one space, and a space is removed only
beside a character that is neither alphanumeric nor sentence punctuation. No
NFKC, no case folding, and the minus sign stays apart. The stored quote is the
passage's own text at the stored offsets, never the critic's wording. Every
cited evidence id gets one class in ``quote_checks``, and every record demoted
for its evidence says why. The declared author context (C1) stays exact-match.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from deepsearch.agent import critique

URL = "https://example.org/source"
PASSAGE = (
    "Two independent annotators reviewed runs, achieving Cohen’s κ=0.87; a promise\n"
    "list reduces conflicting replies to 28% at 20 invitations,\nalthough errors remain. "
    "Results — as “expected” — held… then faded."
)
DRAFT = "# Findings\n\nThe annotators agreed strongly.\nThe repair helped somewhat.\n"
CLAIM = "The annotators agreed strongly."
OTHER_CLAIM = "The repair helped somewhat."


def visibility(passages: dict[str, str] | None = None, author_context: str = "") -> Any:
    shown = critique._GroundingVisibility(draft_chars_visible=len(DRAFT),
                                          author_context=author_context)
    offset = 1000
    for evidence_id, text in (passages or {"E1.1": PASSAGE}).items():
        shown.passages.append({
            "evidence_id": evidence_id, "source_id": "s", "url": URL, "body_sha256": "h",
            "start": offset, "end": offset + len(text), "body_chars": 9000,
            "kind": "source_passage"})
        shown.passage_text[evidence_id] = text
        offset += 1000
    if author_context:
        shown.passages.append({"evidence_id": "C1", "kind": "declared_context",
                               "chars": len(author_context), "truncated": False})
    return shown


def check(quote: Any, evidence_id: Any = "E1.1", *, index: int = 0,
          shown: Any = None) -> tuple[str, dict[str, Any] | None]:
    shown = shown or visibility()
    manifest = {p["evidence_id"]: p for p in shown.passages}
    return critique._checked_evidence(index, evidence_id, {evidence_id: quote}
                                      if isinstance(evidence_id, str) else {},
                                      manifest, shown)


def assess(items: list[dict], shown: Any = None) -> dict[str, Any]:
    raw = critique.CRITIQUE_OUTPUT_DELIMITER + "\n" + json.dumps(
        {"claim_count": len(items), "annotations": [], "assessments": items})
    return critique._assessment_record(raw, DRAFT, shown or visibility(), 0)


@pytest.mark.parametrize(("quote", "source_text"), [
    ("Cohen's κ = 0.87", "Cohen’s κ=0.87"),
    ("A promise list reduces conflicting replies to 28% at 20 invitations, although errors "
     "remain.", "a promise\nlist reduces conflicting replies to 28% at 20 invitations,\n"
                "although errors remain."),
    ('Results - as "expected" - held... then faded.',
     "Results — as “expected” — held… then faded."),
])
def test_spacing_and_typography_differences_match_and_store_the_source_text(
    quote: str, source_text: str,
) -> None:
    assert quote not in PASSAGE
    cls, entry = check(quote.replace("A promise", "a promise"))
    assert cls == "normalized" and entry is not None
    # The stored quote is the passage's own text at the stored offsets.
    assert entry["quote"] == source_text
    start = entry["quote_start"] - entry["start"]
    assert PASSAGE[start:entry["quote_end"] - entry["start"]] == entry["quote"]
    assert entry["quote_match"] == "normalized"


def test_an_exact_quote_stays_exact() -> None:
    cls, entry = check("Two independent annotators reviewed runs")
    assert (cls, entry["quote_match"], entry["quote_start"]) == ("exact", "exact", 1000)


@pytest.mark.parametrize(("quote", "source"), [
    ("Cohen's κ = 0.78", "Cohen’s κ=0.87"),  # a changed number
    ("the rapist said", "the therapist said"),
    ("a sample of 102 agents", "a sample of 10² agents"),
    ("a rate of 10-4 here", "a rate of 10⁻⁴ here"),
    ("in Table 4.87 agents", "in Table 4. 87 agents"),
    ("costs rose 1,000 units", "costs rose 1, 000 units"),
    ("a range of 10–20 units", "a range of 10 − 20 units"),  # range vs subtraction
    ("step 1 begins now", "step ① begins now"),
    ("κ  =  0.8", "κ=0.8"),  # under 8 normalized characters
])
def test_negative_controls_never_match(quote: str, source: str) -> None:
    cls, entry = check(quote, shown=visibility({"E1.1": f"Before. {source} After."}))
    assert (cls, entry) == ("not_found", None)


def test_a_quote_from_one_passage_does_not_match_another() -> None:
    shown = visibility({"E1.1": PASSAGE, "E1.2": "An unrelated passage about caching layers."})
    assert check("Cohen's κ = 0.87", "E1.2", shown=shown) == ("not_found", None)
    assert check("Cohen's κ = 0.87", "E1.1", shown=shown)[0] == "normalized"


def test_declared_context_stays_exact_match() -> None:
    context = json.dumps({"background": "The team’s workflows were disabled."},
                         ensure_ascii=False)
    shown = visibility(author_context=context)
    assert check("The team's workflows were disabled.", "C1", shown=shown) == (
        "not_found", None)
    cls, entry = check("The team’s workflows were disabled.", "C1", shown=shown)
    assert (cls, entry["quote_match"]) == ("exact", "exact")


def test_a_match_ending_inside_an_ellipsis_stores_the_whole_character() -> None:
    cls, entry = check("as \"expected\" - held..")
    assert cls == "normalized"
    assert entry["quote"] == "as “expected” — held…"


def test_the_stored_quote_may_exceed_600_characters() -> None:
    words = [f"word{n}" for n in range(80)]
    spaced = "\n\n".join(words)
    quote = " ".join(words)
    assert len(quote) <= 600 < len(spaced)
    cls, entry = check(quote, shown=visibility({"E1.1": spaced}))
    assert (cls, entry["quote"]) == ("normalized", spaced)


@pytest.mark.parametrize(("ids", "quotes", "classes"), [
    (["E1.1"], {"E1.1": "Two independent annotators reviewed runs"}, {"exact": 1}),
    (["E1.1"], {"E1.1": "Cohen's κ = 0.87"}, {"normalized": 1}),
    (["E9.9"], {"E9.9": "Two independent annotators reviewed runs"}, {"unknown_id": 1}),
    ([7], {}, {"unknown_id": 1}),
    (["E1.1"], {}, {"missing_quote": 1}),
    (["E1.1"], {"E1.1": "short"}, {"quote_length": 1}),
    (["E1.1"], {"E1.1": "x" * 601}, {"quote_length": 1}),
    (["E1.1"], {"E1.1": "A sentence the passage never contains."}, {"not_found": 1}),
    (["E1.1"] * 4, {"E1.1": "Two independent annotators reviewed runs"},
     {"exact": 3, "over_limit": 1}),
])
def test_every_cited_id_gets_exactly_one_class(
    ids: list, quotes: dict, classes: dict[str, int],
) -> None:
    assessment = assess([{"anchor": CLAIM, "verdict": "supported", "evidence_ids": ids,
                          "evidence_quotes": quotes}])
    expected = dict.fromkeys(critique._QUOTE_CHECK_CLASSES, 0) | classes
    assert assessment["quote_checks"] == expected
    assert sum(assessment["quote_checks"].values()) == len(ids)
    (record,) = assessment["records"]
    failed = [c for c in classes if c not in {"exact", "normalized"}]
    if failed:
        assert (record["verdict"], record["reason"], record["quote_failure"]) == (
            "unassessed", "no_visible_quoted_evidence", failed[0])
    else:
        assert record["verdict"] == "supported" and "quote_failure" not in record


def test_a_matcher_fault_keeps_the_exact_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: Any) -> Any:
        raise RuntimeError("injected")

    monkeypatch.setattr(critique, "_normalized_quote_span", broken)
    assessment = assess([
        {"anchor": CLAIM, "verdict": "supported", "evidence_ids": ["E1.1"],
         "evidence_quotes": {"E1.1": "Cohen's κ = 0.87"}},
        {"anchor": OTHER_CLAIM, "verdict": "supported", "evidence_ids": ["E1.1"],
         "evidence_quotes": {"E1.1": "Two independent annotators reviewed runs"}},
    ])
    assert assessment["version"] == 2 and "provenance_error" not in assessment
    assert (assessment["quote_checks"]["matcher_error"], assessment["quote_checks"]["exact"]) == (
        1, 1)
    first, second = assessment["records"]
    assert (first["verdict"], first["quote_failure"]) == ("unassessed", "matcher_error")
    assert second["verdict"] == "supported"


def test_record_level_failures_without_a_failing_id() -> None:
    assessment = assess([
        {"anchor": CLAIM, "verdict": "supported", "evidence_ids": []},
        {"anchor": OTHER_CLAIM, "verdict": "declared_context", "evidence_ids": ["E1.1"],
         "evidence_quotes": {"E1.1": "Two independent annotators reviewed runs"}},
    ])
    first, second = assessment["records"]
    assert first["quote_failure"] == "no_evidence_ids"
    # The quote verifies, but a declared-context verdict needs C1.
    assert second["quote_failure"] == "required_kind_missing"
    assert "failed_quote" not in first and "failed_quote" not in second


def test_a_quote_not_found_keeps_its_first_200_characters() -> None:
    long_quote = "A sentence the passage never contains, " * 12
    assessment = assess([
        {"anchor": CLAIM, "verdict": "supported", "evidence_ids": ["E1.1"],
         "evidence_quotes": {"E1.1": long_quote[:500]}},
        {"anchor": OTHER_CLAIM, "verdict": "supported", "evidence_ids": ["E1.1"],
         "evidence_quotes": {"E1.1": "A short quote not in the passage."}},
    ])
    first, second = assessment["records"]
    assert first["quote_failure"] == "not_found"
    assert first["failed_quote"] == long_quote[:200]
    assert second["failed_quote"] == "A short quote not in the passage."
    assert all(len(r["failed_quote"]) <= 200 for r in assessment["records"])


def test_an_inconsistent_verdict_is_never_given_a_quote_failure() -> None:
    assessment = assess([{"anchor": CLAIM, "verdict": "inconsistent",
                          "conflict_anchor": OTHER_CLAIM, "evidence_ids": ["E9.9"]}])
    (record,) = assessment["records"]
    assert record["verdict"] == "inconsistent" and "quote_failure" not in record
    assert assessment["quote_checks"]["unknown_id"] == 1
