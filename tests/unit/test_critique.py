"""Unit tests for ``deepsearch.agent.critique`` — S66 critique stage.

S85 m01 (decision #823) rewrote the stage from a whole-document LLM rewrite
into SPAN-LEVEL ANNOTATION: the LLM emits a JSON list of ``{anchor, verdict}``
annotations after a delimiter, and a DETERMINISTIC applier renders gap notes
onto the ORIGINAL draft. These tests exercise the applier / structural guard /
hard-fail parsing (all deterministic — the mock model only supplies annotation
JSON, never the report body). Per PD46 there are NO mock-LLM *behavior* tests —
we never assert what the LLM decides, only what our deterministic transform does
with a given annotation set.

Coverage:
- span annotation → deterministic gap note; output is draft-derived, never the
  model's echo (SPOF containment, criterion 1)
- empty annotations → byte-for-byte no-op (identity invariant)
- missing delimiter / unparseable JSON → HARD FAIL, body untouched (criterion 3)
- structural diff guard rejects hollowing / heading loss (criterion 2)
- full-body grounding reads the ``body`` field, not the 300-char snippet
  (criterion 4)
- unmatched / ambiguous / heading anchors are skipped (recall-safe)
- empty draft short-circuits; LLM error falls through; temp/tokens locked
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from deepsearch.agent import critique
from deepsearch.agent.critique import (
    CRITIQUE_OUTPUT_DELIMITER,
    CRITIQUE_SYSTEM_PROMPT,
    GAP_NOTE_DEFAULT,
    SOURCE_GROUNDING_MAX_CHARS,
    SOURCE_SNIPPET_MAX_CHARS,
    CritiqueResult,
    CritiqueTransformError,
    run_critique,
)
from deepsearch.report.citation import CitationAnnotation


def _assert_safe_error(
    value: str | None,
    *,
    category: str,
    exception_class: str,
) -> None:
    """Pin category/class diagnostics without retaining provider text."""

    assert value is not None
    assert json.loads(value) == {
        "category": category,
        "exception_class": exception_class,
    }


# ---------------------------------------------------------------------------
# Helpers — build a realistic-shape mock response carrying annotation JSON.
# ---------------------------------------------------------------------------


def _annotations_raw(annotations: list[dict], *, claim_count: int | None = None) -> str:
    """Render the model output: the delimiter followed by the JSON block."""

    payload = {
        "claim_count": len(annotations) if claim_count is None else claim_count,
        "annotations": annotations,
    }
    return f"Some preamble the applier ignores.\n{CRITIQUE_OUTPUT_DELIMITER}\n{json.dumps(payload)}\n"


def _mock_response(content: str, *, in_tok: int = 100, out_tok: int = 200) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        token_usage=SimpleNamespace(
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=in_tok + out_tok,
        ),
    )


def _mission_ctx() -> dict:
    return {
        "title": "AWS Lambda concurrency limits",
        "objective": "Document concurrency, throttling, and provisioned concurrency.",
    }


def _sources(n: int = 2) -> list[dict]:
    return [
        {
            "url": f"https://docs.aws.amazon.com/lambda/page-{i}.html",
            "title": f"Lambda doc {i}",
            "snippet": f"Snippet for source {i}.",
            "body": f"Full body text for source {i} with supporting detail.",
            "score": 0.8 - 0.05 * i,
            "authority_tier": "high",
        }
        for i in range(n)
    ]


_DRAFT = (
    "# Lambda Concurrency\n\n"
    "## Reserved Concurrency\n\n"
    "Reserved concurrency caps invocations per function.\n"
    "The cold-start penalty is exactly 9.4 seconds on every runtime.\n\n"
    "## Provisioned Concurrency\n\n"
    "Provisioned concurrency keeps a warm pool ready.\n"
)


# ---------------------------------------------------------------------------
# 1. Span annotation → deterministic gap note (happy path).
# ---------------------------------------------------------------------------


def test_critique_applies_gap_note_for_unsupported_claim():
    anns = [
        {
            "anchor": "The cold-start penalty is exactly 9.4 seconds on every runtime.",
            "verdict": "unsupported",
            "note": "no source",
        }
    ]
    fake_model = MagicMock(return_value=_mock_response(_annotations_raw(anns, claim_count=3)))

    result = run_critique(
        draft_markdown=_DRAFT,
        sources_collected=_sources(2),
        mission_context=_mission_ctx(),
        model=fake_model,
    )

    assert isinstance(result, CritiqueResult)
    assert result.error is None
    assert result.claim_count == 3
    assert result.gap_notes_inserted == 1
    # The unsupported claim is replaced by the canonical gap note...
    assert GAP_NOTE_DEFAULT in result.output_markdown
    assert "9.4 seconds" not in result.output_markdown
    # ...and everything else survives byte-for-byte (headings + supported claim).
    assert result.output_markdown.count("#") == _DRAFT.count("#")
    assert "Reserved concurrency caps invocations per function." in result.output_markdown
    assert result.token_usage == {"input": 100, "output": 200, "total": 300}
    assert result.token_usage_complete is True
    fake_model.assert_called_once()


def test_critique_output_is_draft_derived_not_model_echo():
    """SPOF containment (criterion 1): even a model that echoes a *different*
    whole document cannot make that document the report — output_markdown is
    always the input draft with gap notes applied by anchor.
    """

    # The model tries to smuggle a whole replacement body BEFORE the delimiter.
    smuggled = "# HIJACKED REPORT\n\nEntirely different content the model invented.\n"
    raw = f"{smuggled}\n{CRITIQUE_OUTPUT_DELIMITER}\n" + json.dumps(
        {"claim_count": 1, "annotations": []}
    )
    fake_model = MagicMock(return_value=_mock_response(raw))

    result = run_critique(
        draft_markdown=_DRAFT,
        sources_collected=[],
        mission_context=_mission_ctx(),
        model=fake_model,
    )
    assert result.error is None
    assert "HIJACKED" not in result.output_markdown
    assert result.output_markdown == _DRAFT  # nothing to apply → exact draft


# ---------------------------------------------------------------------------
# 2. Empty annotations → byte-for-byte no-op (identity invariant).
# ---------------------------------------------------------------------------


def test_critique_no_annotations_is_byte_identical_noop():
    """When the model finds nothing unsupported, output MUST equal the input
    draft byte-for-byte — the finalize node's scored/persisted SHA identity and
    coverage recompute both depend on a no-change run producing an unchanged body.
    """

    fake_model = MagicMock(return_value=_mock_response(_annotations_raw([], claim_count=4)))
    result = run_critique(
        draft_markdown=_DRAFT,
        sources_collected=_sources(2),
        mission_context=_mission_ctx(),
        model=fake_model,
    )
    assert result.error is None
    assert result.gap_notes_inserted == 0
    assert result.output_markdown == _DRAFT


# ---------------------------------------------------------------------------
# 3. Hard-fail on missing delimiter / unparseable JSON (criterion 3).
# ---------------------------------------------------------------------------


def test_critique_hard_fail_missing_delimiter_preserves_body():
    """No delimiter → HARD FAIL: error populated, body untouched, zero gaps.

    This INVERTS the pre-S85 silent-degrade that shipped the clipped LLM echo
    (the CAT-era 'ends mid-sentence' defect). A clipped output must now fail
    loudly so the finalize node ships the pre-critique stripped markdown.
    """

    clipped = "# Lambda Concurrency\n\nReserved concurrency caps invocations per fun"
    fake_model = MagicMock(return_value=_mock_response(clipped))
    result = run_critique(
        draft_markdown=_DRAFT,
        sources_collected=_sources(),
        mission_context=_mission_ctx(),
        model=fake_model,
    )
    _assert_safe_error(
        result.error,
        category="validation",
        exception_class="CritiqueOutputError",
    )
    assert result.output_markdown == _DRAFT  # body preserved, not the clip
    assert result.gap_notes_inserted == 0


def test_critique_hard_fail_unparseable_json_preserves_body():
    raw = f"{CRITIQUE_OUTPUT_DELIMITER}\n{{not valid json at all,,,"
    fake_model = MagicMock(return_value=_mock_response(raw))
    result = run_critique(
        draft_markdown=_DRAFT,
        sources_collected=[],
        mission_context=_mission_ctx(),
        model=fake_model,
    )
    _assert_safe_error(
        result.error,
        category="validation",
        exception_class="CritiqueOutputError",
    )
    assert result.output_markdown == _DRAFT


def test_critique_hard_fail_annotations_not_a_list():
    """A JSON object whose ``annotations`` is not a list is a hard failure."""

    raw = f"{CRITIQUE_OUTPUT_DELIMITER}\n" + json.dumps({"annotations": "oops", "claim_count": 1})
    fake_model = MagicMock(return_value=_mock_response(raw))
    result = run_critique(
        draft_markdown=_DRAFT, sources_collected=[], mission_context=_mission_ctx(), model=fake_model
    )
    _assert_safe_error(
        result.error,
        category="validation",
        exception_class="CritiqueOutputError",
    )
    assert result.output_markdown == _DRAFT


# ---------------------------------------------------------------------------
# 4. Structural diff guard (criterion 2).
# ---------------------------------------------------------------------------


def test_guard_rejects_hollowing_transform():
    """A transform that guts the body below the hollow floor is rejected."""

    draft = "# H\n\n" + ("Substantive sentence number %d. " % 0) * 200
    hollow = "# H\n\ntiny"
    # Fabricate an edit list that claims to turn draft->hollow by one giant
    # replacement (a hollowing transform); the guard must reject it.
    edits = [(draft.index("Substantive"), len(draft), "tiny")]
    with pytest.raises(CritiqueTransformError):
        critique._assert_bounded_transform(draft, hollow, edits)


def test_guard_rejects_heading_deletion():
    """An edit whose range covers a heading line is rejected."""

    draft = "# Title\n\nA claim here that is long enough.\n\n## Section\n\nMore text.\n"
    # An edit spanning from mid-body across the '## Section' heading.
    start = draft.index("A claim")
    end = draft.index("More text")
    candidate = draft[:start] + GAP_NOTE_DEFAULT + draft[end:]
    edits = [(start, end, GAP_NOTE_DEFAULT)]
    with pytest.raises(CritiqueTransformError):
        critique._assert_bounded_transform(draft, candidate, edits)


def test_guard_rejects_non_gapnote_replacement():
    draft = "# Title\n\nA claim here that is long enough to anchor.\n"
    start = draft.index("A claim")
    end = draft.index("\n", start)
    candidate = draft[:start] + "INJECTED NOVEL CONTENT" + draft[end:]
    edits = [(start, end, "INJECTED NOVEL CONTENT")]
    with pytest.raises(CritiqueTransformError):
        critique._assert_bounded_transform(draft, candidate, edits)


def test_guard_accepts_annotation_only_transform():
    draft = "# Title\n\nA claim here that is long enough to anchor.\nKept sentence.\n"
    start = draft.index("A claim here that is long enough to anchor.")
    end = start + len("A claim here that is long enough to anchor.")
    candidate = draft[:start] + GAP_NOTE_DEFAULT + draft[end:]
    edits = [(start, end, GAP_NOTE_DEFAULT)]
    # Must not raise.
    critique._assert_bounded_transform(draft, candidate, edits)


def test_critique_guard_breach_degrades_and_preserves_body(monkeypatch):
    """If the applier ever produced a guard-breaching candidate, run_critique
    degrades loudly (error populated) and returns the untouched draft.
    """

    draft = "# Title\n\nClaim one is here and long.\nClaim two is here and long.\n"

    def _boom_applier(d, anns, citations=None, **kwargs):
        del citations, kwargs
        # Simulate an applier bug: delete a heading (candidate + bad edit).
        bad_candidate = "gutted"
        outcomes = [
            critique.CritiqueAnnotation(
                anchor=ann.anchor,
                verdict=ann.verdict,
                applied=True,
                reason="applied_gap_note",
            )
            for ann in anns
        ]
        return bad_candidate, [(0, len(d), GAP_NOTE_DEFAULT)], outcomes

    monkeypatch.setattr(critique, "_apply_span_annotations", _boom_applier)
    anns = [{"anchor": "Claim one is here and long.", "verdict": "unsupported"}]
    fake_model = MagicMock(return_value=_mock_response(_annotations_raw(anns)))
    result = run_critique(draft, [], _mission_ctx(), model=fake_model)
    _assert_safe_error(
        result.error,
        category="validation",
        exception_class="CritiqueTransformError",
    )
    assert result.output_markdown == draft
    assert result.gap_notes_inserted == 0


# ---------------------------------------------------------------------------
# 5. Recall-safe anchor matching — unmatched / ambiguous / heading skipped.
# ---------------------------------------------------------------------------


def test_critique_unmatched_anchor_skipped_recall_safe():
    anns = [{"anchor": "This exact sentence does not appear in the draft.", "verdict": "unsupported"}]
    fake_model = MagicMock(return_value=_mock_response(_annotations_raw(anns)))
    result = run_critique(_DRAFT, [], _mission_ctx(), model=fake_model)
    assert result.error is None
    assert result.gap_notes_inserted == 0
    assert result.output_markdown == _DRAFT


def test_critique_ambiguous_anchor_skipped():
    draft = "Alpha. The value is measured. Beta. The value is measured. Gamma.\n"
    anns = [{"anchor": "The value is measured.", "verdict": "unsupported"}]
    fake_model = MagicMock(return_value=_mock_response(_annotations_raw(anns)))
    result = run_critique(draft, [], _mission_ctx(), model=fake_model)
    assert result.error is None
    assert result.gap_notes_inserted == 0
    assert result.output_markdown == draft  # non-unique → preserved


def test_critique_heading_anchor_refused():
    anns = [{"anchor": "## Provisioned Concurrency", "verdict": "unsupported"}]
    fake_model = MagicMock(return_value=_mock_response(_annotations_raw(anns)))
    result = run_critique(_DRAFT, [], _mission_ctx(), model=fake_model)
    assert result.error is None
    assert result.gap_notes_inserted == 0
    assert "## Provisioned Concurrency" in result.output_markdown


def test_critique_too_short_anchor_skipped():
    anns = [{"anchor": "caps", "verdict": "unsupported"}]
    fake_model = MagicMock(return_value=_mock_response(_annotations_raw(anns)))
    result = run_critique(_DRAFT, [], _mission_ctx(), model=fake_model)
    assert result.gap_notes_inserted == 0
    assert result.output_markdown == _DRAFT


# ---------------------------------------------------------------------------
# 6. Empty / whitespace draft short-circuits without an LLM call.
# ---------------------------------------------------------------------------


def test_critique_empty_draft():
    fake_model = MagicMock()
    result = run_critique(
        draft_markdown="",
        sources_collected=_sources(),
        mission_context=_mission_ctx(),
        model=fake_model,
    )
    assert result.output_markdown == ""
    assert result.claim_count == 0
    assert result.gap_notes_inserted == 0
    assert result.error is None
    fake_model.assert_not_called()


def test_critique_whitespace_only_draft_echoes_input_no_llm():
    """A whitespace-only draft short-circuits WITHOUT an LLM call and echoes the
    input byte-for-byte (I1 no-op identity — must not blank the body to "").
    """

    fake_model = MagicMock()
    draft = "   \n\t  \n"
    result = run_critique(
        draft_markdown=draft,
        sources_collected=_sources(),
        mission_context=_mission_ctx(),
        model=fake_model,
    )
    assert result.output_markdown == draft
    assert result.error is None
    assert result.gap_notes_inserted == 0
    fake_model.assert_not_called()


# ---------------------------------------------------------------------------
# 7. LLM error / model-build error — fallthrough, no raise.
# ---------------------------------------------------------------------------


def test_critique_llm_error_fallthrough():
    fake_model = MagicMock(side_effect=RuntimeError("model timeout"))
    draft = "Draft body that must survive intact when critique LLM fails."
    result = run_critique(
        draft_markdown=draft,
        sources_collected=_sources(),
        mission_context=_mission_ctx(),
        model=fake_model,
    )
    assert result.output_markdown == draft
    _assert_safe_error(
        result.error,
        category="execution",
        exception_class="RuntimeError",
    )
    assert result.claim_count == 0
    assert result.gap_notes_inserted == 0
    assert result.token_usage == {"input": 0, "output": 0, "total": 0}
    assert result.token_usage_complete is False


def test_critique_error_drops_hostile_provider_text() -> None:
    """Signed URLs and response bodies cannot survive the critique boundary."""

    signed_url = "https://user:pass@example.test/private?signature=critique-secret"
    body_marker = "HOSTILE_CRITIQUE_RESPONSE_BODY"
    fake_model = MagicMock(
        side_effect=RuntimeError(f"{signed_url} {body_marker}")
    )

    result = run_critique(
        draft_markdown="Draft body remains reviewable.",
        sources_collected=[],
        mission_context=_mission_ctx(),
        model=fake_model,
    )

    _assert_safe_error(
        result.error,
        category="execution",
        exception_class="RuntimeError",
    )
    assert result.token_usage_complete is False
    assert signed_url not in result.error
    assert body_marker not in result.error


def test_critique_default_model_build_failure_fallthrough(monkeypatch: pytest.MonkeyPatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    monkeypatch.setattr(critique, "build_default_model", _boom)
    draft = "Draft preserved on construction error."
    result = run_critique(
        draft_markdown=draft,
        sources_collected=_sources(),
        mission_context=_mission_ctx(),
    )
    assert result.output_markdown == draft
    _assert_safe_error(
        result.error,
        category="execution",
        exception_class="RuntimeError",
    )
    assert result.token_usage_complete is True


def test_critique_handles_bare_string_response():
    """A bare-string response (no .content / token_usage) is accepted."""

    raw = _annotations_raw([], claim_count=1)
    fake_model = MagicMock(return_value=raw)
    result = run_critique(
        draft_markdown=_DRAFT,
        sources_collected=_sources(),
        mission_context=_mission_ctx(),
        model=fake_model,
    )
    assert result.error is None
    assert result.claim_count == 1
    assert result.gap_notes_inserted == 0
    assert result.token_usage == {"input": 0, "output": 0, "total": 0}
    assert result.token_usage_complete is False


def test_critique_explicit_zero_usage_is_complete() -> None:
    """Provider-reported zero differs from an absent usage envelope."""

    result = run_critique(
        draft_markdown=_DRAFT,
        sources_collected=_sources(),
        mission_context=_mission_ctx(),
        model=MagicMock(
            return_value=_mock_response(
                _annotations_raw([], claim_count=1),
                in_tok=0,
                out_tok=0,
            )
        ),
    )

    assert result.error is None
    assert result.token_usage == {"input": 0, "output": 0, "total": 0}
    assert result.token_usage_complete is True


def test_critique_contradictory_usage_is_a_lower_bound() -> None:
    """A total smaller than its components cannot be declared exact."""

    usage, complete = critique._extract_token_usage(
        {
            "token_usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 10,
            }
        }
    )

    assert usage == {"input": 100, "output": 50, "total": 150}
    assert complete is False


# ---------------------------------------------------------------------------
# 8. Full-body grounding (criterion 4) — reads body, not the 300-char snippet.
# ---------------------------------------------------------------------------


def test_critique_grounding_reads_full_body_not_snippet():
    """The snippet 'lies' (300 chars that miss the fact); the body verifies it.

    The critique prompt's SOURCE LIST must carry the body content, so a fact
    living beyond the snippet boundary is available for grounding — the exact
    class that gap-noted verifiable facts in the CAT runs (decision #823).
    """

    lie = "preamble " * 40  # > 300 chars, none of it the fact
    assert len(lie) > SOURCE_SNIPPET_MAX_CHARS
    fact = "The provisioned concurrency ceiling is 1000 per region."
    sources = [{"url": "https://docs.example.com/p", "title": "Doc", "snippet": lie[:300], "body": lie + fact}]

    messages = critique._build_critique_messages(
        draft_markdown="draft", sources_collected=sources, mission_context=_mission_ctx()
    )
    user = messages[1]["content"]
    assert fact in user, "the body fact beyond the snippet must reach the prompt"


def test_critique_grounding_falls_back_to_snippet_when_no_body():
    sources = [{"url": "https://docs.example.com/p", "title": "Doc", "snippet": "only a snippet here"}]
    messages = critique._build_critique_messages(
        draft_markdown="draft", sources_collected=sources, mission_context=_mission_ctx()
    )
    assert "only a snippet here" in messages[1]["content"]


def test_critique_grounding_per_source_cap_bounds_prompt():
    huge = "z" * 50_000
    sources = [{"url": "https://docs.example.com/p", "title": "Doc", "snippet": "s", "body": huge}]
    messages = critique._build_critique_messages(
        draft_markdown="draft", sources_collected=sources, mission_context=_mission_ctx()
    )
    user = messages[1]["content"]
    # Body is capped per source — the full 50k never reaches the prompt.
    assert huge not in user
    assert ("z" * SOURCE_GROUNDING_MAX_CHARS) not in user
    assert "[E1.1 chars=" in user  # Budget now includes bounded passage provenance.
    assert ("z" * (SOURCE_GROUNDING_MAX_CHARS + 1)) not in user


def test_critique_grounding_aggregate_budget_shrinks_per_source_cap():
    from deepsearch.agent.critique import GROUNDING_TOTAL_BUDGET_CHARS

    # Many big sources: the aggregate budget forces a smaller effective cap.
    n = 500
    sources = [
        {"url": f"https://d/{i}", "title": "D", "snippet": "s", "body": "q" * 5000}
        for i in range(n)
    ]
    prepared = critique._prepare_sources(sources)
    total = sum(len(p["snippet"]) for p in prepared)
    # Stays within (a small multiple of) the declared aggregate budget.
    assert total <= GROUNDING_TOTAL_BUDGET_CHARS + n  # +n rounding slack
    # Every source still keeps at least the snippet floor.
    assert all(len(p["snippet"]) >= 1 for p in prepared)


def test_critique_drops_sourceless_entries_in_prompt():
    sources = [
        {"url": "", "title": "no url", "snippet": "x", "body": "x"},
        {"url": "https://docs.example.com/keep", "title": "kept", "snippet": "ok", "body": "ok body"},
    ]
    fake_model = MagicMock(return_value=_mock_response(_annotations_raw([])))
    run_critique(
        draft_markdown="draft",
        sources_collected=sources,
        mission_context=_mission_ctx(),
        model=fake_model,
    )
    user = next(m for m in fake_model.call_args.kwargs["messages"] if m["role"] == "user")
    assert "https://docs.example.com/keep" in user["content"]
    assert "no url" not in user["content"]
    assert "1 sources" in user["content"]


def test_critique_empty_sources_still_invokes_llm():
    fake_model = MagicMock(return_value=_mock_response(_annotations_raw([])))
    result = run_critique(
        draft_markdown="some draft with claims",
        sources_collected=[],
        mission_context=_mission_ctx(),
        model=fake_model,
    )
    fake_model.assert_called_once()
    assert result.error is None
    sent = fake_model.call_args.kwargs["messages"]
    user_msg = next(m for m in sent if m["role"] == "user")
    assert "no sources collected" in user_msg["content"]


# ---------------------------------------------------------------------------
# 9. Temperature / max_tokens locked at inference time.
# ---------------------------------------------------------------------------


def test_critique_inherits_resolved_model_parameters():
    fake_model = MagicMock(return_value=_mock_response(_annotations_raw([])))
    run_critique(
        draft_markdown="draft",
        sources_collected=_sources(),
        mission_context=_mission_ctx(),
        model=fake_model,
    )
    assert fake_model.call_count == 1
    kwargs = fake_model.call_args.kwargs
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs
    assert "max_completion_tokens" not in kwargs
    msgs = kwargs["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]


# ---------------------------------------------------------------------------
# 10. System prompt contract pins (annotation-only output + preservation).
# ---------------------------------------------------------------------------


def test_critique_system_prompt_is_annotation_only_contract():
    """The prompt must instruct annotation-only output (no body round-trip),
    name the delimiter + gap-note form, and require verbatim anchors.
    """

    p = CRITIQUE_SYSTEM_PROMPT
    assert "SPAN ANNOTATION" in p or "span annotation" in p.lower()
    assert "You NEVER output the report body" in p or "never output the report body" in p.lower()
    assert CRITIQUE_OUTPUT_DELIMITER in p
    assert "VERBATIM" in p or "verbatim" in p.lower()
    assert GAP_NOTE_DEFAULT in p
    assert '"anchor"' in p and '"verdict"' in p
    # Hard-fail contract advertised so the model knows malformed output is fatal.
    assert "hard failure" in p.lower()


def test_critique_system_prompt_preservation_and_liveness_rules():
    """Preservation bias + the S84-narrowed hallucinated-only handling survive.

    If this softens, the quarantine amplifier that gutted the CAT runs
    (decision #823) or an over-aggressive gap-noter could return.
    """

    p = CRITIQUE_SYSTEM_PROMPT
    # Preservation bias.
    assert "PRESERVATION" in p or "preservation" in p.lower()
    assert "no support" in p.lower()
    assert "do not annotate" in p.lower()
    # Do not annotate structure.
    assert "headings" in p.lower()
    # S84 m02 hallucinated-only: hallucinated → annotate; dead/orphan/unknown → preserved.
    assert "hallucinated" in p.lower()
    assert "Data unavailable in primary sources" in p
    assert "flagged unverified downstream" in p
    assert "per-citation context block" in p
    assert "live=None" in p or "unknown" in p.lower()
    # No retrieval surface rationale.
    assert "no tools" in p.lower() or "cannot retrieve" in p.lower()


# ---------------------------------------------------------------------------
# 11. Liveness context block (S68 / S82 m05) — preserved rendering.
# ---------------------------------------------------------------------------


def test_build_critique_messages_emits_liveness_context_section():
    """`citations=` non-empty adds the # Citation Liveness Context block with
    the shared 5-state classification rendered '[cite:URL]  →  descriptor'.
    """

    sources = [
        {"url": "https://docs.aws.amazon.com/lambda/0", "title": "L0"},
        {"url": "https://docs.aws.amazon.com/lambda/1", "title": "L1"},
    ]
    citations = [
        CitationAnnotation(start_index=0, end_index=5, url="https://docs.aws.amazon.com/lambda/0", live=True),
        CitationAnnotation(start_index=10, end_index=20, url="https://docs.aws.amazon.com/lambda/1", live=False),
        CitationAnnotation(start_index=30, end_index=40, url="https://orphan.example", live=False),
        CitationAnnotation(start_index=50, end_index=60, url="https://unknown.example", live=None),
    ]

    messages = critique._build_critique_messages(
        draft_markdown="# Draft", sources_collected=sources, mission_context={"title": "T", "objective": "O"}, citations=citations
    )
    user = messages[1]["content"]

    assert "# Citation Liveness Context" in user
    assert "[cite:https://docs.aws.amazon.com/lambda/0]  →  live" in user
    assert "[cite:https://docs.aws.amazon.com/lambda/1]  →  dead" in user
    assert "[cite:https://orphan.example]  →  hallucinated" in user
    assert "[cite:https://unknown.example]  →  unknown" in user
    # Block lands AFTER the draft body — annotation anchors read the draft above it.
    assert user.index("DRAFT TO CRITIQUE") < user.index("# Citation Liveness Context")


def test_build_critique_messages_omits_section_when_citations_none_or_empty():
    for citations in (None, []):
        messages = critique._build_critique_messages(
            draft_markdown="# Draft",
            sources_collected=[{"url": "https://x", "title": "x"}],
            mission_context={"title": "T", "objective": "O"},
            citations=citations,
        )
        assert "# Citation Liveness Context" not in messages[1]["content"]


def test_build_critique_messages_uses_shared_normalised_source_set():
    """S82 m05 carry — the builder resolves membership via the SHARED
    build_source_url_set so critique and the liveness gate agree.
    """

    import inspect

    src = inspect.getsource(critique._build_critique_messages)
    assert "build_source_url_set(" in src


# ---------------------------------------------------------------------------
# 12. Adversarial regression pins (s85-m01 skeptic round — decision #823).
# Each locks a confirmed finding from the pre-land adversarial verification.
# ---------------------------------------------------------------------------


class _RaisingContent:
    @property
    def content(self):  # noqa: D401 — pathological model return
        raise ValueError("boom-content")


class _RaisingTokenUsage:
    content = f"{CRITIQUE_OUTPUT_DELIMITER}\n" + json.dumps({"annotations": []})

    @property
    def token_usage(self):  # noqa: D401 — pathological provider envelope
        raise ValueError("boom-token-usage")


def test_critique_never_raises_on_pathological_model_return():
    """MUST-NEVER-RAISE: response/token extraction is inside the try, so a model
    whose .content raises (or a non-numeric token_usage) degrades, never crashes.
    """

    r = run_critique(_DRAFT, [], _mission_ctx(), model=lambda **k: _RaisingContent())
    _assert_safe_error(
        r.error,
        category="execution",
        exception_class="ValueError",
    )
    assert r.output_markdown == _DRAFT

    class _BadTU:
        content = f"{CRITIQUE_OUTPUT_DELIMITER}\n" + json.dumps({"annotations": []})
        token_usage = {"input_tokens": "not-a-number"}

    r2 = run_critique(_DRAFT, [], _mission_ctx(), model=lambda **k: _BadTU())
    # Must not raise; either parses fine or degrades with error — never crashes.
    assert isinstance(r2, CritiqueResult)


def test_critique_marks_usage_incomplete_when_usage_accessor_raises():
    """A returned response is paid work even when its usage property is broken."""

    result = run_critique(
        _DRAFT,
        [],
        _mission_ctx(),
        model=lambda **kwargs: _RaisingTokenUsage(),
    )

    _assert_safe_error(
        result.error,
        category="execution",
        exception_class="ValueError",
    )
    assert result.output_markdown == _DRAFT
    assert result.token_usage == {"input": 0, "output": 0, "total": 0}
    assert result.token_usage_complete is False


def test_critique_guard_protects_setext_heading():
    """A setext heading (title + === / --- underline) must be protected: an
    anchor spanning it is SKIPPED, not deleted with error=None.
    """

    draft = "# ATX\n\nExecutive Summary\n=================\n\n" + "Body prose. " * 20 + "\n"
    anns = [{"anchor": "Executive Summary\n=================", "verdict": "unsupported"}]
    r = run_critique(draft, [], _mission_ctx(), model=MagicMock(return_value=_mock_response(_annotations_raw(anns))))
    assert r.error is None
    assert r.gap_notes_inserted == 0
    assert "Executive Summary" in r.output_markdown


def test_critique_guard_protects_blockquoted_heading():
    draft = "> ## Risks in the market\n\n" + "Prose sentence here that is long. " * 10 + "\n"
    anns = [{"anchor": "> ## Risks in the market", "verdict": "unsupported"}]
    r = run_critique(draft, [], _mission_ctx(), model=MagicMock(return_value=_mock_response(_annotations_raw(anns))))
    assert r.error is None
    assert r.gap_notes_inserted == 0
    assert "Risks in the market" in r.output_markdown


def test_guard_rejects_setext_heading_deletion_directly():
    draft = "# ATX\n\nExecutive Summary\n=================\n\nBody.\n"
    start = draft.index("Executive")
    end = draft.index("\n\nBody")
    candidate = draft[:start] + GAP_NOTE_DEFAULT + draft[end:]
    with pytest.raises(CritiqueTransformError):
        critique._assert_bounded_transform(draft, candidate, [(start, end, GAP_NOTE_DEFAULT)])


def test_critique_applies_dominant_single_claim_gap_note_no_false_hollow():
    """A single unsupported claim that dominates a short report must still be
    gap-noted — the removed 0.5 hollow floor previously rejected this legit case.
    """

    claim = (
        "The vendor's proprietary reactor achieves a sustained net-energy gain of four hundred "
        "percent under continuous commercial operation, a figure absent from every collected source."
    )
    draft = "## Findings\n\n" + claim + "\n"
    anns = [{"anchor": claim, "verdict": "unsupported"}]
    r = run_critique(draft, [], _mission_ctx(), model=MagicMock(return_value=_mock_response(_annotations_raw(anns))))
    assert r.error is None
    assert r.gap_notes_inserted == 1
    assert GAP_NOTE_DEFAULT in r.output_markdown


def test_outer_whitespace_is_trimmed_for_audit_but_skipped():
    """A padded quote cannot delete a shorter incidental draft substring."""

    draft = "Please read the report carefully before deciding anything today.\n"
    anns = [{"anchor": "   the report   ", "verdict": "unsupported"}]
    r = run_critique(draft, [], _mission_ctx(), model=MagicMock(return_value=_mock_response(_annotations_raw(anns))))
    assert r.error is None
    assert r.gap_notes_inserted == 0
    assert r.output_markdown == draft
    assert r.annotations[0].anchor == "the report"
    assert r.annotations[0].applied is False
    assert r.annotations[0].reason == "anchor_outer_whitespace"


def test_applied_anchor_is_trimmed_for_tracelab_exact_claim_contract():
    """An applied telemetry anchor has no outer whitespace under TL 043."""

    claim = (
        "The control URL returns HTTP 404 and is intentionally retained as "
        "negative evidence."
    )
    draft = f"## Control\n\n{claim}\n"
    anns = [{"anchor": claim, "verdict": "unsupported"}]

    r = run_critique(
        draft,
        [],
        _mission_ctx(),
        model=MagicMock(return_value=_mock_response(_annotations_raw(anns))),
    )

    assert r.error is None
    assert r.gap_notes_inserted == 1
    assert r.annotations[0].applied is True
    assert r.annotations[0].anchor == claim
    assert r.annotations[0].anchor == r.annotations[0].anchor.strip()


def test_critique_stray_leading_json_object_does_not_shadow_real_annotations():
    """A stray leading dict (no annotations key, or an empty annotations list)
    must not shadow the real annotations block — real findings must still apply.
    """

    draft = "# R\n\nGood supported sentence about apples here.\n\nFabricated claim with no source at all.\n"
    real = json.dumps(
        {"claim_count": 1, "annotations": [{"anchor": "Fabricated claim with no source at all.", "verdict": "hallucinated"}]}
    )
    for stray in ('{"meta": 1}', '{"annotations": []}'):
        raw = f"{CRITIQUE_OUTPUT_DELIMITER}\n{stray}\n{real}"
        r = run_critique(draft, [], _mission_ctx(), model=lambda _raw=raw, **k: _raw)
        assert r.error is None, (stray, r.error)
        assert r.gap_notes_inserted == 1, stray
        assert "Good supported sentence about apples here." in r.output_markdown
        assert "Fabricated claim with no source at all." not in r.output_markdown


def test_critique_grounding_aggregate_budget_hard_bounded():
    """The grounding block must never exceed GROUNDING_TOTAL_BUDGET_CHARS, even
    with thousands of sources (the 300-char floor previously let it overflow).
    """

    from deepsearch.agent.critique import GROUNDING_TOTAL_BUDGET_CHARS

    for n in (534, 2000):
        sources = [{"url": f"https://d/{i}", "title": "D", "snippet": "s", "body": "q" * 5000} for i in range(n)]
        prepared = critique._prepare_sources(sources)
        total = sum(len(p["snippet"]) for p in prepared)
        assert total <= GROUNDING_TOTAL_BUDGET_CHARS, (n, total)


def test_live_cited_sources_are_selected_first_with_exact_drop_telemetry():
    """A live cited source at the tail survives a tight grounding budget."""

    cited_url = "https://primary.example/cited"
    sources = [
        {"url": "https://other.example/one", "title": "One", "body": "a" * 200},
        {"url": "https://other.example/two", "title": "Two", "body": "b" * 200},
        {"url": cited_url, "title": "Cited", "body": "c" * 200},
    ]
    citations = [
        CitationAnnotation(
            start_index=0,
            end_index=1,
            url=cited_url,
            live=True,
        )
    ]

    prepared, telemetry = critique._prepare_sources_with_telemetry(
        sources,
        citations=citations,
        per_source_cap=80,
        total_budget=140,
    )

    assert prepared[0]["url"] == cited_url
    assert telemetry["sources_available"] == 3
    assert telemetry["sources_included"] == len(prepared) == 1
    assert telemetry["sources_dropped"] == 2
    assert telemetry["cited_live_available"] == 1
    assert telemetry["cited_live_included"] == 1
    assert telemetry["source_block_chars"] <= 140


def test_critique_context_budget_and_truncation_are_observable():
    draft = "d" * (critique.CRITIQUE_DRAFT_MAX_CHARS + 101)
    telemetry: dict[str, int] = {}

    messages = critique._build_critique_messages(
        draft_markdown=draft,
        sources_collected=[
            {
                "url": "https://primary.example/source",
                "title": "Primary",
                "body": "e" * 50_000,
            }
        ],
        mission_context=_mission_ctx(),
        grounding_telemetry=telemetry,
    )

    assert len(messages[1]["content"]) <= critique.CRITIQUE_CONTEXT_MAX_CHARS
    assert telemetry["draft_chars_available"] == len(draft)
    assert telemetry["draft_chars_included"] == critique.CRITIQUE_DRAFT_MAX_CHARS
    assert telemetry["draft_chars_dropped"] == 101
    assert telemetry["sources_included"] == 1
    assert telemetry["sources_truncated"] == 1
    assert telemetry["context_chars"] == len(messages[1]["content"])


def test_run_critique_surfaces_grounding_telemetry():
    fake_model = MagicMock(return_value=_mock_response(_annotations_raw([])))

    result = run_critique(
        draft_markdown=_DRAFT,
        sources_collected=_sources(),
        mission_context=_mission_ctx(),
        model=fake_model,
    )

    assert result.error is None
    assert result.grounding_telemetry["sources_included"] == len(_sources())
    assert result.grounding_telemetry["sources_dropped"] == 0
    assert result.grounding_telemetry["grounding_context_complete"] == 1
    assert result.grounding_telemetry["source_urls_incomplete"] == 0
    assert result.grounding_telemetry["context_budget_chars"] == (
        critique.CRITIQUE_CONTEXT_MAX_CHARS
    )


# ---------------------------------------------------------------------------
# S92 m07 — negative grounding claims require a complete evidence view. R2
# proved the critic was deleting supported claims whose passages fell just
# beyond the 2,400-char body prefix or in source rows omitted by the aggregate
# budget. A bounded excerpt cannot prove absence; hallucinated URLs remain the
# independent, actionable exception.
# ---------------------------------------------------------------------------


def test_truncated_cited_source_cannot_authorize_unsupported_gap_note():
    claim = "Baymard observed 219 sessions across more than 5,550 research hours."
    draft = f"# Findings\n\n{claim}\n"
    url = "https://baymard.example/search"
    start = draft.index(claim)
    citation = CitationAnnotation(
        start_index=start,
        end_index=start + len(claim),
        url=url,
        live=True,
    )
    sources = [
        {
            "url": url,
            "title": "Search UX",
            "body": "navigation " * 300 + claim,
        }
    ]
    annotations = [
        {
            "anchor": claim,
            "verdict": "unsupported",
            "citation_urls": [url],
        }
    ]

    result = run_critique(
        draft,
        sources,
        _mission_ctx(),
        model=MagicMock(return_value=_mock_response(_annotations_raw(annotations))),
        citations=[citation],
    )

    assert result.error is None
    assert result.output_markdown == draft
    assert result.gap_notes_inserted == 0
    assert result.annotations[0].reason == "preserved_incomplete_grounding"
    assert result.grounding_telemetry["sources_truncated"] == 1
    assert result.grounding_telemetry["source_urls_incomplete"] == 1
    assert result.grounding_telemetry["grounding_context_complete"] == 0


@pytest.mark.parametrize(
    "source",
    [
        {
            "url": "https://evidence.example/snippet-only",
            "title": "Snippet only",
            "snippet": "A search-result excerpt is not the complete source.",
        },
        {
            "url": "https://evidence.example/non-readable",
            "title": "Non-readable payload",
            "body": "Text exists, but retrieval classified it as a snippet.",
            "content_quality": "snippet",
        },
        {
            "url": "https://evidence.example/upstream-truncated",
            "title": "Truncated fetch",
            "body": "Only the leading portion of the fetched page is available.",
            "content_quality": "readable_body",
            "body_truncated": True,
        },
    ],
    ids=["snippet-only", "non-readable-quality", "upstream-truncated"],
)
def test_incomplete_source_provenance_cannot_authorize_negative_claim(
    source: dict,
) -> None:
    claim = "The collected source contains no measurements of field reliability."
    draft = f"# Findings\n\n{claim}\n"
    start = draft.index(claim)
    citation = CitationAnnotation(
        start_index=start,
        end_index=start + len(claim),
        url=source["url"],
        live=True,
    )
    annotations = [
        {
            "anchor": claim,
            "verdict": "unsupported",
            "citation_urls": [source["url"]],
        }
    ]

    result = run_critique(
        draft,
        [source],
        _mission_ctx(),
        model=MagicMock(return_value=_mock_response(_annotations_raw(annotations))),
        citations=[citation],
    )

    assert result.error is None
    assert result.output_markdown == draft
    assert result.gap_notes_inserted == 0
    assert result.annotations[0].reason == "preserved_incomplete_grounding"
    assert result.grounding_telemetry["sources_truncated"] == 0
    assert result.grounding_telemetry["sources_dropped"] == 0
    assert result.grounding_telemetry["sources_provenance_incomplete"] == 1
    assert result.grounding_telemetry["source_urls_incomplete"] == 1
    assert result.grounding_telemetry["grounding_context_complete"] == 0


def test_full_cited_url_cannot_prove_absence_when_corroborating_row_is_omitted():
    """Support can live outside the cited URL, so a partial corpus proves no negative."""

    claim = "The field workflow retains asset identity while connectivity is unavailable."
    draft = f"# Findings\n\n{claim}\n"
    cited_url = "https://evidence.example/cited"
    corroborating_url = "https://evidence.example/corroborating"
    start = draft.index(claim)
    citation = CitationAnnotation(
        start_index=start,
        end_index=start + len(claim),
        url=cited_url,
        live=True,
    )
    sources = [
        {
            "url": cited_url,
            "title": "Cited source",
            "body": "A complete but non-corroborating source body.",
            "content_quality": "readable_body",
        },
        *[
            {
                "url": f"https://evidence.example/filler-{index}",
                "title": f"Filler {index}",
                "body": "x" * 2_300,
                "content_quality": "readable_body",
            }
            for index in range(80)
        ],
        {
            "url": corroborating_url,
            "title": "Corroborating source",
            "body": f"Independent evidence states: {claim}",
            "content_quality": "readable_body",
        },
    ]
    annotations = [
        {
            "anchor": claim,
            "verdict": "unsupported",
            "citation_urls": [cited_url],
        }
    ]
    fake_model = MagicMock(
        return_value=_mock_response(_annotations_raw(annotations))
    )

    result = run_critique(
        draft,
        sources,
        _mission_ctx(),
        model=fake_model,
        citations=[citation],
    )

    prompt = fake_model.call_args.kwargs["messages"][1]["content"]
    assert "A complete but non-corroborating source body." in prompt
    assert corroborating_url not in prompt
    assert result.output_markdown == draft
    assert result.gap_notes_inserted == 0
    assert result.annotations[0].reason == "preserved_incomplete_grounding"
    assert result.grounding_telemetry["cited_dropped"] == 0
    assert result.grounding_telemetry["source_urls_fully_visible"] > 0
    assert result.grounding_telemetry["sources_provenance_incomplete"] == 0
    assert result.grounding_telemetry["sources_dropped"] > 0
    assert result.grounding_telemetry["grounding_context_complete"] == 0


@pytest.mark.parametrize(
    ("live", "sources"),
    [
        (
            True,
            [
                {
                    "url": "https://evidence.example/unrelated",
                    "title": "Unrelated source",
                    "body": "A complete source unrelated to the cited claim.",
                    "content_quality": "readable_body",
                }
            ],
        ),
        (
            None,
            [
                {
                    "url": "https://evidence.example/unrelated",
                    "title": "Unrelated source",
                    "body": "A complete source unrelated to the cited claim.",
                    "content_quality": "readable_body",
                }
            ],
        ),
        (True, []),
    ],
    ids=["reachable-orphan", "unknown", "empty-source-corpus"],
)
def test_cited_url_missing_from_sources_cannot_authorize_negative_claim(
    live: bool | None,
    sources: list[dict],
) -> None:
    claim = "The uncollected citation contains no evidence for offline operation."
    draft = f"# Findings\n\n{claim}\n"
    missing_url = "https://evidence.example/uncollected"
    start = draft.index(claim)
    citation = CitationAnnotation(
        start_index=start,
        end_index=start + len(claim),
        url=missing_url,
        live=live,
    )
    annotations = [
        {
            "anchor": claim,
            "verdict": "unsupported",
            "citation_urls": [missing_url],
        }
    ]

    result = run_critique(
        draft,
        sources,
        _mission_ctx(),
        model=MagicMock(return_value=_mock_response(_annotations_raw(annotations))),
        citations=[citation],
    )

    assert result.error is None
    assert result.output_markdown == draft
    assert result.gap_notes_inserted == 0
    assert result.annotations[0].reason == "preserved_incomplete_grounding"
    assert result.grounding_telemetry["cited_available"] == 0
    assert result.grounding_telemetry["cited_missing_from_sources"] == 1
    assert result.grounding_telemetry["grounding_context_complete"] == 0


def test_mixed_live_and_hallucinated_overlap_cannot_delete_supported_claim():
    claim = "The primary documentation supports offline asset-identity retention."
    draft = f"# Findings\n\n{claim}\n"
    live_url = "https://evidence.example/primary"
    hallucinated_url = "https://fabricated.example/claim"
    start = draft.index(claim)
    citations = [
        CitationAnnotation(
            start_index=start,
            end_index=start + len(claim),
            url=live_url,
            live=True,
        ),
        CitationAnnotation(
            start_index=start,
            end_index=start + len(claim),
            url=hallucinated_url,
            live=False,
        ),
    ]
    annotations = [
        {
            "anchor": claim,
            "verdict": "hallucinated",
            "citation_urls": [hallucinated_url],
        }
    ]

    result = run_critique(
        draft,
        [
            {
                "url": live_url,
                "title": "Primary documentation",
                "body": claim,
                "content_quality": "readable_body",
            }
        ],
        _mission_ctx(),
        model=MagicMock(return_value=_mock_response(_annotations_raw(annotations))),
        citations=citations,
    )

    assert result.error is None
    assert result.output_markdown == draft
    assert result.gap_notes_inserted == 0
    assert result.annotations[0].reason == "preserved_incomplete_grounding"
    assert result.grounding_telemetry["cited_missing_from_sources"] == 1


def test_unrelated_requested_hallucinated_url_cannot_unlock_deletion():
    supported_claim = "The primary documentation supports offline operation."
    unrelated_claim = "A separate fabricated source makes another claim."
    draft = f"# Findings\n\n{supported_claim}\n\n{unrelated_claim}\n"
    live_url = "https://evidence.example/primary"
    hallucinated_url = "https://fabricated.example/unrelated"
    supported_start = draft.index(supported_claim)
    unrelated_start = draft.index(unrelated_claim)
    citations = [
        CitationAnnotation(
            start_index=supported_start,
            end_index=supported_start + len(supported_claim),
            url=live_url,
            live=True,
        ),
        CitationAnnotation(
            start_index=unrelated_start,
            end_index=unrelated_start + len(unrelated_claim),
            url=hallucinated_url,
            live=False,
        ),
    ]
    annotations = [
        {
            "anchor": supported_claim,
            "verdict": "hallucinated",
            "citation_urls": [hallucinated_url],
        }
    ]

    result = run_critique(
        draft,
        [
            {
                "url": live_url,
                "title": "Primary documentation",
                "body": supported_claim,
                "content_quality": "readable_body",
            }
        ],
        _mission_ctx(),
        model=MagicMock(return_value=_mock_response(_annotations_raw(annotations))),
        citations=citations,
    )

    assert result.error is None
    assert result.output_markdown == draft
    assert result.gap_notes_inserted == 0
    assert result.annotations[0].reason == "preserved_incomplete_grounding"
    assert result.annotations[0].citation_urls == (live_url, hallucinated_url)


def test_uncited_unsupported_claim_is_preserved_when_corpus_view_is_incomplete():
    claim = "MSI Data supports offline HVAC workflows in disconnected environments."
    draft = f"# Findings\n\n{claim}\n"
    sources = [
        {
            "url": "https://example.com/large-source",
            "title": "Large source",
            "body": "x" * (SOURCE_GROUNDING_MAX_CHARS + 1),
        }
    ]
    annotations = [{"anchor": claim, "verdict": "unsupported"}]

    result = run_critique(
        draft,
        sources,
        _mission_ctx(),
        model=MagicMock(return_value=_mock_response(_annotations_raw(annotations))),
    )

    assert result.output_markdown == draft
    assert result.gap_notes_inserted == 0
    assert result.annotations[0].reason == "preserved_incomplete_grounding"


def test_hallucinated_annotation_still_applies_with_incomplete_grounding():
    claim = "A fabricated source claims a nonexistent field-service standard."
    draft = f"# Findings\n\n{claim}\n"
    hallucinated_url = "https://fabricated.example/standard"
    start = draft.index(claim)
    citation = CitationAnnotation(
        start_index=start,
        end_index=start + len(claim),
        url=hallucinated_url,
        live=False,
    )
    sources = [
        {
            "url": "https://real.example/large-source",
            "title": "Real source",
            "body": "x" * (SOURCE_GROUNDING_MAX_CHARS + 1),
        }
    ]
    annotations = [
        {
            "anchor": claim,
            "verdict": "hallucinated",
            "citation_urls": [hallucinated_url],
        }
    ]

    result = run_critique(
        draft,
        sources,
        _mission_ctx(),
        model=MagicMock(return_value=_mock_response(_annotations_raw(annotations))),
        citations=[citation],
    )

    assert result.error is None
    assert result.gap_notes_inserted == 1
    assert GAP_NOTE_DEFAULT in result.output_markdown
    assert result.annotations[0].reason == "applied_gap_note"


# ---------------------------------------------------------------------------
# S88 lever 2 — deterministic rule-7 backstop. A claim whose sole/every
# overlapping citation is DEAD-with-no-collected-content is PRESERVED (the
# critique had no basis; the dead citation is flagged unverified downstream).
# The anti-gaming line: DEAD-with-FULL-body damage STANDS (state B); UNKNOWN
# and HALLUCINATED are never suppressed (silent-bless / rule-7 violations).
# ---------------------------------------------------------------------------

_L2_DRAFT = (
    "# Fitment\n\n## VCdb\n\n"
    "The VCdb contains over 60,000 unique year make model combinations.\n"
)
_L2_CLAIM = "The VCdb contains over 60,000 unique year make model combinations."
_DEAD_URL = "https://autocare.example/vcdb"
_LIVE_URL = "https://autocare.example/live"


def _l2_citation(url: str, live, *, offset: int = 2, span: int = 8):
    start = _L2_DRAFT.index(_L2_CLAIM)
    return CitationAnnotation(
        start_index=start + offset,
        end_index=start + offset + span,
        url=url,
        live=live,
    )


def _l2_run(sources, citations):
    anns = [{"anchor": _L2_CLAIM, "verdict": "unsupported", "note": "no basis"}]
    fake_model = MagicMock(return_value=_mock_response(_annotations_raw(anns)))
    return run_critique(
        draft_markdown=_L2_DRAFT,
        sources_collected=sources,
        mission_context=_mission_ctx(),
        model=fake_model,
        citations=citations,
    )


def test_dead_empty_body_helper_returns_only_dead_with_no_content():
    sources = [
        {"url": _DEAD_URL, "title": "dead-empty", "body": "", "snippet": ""},
        {"url": "https://x/full", "title": "dead-full", "body": "full body text"},
        {"url": "https://x/unk", "title": "unknown-empty", "body": "", "snippet": ""},
    ]
    citations = [
        _l2_citation(_DEAD_URL, False),                 # DEAD + empty  -> returned
        _l2_citation("https://x/full", False),          # DEAD + full   -> excluded (state B)
        _l2_citation("https://x/unk", None),            # UNKNOWN       -> excluded
        _l2_citation("https://x/halluc", False),        # HALLUCINATED  -> excluded (not in sources)
    ]
    result = critique._dead_empty_body_citation_urls(citations, sources)
    assert result == {_DEAD_URL}


def test_lever2_preserves_claim_cited_only_to_dead_empty_body_source():
    """SKIP: sole overlapping citation is dead-with-no-content -> gap note NOT
    applied, claim preserved, outcome auditable, T2 not incremented."""
    sources = [{"url": _DEAD_URL, "title": "VCdb", "body": "", "snippet": ""}]
    result = _l2_run(sources, [_l2_citation(_DEAD_URL, False)])
    assert result.error is None
    assert result.gap_notes_inserted == 0
    assert GAP_NOTE_DEFAULT not in result.output_markdown
    assert _L2_CLAIM in result.output_markdown  # preserved verbatim
    summary = result.annotation_summary()
    assert summary["applied"] == 0
    assert summary["by_reason"].get("preserved_dead_no_basis") == 1


def test_lever2_damage_stands_for_dead_citation_with_full_body_state_b():
    """STATE B (anti-gaming): a DEAD citation whose body WAS in context is a
    genuine judgment — the gap note MUST still apply."""
    sources = [{"url": _DEAD_URL, "title": "VCdb", "body": "a full, present body"}]
    result = _l2_run(sources, [_l2_citation(_DEAD_URL, False)])
    assert result.error is None
    assert result.gap_notes_inserted == 1
    assert GAP_NOTE_DEFAULT in result.output_markdown


def test_unknown_empty_source_preserves_claim_due_to_incomplete_corpus():
    """An unknown citation with no body cannot support a corpus-wide negative."""

    sources = [{"url": "https://x/unk", "title": "u", "body": "", "snippet": ""}]
    result = _l2_run(sources, [_l2_citation("https://x/unk", None)])
    assert result.error is None
    assert result.gap_notes_inserted == 0
    assert result.annotations[0].reason == "preserved_incomplete_grounding"
    assert result.grounding_telemetry["grounding_context_complete"] == 0


def test_lever2_does_not_suppress_hallucinated_cited_claim():
    """HALLUCINATED (live=False, not in sources) must be annotated per rule 7 —
    never suppressed."""
    sources = [{"url": "https://x/other", "title": "o", "body": "present"}]
    result = _l2_run(sources, [_l2_citation("https://x/halluc", False)])
    assert result.error is None
    assert result.gap_notes_inserted == 1


def test_mixed_empty_and_live_sources_preserve_claim_due_to_incomplete_corpus():
    """One visible URL cannot prove absence while another source body is missing."""

    sources = [
        {"url": _DEAD_URL, "title": "d", "body": "", "snippet": ""},
        {"url": _LIVE_URL, "title": "l", "body": "supporting live body text"},
    ]
    citations = [
        _l2_citation(_DEAD_URL, False, offset=2, span=6),
        _l2_citation(_LIVE_URL, True, offset=10, span=6),
    ]
    result = _l2_run(sources, citations)
    assert result.error is None
    assert result.gap_notes_inserted == 0
    assert result.annotations[0].reason == "preserved_incomplete_grounding"
    assert result.grounding_telemetry["grounding_context_complete"] == 0


def test_lever2_dead_empty_body_is_classified_dead_so_downstream_flags_it():
    """RC1 composition: the exact class lever 2 suppresses is `dead` under the
    SAME classifier the finalize dead-flag path uses, so a suppressed claim's
    citation is independently flagged unverified downstream (never silently
    blessed). The unknown negative-twin is NOT dead (justifying dead-only)."""
    from deepsearch.agent.citation_liveness import (
        CLASSIFICATION_DEAD,
        build_source_url_set,
        classify_citation,
    )

    sources = [{"url": _DEAD_URL, "body": "", "snippet": ""}]
    url_set = build_source_url_set(sources)
    assert classify_citation(live=False, url=_DEAD_URL, source_url_set=url_set) == (
        CLASSIFICATION_DEAD
    )
    # unknown twin is not dead -> would NOT be flagged downstream -> not suppressed
    assert classify_citation(live=None, url=_DEAD_URL, source_url_set=url_set) != (
        CLASSIFICATION_DEAD
    )


# ---------------------------------------------------------------------------
# S88 lever 1 — front ALL draft-cited sources (not only live==True) ahead of
# uncited sources when packing the fixed grounding budget, so a claim cited to
# a present-but-non-live source is judged against its own evidence instead of
# flagged for a packer eviction it never saw. REORDER within the same budget
# (flat token cost). The cited_* telemetry makes every eviction auditable and
# drives the measured re-run gate (cited_dropped == 0).
# ---------------------------------------------------------------------------


def test_lever1_fronts_non_live_cited_source_over_uncited():
    """A DEAD-cited (live=False) source at the tail now survives a tight budget
    that drops uncited sources — the fix for the R001 dropped-cited false
    positives — and cited_dropped is 0."""
    dead_cited = "https://primary.example/dead-cited"
    sources = [
        {"url": "https://other.example/one", "title": "One", "body": "a" * 200},
        {"url": "https://other.example/two", "title": "Two", "body": "b" * 200},
        {"url": dead_cited, "title": "DeadCited", "body": "c" * 60},
    ]
    citations = [CitationAnnotation(start_index=0, end_index=1, url=dead_cited, live=False)]

    prepared, telem = critique._prepare_sources_with_telemetry(
        sources, citations=citations, per_source_cap=80, total_budget=200
    )

    included = [p["url"] for p in prepared]
    assert dead_cited in included                     # fronted despite live=False
    assert telem["cited_available"] == 1
    assert telem["cited_included"] == 1
    assert telem["cited_dropped"] == 0
    assert telem["sources_dropped"] >= 1              # an uncited source was evicted
    assert telem["source_block_chars"] <= 200         # reorder did NOT grow the budget


def test_lever1_reorder_does_not_grow_budget():
    """Fronting more cited sources must never exceed GROUNDING_TOTAL_BUDGET_CHARS."""
    from deepsearch.agent.critique import GROUNDING_TOTAL_BUDGET_CHARS

    sources = [
        {"url": f"https://d/{i}", "title": "D", "body": "q" * 5000} for i in range(600)
    ]
    citations = [
        CitationAnnotation(start_index=0, end_index=1, url=f"https://d/{i}", live=False)
        for i in range(600)
    ]
    prepared, telem = critique._prepare_sources_with_telemetry(
        sources, citations=citations
    )
    assert telem["source_block_chars"] <= GROUNDING_TOTAL_BUDGET_CHARS
    total = sum(len(p["snippet"]) for p in prepared)
    assert total <= GROUNDING_TOTAL_BUDGET_CHARS


def test_lever1_cited_dropped_is_exposed_when_cited_overflow_budget():
    """When cited sources exceed the budget, the eviction is AUDITABLE — the
    re-run gate reads cited_dropped, and a nonzero value means the reorder alone
    could not seat every cited source (the honest limit)."""
    sources = [
        {"url": f"https://c/{i}", "title": "C", "body": "z" * 5000} for i in range(200)
    ]
    citations = [
        CitationAnnotation(start_index=0, end_index=1, url=f"https://c/{i}", live=False)
        for i in range(200)
    ]
    _, telem = critique._prepare_sources_with_telemetry(sources, citations=citations)
    assert telem["cited_available"] == 200
    assert telem["cited_dropped"] > 0
    assert telem["cited_included"] + telem["cited_dropped"] == telem["cited_available"]


def test_lever1_preserves_live_cited_priority_over_non_live_cited():
    """Within the cited front, a LIVE-cited full body is seated before a
    non-live cited body — fronting dead-cited must not starve live-cited backing."""
    live_url = "https://primary.example/live"
    dead_url = "https://primary.example/dead"
    sources = [
        {"url": dead_url, "title": "Dead", "body": "d" * 200},   # non-live, earlier in list
        {"url": live_url, "title": "Live", "body": "l" * 200},   # live, later in list
    ]
    citations = [
        CitationAnnotation(start_index=0, end_index=1, url=dead_url, live=False),
        CitationAnnotation(start_index=2, end_index=3, url=live_url, live=True),
    ]
    prepared, _ = critique._prepare_sources_with_telemetry(
        sources, citations=citations, per_source_cap=80, total_budget=140
    )
    assert prepared[0]["url"] == live_url  # live-cited wins the scarce first slot
