"""S92 m05 — strict lossless producer for TraceLab LEDGER-2.

These tests pin WHY the ledger audit is separate from generic tool telemetry:
the 250-row diagnostic sample may omit late failures and used to truncate URLs
at 200 characters, either of which changes TraceLab source identity.
"""

from __future__ import annotations

import pytest

from deepsearch.agent import codeagent_runner as runner
from deepsearch.agent.codeagent_finalize_node import (
    _build_synthesis_telemetry,
    _enforce_ledger_projection_budget,
    _ledger_projection_upper_bound,
    _merge_ledger_outcomes,
    _merge_tool_outcomes,
)
from deepsearch.agent.codeagent_runner import (
    LEDGER_RECORD_MAX_ENTRIES,
    CodeAgentToolBundle,
    _canonicalize_sources_collected,
)
from deepsearch.worker.writer import ExecutionMetadata


def _fetch_record(
    url: str,
    *,
    status: str = "error",
    status_code: int | None = 503,
    error_category: str | None = "http_error",
) -> dict:
    return {
        "tool": "source_fetch",
        "url": url,
        "status": status,
        "status_code": status_code,
        "error_category": error_category,
        "result_count": 0,
        "error": "raw diagnostic detail must never enter ledger_records",
    }


def _liveness_record(
    url: str,
    *,
    status: str = "error",
    status_code: int | None = 0,
    error_category: str | None = "timeout",
    alive: bool = False,
) -> dict:
    return {
        "tool": "url_liveness",
        "url": url,
        "status": status,
        "status_code": status_code,
        "error_category": error_category,
        "alive": alive,
        "diagnostic_only": "not part of the ledger schema",
    }


def _strict_fetch_record(url: str) -> dict:
    """Return the exact five-field TraceLab source-fetch envelope."""

    return {
        "tool": "source_fetch",
        "url": url,
        "status": "error",
        "status_code": 503,
        "error_category": "http_error",
    }


def _strict_liveness_record(url: str) -> dict:
    """Return the exact six-field TraceLab liveness envelope."""

    return {
        "tool": "url_liveness",
        "url": url,
        "status": "error",
        "status_code": 0,
        "error_category": "timeout",
        "alive": False,
    }


def test_empty_audit_is_explicitly_complete() -> None:
    outcomes = CodeAgentToolBundle(tools=[]).tool_outcomes()

    assert outcomes["ledger_records"] == []
    assert outcomes["ledger_records_truncated"] == 0


def test_late_relevant_attempt_survives_generic_250_record_cap() -> None:
    calls = [
        {
            "tool": "web_search",
            "query": f"diagnostic-{index}",
            "status": "ok",
            "error_category": None,
            "result_count": 1,
        }
        for index in range(300)
    ]
    calls.append(
        _fetch_record(
            "https://dead.example.test/late-failure",
            status_code=404,
            error_category="http_404",
        )
    )

    outcomes = CodeAgentToolBundle(tools=[], call_log=calls).tool_outcomes()

    assert len(outcomes["records"]) == 250
    assert outcomes["records_truncated"] == 51
    assert outcomes["ledger_records_truncated"] == 0
    assert outcomes["ledger_records"] == [
        {
            "tool": "source_fetch",
            "url": "https://dead.example.test/late-failure",
            "status": "error",
            "status_code": 404,
            "error_category": "http_404",
        }
    ]


def test_long_url_is_canonical_lossless_and_shares_tracelab_fixture_shape() -> None:
    path = "source-" + ("a" * 240)
    raw_url = (
        f"HTTPS://LONG.Example.TEST:443/{path}?username=researcher&view=complete"
    )
    canonical_url = (
        f"https://long.example.test/{path}?username=researcher&view=complete"
    )
    calls = [
        _fetch_record(
            raw_url,
            status="ok",
            status_code=200,
            error_category=None,
        ),
        _liveness_record(
            canonical_url,
            status="ok",
            status_code=200,
            error_category=None,
            alive=True,
        ),
    ]

    outcomes = CodeAgentToolBundle(tools=[], call_log=calls).tool_outcomes()

    assert len(canonical_url) > 200
    assert outcomes["ledger_records_truncated"] == 0
    assert outcomes["ledger_records"] == [
        {
            "tool": "source_fetch",
            "url": canonical_url,
            "status": "ok",
            "status_code": 200,
            "error_category": None,
        },
        {
            "tool": "url_liveness",
            "url": canonical_url,
            "status": "ok",
            "status_code": 200,
            "error_category": None,
            "alive": True,
        },
    ]
    assert set(outcomes["ledger_records"][0]) == {
        "tool",
        "url",
        "status",
        "status_code",
        "error_category",
    }
    assert set(outcomes["ledger_records"][1]) == {
        "tool",
        "url",
        "status",
        "status_code",
        "error_category",
        "alive",
    }
    # Generic telemetry retains its original compact diagnostic contract; it
    # is deliberately not a projection input.
    assert outcomes["records"][0]["url"] == raw_url[:200]


def test_raw_or_unknown_errors_collapse_to_secret_safe_taxonomy() -> None:
    outcomes = CodeAgentToolBundle(
        tools=[],
        call_log=[
            _fetch_record(
                "https://example.test/unsafe-target",
                error_category="non_public_address:192.0.2.10",
            ),
            _fetch_record(
                "https://example.test/raw-error",
                error_category="credential=must-not-survive",
            ),
        ],
    ).tool_outcomes()

    assert [
        record["error_category"] for record in outcomes["ledger_records"]
    ] == ["non_public_address", "unexpected"]
    assert "credential" not in repr(outcomes["ledger_records"])
    assert all("error" not in record for record in outcomes["ledger_records"])


def test_nonstandard_three_digit_http_status_remains_complete_evidence() -> None:
    """A real bot-wall 999 response must not dead-letter the whole batch."""

    outcomes = CodeAgentToolBundle(
        tools=[],
        call_log=[
            _fetch_record(
                "https://example.test/bot-wall",
                status_code=999,
                error_category="http_error",
            )
        ],
    ).tool_outcomes()

    assert outcomes["ledger_records"] == [
        {
            "tool": "source_fetch",
            "url": "https://example.test/bot-wall",
            "status": "error",
            "status_code": 999,
            "error_category": "http_error",
        }
    ]
    assert outcomes["ledger_records_truncated"] == 0


def test_invalid_or_overbound_urls_fail_loudly_without_clipping() -> None:
    overbound_url = "https://example.test/" + ("x" * 4_100)
    outcomes = CodeAgentToolBundle(
        tools=[],
        call_log=[
            _fetch_record(
                "https://user:credential@example.test/source"  # secret-scan: allow -- fixture
            ),
            _liveness_record(overbound_url),
        ],
    ).tool_outcomes()

    assert outcomes["ledger_records"] == []
    assert outcomes["ledger_records_truncated"] == 2


def test_userinfo_is_redacted_from_persisted_generic_diagnostics() -> None:
    """A loud ledger rejection must not leave the credential in metadata."""

    unsafe_url = "https://user:password@example.com/private"
    outcomes = CodeAgentToolBundle(
        tools=[],
        call_log=[_fetch_record(unsafe_url)],
    ).tool_outcomes()
    serialized = ExecutionMetadata(
        total_tokens=1,
        duration_seconds=1.0,
        synthesis_telemetry={"tool_outcomes": outcomes},
    ).to_dict()

    assert outcomes["ledger_records"] == []
    assert outcomes["ledger_records_truncated"] == 1
    assert outcomes["records"][0]["url"] == "[redacted-invalid-url]"
    assert unsafe_url not in repr(serialized)
    assert "password" not in repr(serialized)


def test_invalid_typed_fields_fail_loudly_instead_of_being_coerced() -> None:
    malformed = [
        _fetch_record("https://example.test/bool-code", status_code=True),
        _fetch_record("https://example.test/four-digit-code", status_code=1000),
        _liveness_record("https://example.test/non-bool-alive", alive=False),
    ]
    malformed[2]["alive"] = "false"

    outcomes = CodeAgentToolBundle(tools=[], call_log=malformed).tool_outcomes()

    assert outcomes["ledger_records"] == []
    assert outcomes["ledger_records_truncated"] == 3


def test_collected_sources_use_the_same_canonical_url_identity_as_ledger() -> None:
    """Compatible raw variants enrich one strict TraceLab source envelope."""

    sources = _canonicalize_sources_collected(
        [
            {
                "url": "HTTPS://Example.COM:443/path/../canonical?view=full",
                "title": "Canonical source",
            },
            {
                "url": "https://example.com/canonical?view=full",
                "snippet": "Compatible metadata remains attached.",
            },
        ]
    )

    assert sources == [
        {
            "url": "https://example.com/canonical?view=full",
            "title": "Canonical source",
            "snippet": "Compatible metadata remains attached.",
        }
    ]
    assert set(sources[0]) == {"url", "title", "snippet"}


@pytest.mark.parametrize(
    ("field", "first", "second"),
    (
        ("title", "First title", "Second title"),
        ("snippet", "First snippet", "Second snippet"),
        ("alive", True, False),
    ),
)
def test_canonical_duplicate_source_metadata_conflicts_fail_loudly(
    field: str,
    first: str | bool,
    second: str | bool,
) -> None:
    """Choosing one conflicting value would silently change ledger evidence."""

    with pytest.raises(
        runner.SourceCollectionValidationError,
        match=rf"conflicting {field} metadata",
    ):
        _canonicalize_sources_collected(
            [
                {
                    "url": "HTTPS://Example.COM:443/a/../canonical",
                    field: first,
                },
                {
                    "url": "https://example.com/canonical",
                    field: second,
                },
            ]
        )


def test_collected_source_userinfo_fails_without_leaking_url_text() -> None:
    """Embedded credentials cannot disappear or enter a blocked-result log."""

    with pytest.raises(runner.SourceCollectionValidationError) as raised:
        _canonicalize_sources_collected(
            [
                {
                    "url": "https://user:password@example.com/private",
                    "title": "Unsafe source",
                }
            ]
        )

    assert str(raised.value) == "sources_collected[0].url is invalid"
    assert "user" not in str(raised.value)
    assert "password" not in str(raised.value)


def test_ledger_budget_is_loud_and_independent_of_generic_budget() -> None:
    calls = [
        _fetch_record(f"https://example.test/fetch/{index}")
        for index in range(LEDGER_RECORD_MAX_ENTRIES + 1)
    ]

    outcomes = CodeAgentToolBundle(tools=[], call_log=calls).tool_outcomes()

    assert len(outcomes["ledger_records"]) == LEDGER_RECORD_MAX_ENTRIES
    assert outcomes["ledger_records_truncated"] == 1
    assert len(outcomes["records"]) == 250
    assert outcomes["records_truncated"] == 751


def test_recovery_merge_preserves_order_duplicates_and_loud_overflow() -> None:
    first_record = _strict_fetch_record("https://example.test/first")
    second_record = _strict_liveness_record("https://example.test/second")
    first = {
        "ledger_records": [first_record, first_record],
        "ledger_records_truncated": 0,
    }
    second = {
        "ledger_records": [second_record],
        "ledger_records_truncated": 2,
    }

    merged = _merge_ledger_outcomes(first, second, max_records=2)

    assert merged["ledger_records"] == [first_record, first_record]
    assert merged["ledger_records_truncated"] == 3


def test_recovery_merge_rejects_extra_fields_as_loud_incompleteness() -> None:
    malformed = {
        **_strict_fetch_record("https://example.test/malformed"),
        "error": "raw transport detail",
    }

    merged = _merge_ledger_outcomes(
        {
            "ledger_records": [malformed],
            "ledger_records_truncated": 0,
        },
        {
            "ledger_records": [],
            "ledger_records_truncated": 0,
        },
    )

    assert merged["ledger_records"] == []
    assert merged["ledger_records_truncated"] == 1


def test_recovery_merge_has_no_legacy_missing_envelope_fallback() -> None:
    valid = {
        "ledger_records": [],
        "ledger_records_truncated": 0,
    }

    merged = _merge_ledger_outcomes({}, valid)

    assert merged["ledger_records"] == []
    assert merged["ledger_records_truncated"] == 2


def test_complete_tool_outcome_merge_keeps_ledger_outside_diagnostic_cap() -> None:
    first_ledger = _strict_fetch_record("https://example.test/first")
    second_ledger = _strict_liveness_record("https://example.test/second")
    first = {
        "by_tool": {},
        "records": [{"tool": "web_search", "status": "ok"}] * 200,
        "records_truncated": 0,
        "ledger_records": [first_ledger],
        "ledger_records_truncated": 0,
    }
    second = {
        "by_tool": {},
        "records": [{"tool": "web_search", "status": "ok"}] * 100,
        "records_truncated": 0,
        "ledger_records": [second_ledger],
        "ledger_records_truncated": 0,
    }

    merged = _merge_tool_outcomes(first, second)

    assert len(merged["records"]) == 250
    assert merged["records_truncated"] == 50
    assert merged["ledger_records"] == [first_ledger, second_ledger]
    assert merged["ledger_records_truncated"] == 0


def test_total_projection_bound_counts_all_tracelab_input_surfaces() -> None:
    telemetry = {
        "tool_outcomes": {
            "ledger_records": [
                _fetch_record(f"https://example.test/tool/{index}")
                for index in range(40)
            ],
            "ledger_records_truncated": 0,
        },
        "critique_telemetry": {
            "annotations": [
                {
                    "applied": True,
                    "verdict": "unsupported",
                    "citation_urls": [
                        f"https://example.test/critique/{index}/{url_index}"
                        for url_index in range(5)
                    ],
                }
                for index in range(10)
            ]
        },
    }

    assert (
        _ledger_projection_upper_bound(
            source_count=600,
            citation_count=300,
            synthesis_telemetry=telemetry,
        )
        == 990
    )


def test_total_projection_overflow_marks_a_complete_audit_loudly_incomplete() -> None:
    records = [
        _fetch_record(f"https://example.test/tool/{index}")
        for index in range(101)
    ]
    telemetry = {
        "tool_outcomes": {
            "ledger_records": records,
            "ledger_records_truncated": 0,
        },
        "critique_telemetry": {
            "annotations": [
                {
                    "applied": True,
                    "verdict": "hallucinated",
                    "citation_urls": [f"https://example.test/critique/{index}"],
                }
                for index in range(100)
            ]
        },
    }

    upper_bound = _enforce_ledger_projection_budget(
        source_count=500,
        citation_count=300,
        synthesis_telemetry=telemetry,
    )

    assert upper_bound == 1_001
    assert telemetry["tool_outcomes"]["ledger_records"] == records
    assert telemetry["tool_outcomes"]["ledger_records_truncated"] == 1


def test_final_budget_guard_has_no_legacy_missing_envelope_fallback() -> None:
    telemetry: dict = {"tool_outcomes": {}}

    upper_bound = _enforce_ledger_projection_budget(
        source_count=1,
        citation_count=1,
        synthesis_telemetry=telemetry,
    )

    assert upper_bound == 4
    assert telemetry["tool_outcomes"] == {
        "ledger_records": [],
        "ledger_records_truncated": 2,
    }


def test_execution_metadata_serializes_1000_records_without_loss() -> None:
    outcomes = CodeAgentToolBundle(
        tools=[],
        call_log=[
            _fetch_record(f"https://example.test/accepted/{index}")
            for index in range(LEDGER_RECORD_MAX_ENTRIES)
        ],
    ).tool_outcomes()

    serialized = ExecutionMetadata(
        total_tokens=1,
        duration_seconds=1.0,
        synthesis_telemetry={"tool_outcomes": outcomes},
    ).to_dict()
    persisted = serialized["synthesis_telemetry"]["tool_outcomes"]

    assert len(persisted["ledger_records"]) == LEDGER_RECORD_MAX_ENTRIES
    assert persisted["ledger_records_truncated"] == 0


def test_execution_metadata_serializes_1001st_record_as_loud_loss() -> None:
    outcomes = CodeAgentToolBundle(
        tools=[],
        call_log=[
            _fetch_record(f"https://example.test/overflow/{index}")
            for index in range(LEDGER_RECORD_MAX_ENTRIES + 1)
        ],
    ).tool_outcomes()

    serialized = ExecutionMetadata(
        total_tokens=1,
        duration_seconds=1.0,
        synthesis_telemetry={"tool_outcomes": outcomes},
    ).to_dict()
    persisted = serialized["synthesis_telemetry"]["tool_outcomes"]

    assert len(persisted["ledger_records"]) == LEDGER_RECORD_MAX_ENTRIES
    assert persisted["ledger_records_truncated"] == 1


def _serialized_production_wrapper_attempts(
    monkeypatch: pytest.MonkeyPatch,
    *,
    relevant_attempts: int,
) -> tuple[dict, dict, str]:
    """Run the real tool closures and return their persisted telemetry.

    Only the transport functions behind the closures are replaced.  This keeps
    URL canonicalization, outcome logging, the independent ledger projection,
    ``SynthesisResult`` threading, and ``ExecutionMetadata`` serialization on
    the production path under test.
    """

    from tests.unit.test_codeagent_runner import _install_fake_smolagents

    _install_fake_smolagents(monkeypatch)

    import deepsearch.agent.tools.source_fetch as source_fetch_module
    import deepsearch.agent.tools.url_liveness as url_liveness_module
    import deepsearch.agent.tools.web_search as web_search_module

    monkeypatch.setattr(
        web_search_module,
        "web_search",
        lambda query, max_results=20: [],
    )
    monkeypatch.setattr(
        source_fetch_module,
        "source_fetch",
        lambda url, timeout_s=15.0: {
            "text": "",
            "status": "dead",
            "status_code": 404,
            "fetched_at": "2026-08-21T00:00:00Z",
            "final_url": url,
            "error": "http_404",
        },
    )
    monkeypatch.setattr(
        url_liveness_module,
        "url_liveness",
        lambda url, timeout_s=5.0: {
            "alive": False,
            "status_code": 404,
            "final_url": url,
            "error": "http_404",
        },
    )

    bundle = runner.build_codeagent_tools()
    web_search, source_fetch, url_liveness, *_ = bundle.tools

    # Consume the complete generic 250-row sample before any Ledger-relevant
    # attempt.  The extra generic call proves the two caps are independent.
    for index in range(251):
        web_search(f"generic-diagnostic-{index}", 1)

    long_url = (
        "https://ordinary.example.test/source-"
        + ("a" * 240)
        + "?view=complete&revision=1"
    )
    for index in range(relevant_attempts):
        url = long_url if index == 0 else f"https://ordinary.example.test/{index}"
        if index % 2 == 0:
            source_fetch(url)
        else:
            url_liveness(url)

    outcomes = bundle.tool_outcomes()
    synthesis_result = runner.SynthesisResult(
        output_markdown="# Ledger boundary",
        step_count=1,
        self_terminated=True,
        hit_max_steps=False,
        runtime_seconds=1.0,
        token_usage={"input": 1, "output": 1, "total": 2},
        tool_call_summary=bundle.tool_call_summary(),
        authority_tier_distribution=bundle.authority_tier_summary(),
        pool_size=bundle.pool_size,
        step_trace=[],
        retrieval_state=bundle.retrieval_state_summary(),
        tool_outcomes=outcomes,
    )
    synthesis_telemetry = _build_synthesis_telemetry(synthesis_result)
    serialized = ExecutionMetadata(
        total_tokens=2,
        duration_seconds=1.0,
        synthesis_telemetry=synthesis_telemetry,
    ).to_dict()
    persisted = serialized["synthesis_telemetry"]["tool_outcomes"]
    return outcomes, persisted, long_url


def _assert_strict_wrapper_records(records: list[dict]) -> None:
    """Pin TraceLab's exact five-/six-key per-tool record variants."""

    source_fetch_keys = {
        "tool",
        "url",
        "status",
        "status_code",
        "error_category",
    }
    url_liveness_keys = {*source_fetch_keys, "alive"}
    for record in records:
        expected = (
            url_liveness_keys
            if record["tool"] == "url_liveness"
            else source_fetch_keys
        )
        assert set(record) == expected


def test_production_wrappers_serialize_exactly_1000_lossless_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The accepted boundary survives every production serialization seam."""

    outcomes, persisted, long_url = _serialized_production_wrapper_attempts(
        monkeypatch,
        relevant_attempts=LEDGER_RECORD_MAX_ENTRIES,
    )

    assert len(long_url) > 200
    assert len(outcomes["records"]) == 250
    assert outcomes["records_truncated"] == 1_001
    assert all(record["tool"] == "web_search" for record in outcomes["records"])
    assert persisted["ledger_records"] == outcomes["ledger_records"]
    assert len(persisted["ledger_records"]) == LEDGER_RECORD_MAX_ENTRIES
    assert persisted["ledger_records_truncated"] == 0
    assert persisted["ledger_records"][0]["url"] == long_url
    _assert_strict_wrapper_records(persisted["ledger_records"])
    assert (
        _ledger_projection_upper_bound(
            source_count=0,
            citation_count=0,
            synthesis_telemetry={"tool_outcomes": persisted},
        )
        == LEDGER_RECORD_MAX_ENTRIES
    )


def test_production_wrappers_serialize_1001st_attempt_as_loud_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attempt 1,001 is never hidden behind the generic diagnostic cap."""

    outcomes, persisted, long_url = _serialized_production_wrapper_attempts(
        monkeypatch,
        relevant_attempts=LEDGER_RECORD_MAX_ENTRIES + 1,
    )

    assert len(outcomes["records"]) == 250
    assert outcomes["records_truncated"] == 1_002
    assert len(persisted["ledger_records"]) == LEDGER_RECORD_MAX_ENTRIES
    assert persisted["ledger_records_truncated"] == 1
    assert persisted["ledger_records"][0]["url"] == long_url
    _assert_strict_wrapper_records(persisted["ledger_records"])
    # TraceLab's frozen envelope requires zero truncation and <=1,000 total
    # projection inputs.  Counting the loud loss makes this handoff 1,001 and
    # therefore deterministically rejectable rather than falsely complete.
    assert (
        _ledger_projection_upper_bound(
            source_count=0,
            citation_count=0,
            synthesis_telemetry={"tool_outcomes": persisted},
        )
        == LEDGER_RECORD_MAX_ENTRIES + 1
    )
