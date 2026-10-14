"""All-attempt spend remains visible when parsing, recovery or judgment fails."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from deepsearch.eval.cost import estimate_all_attempts, estimate_request
from deepsearch.llm.usage import (
    normalize_usage,
    observe_model_requests,
    summarize_usage,
    usage_event,
)


def event(model: str = "gpt-5.6-luna", **counts: int) -> dict:
    return usage_event(
        {
            "model": model,
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1100,
                **counts,
            },
        },
        config={"requested_model": model},
        stage="synthesis",
        elapsed=1,
        started_at="2026-09-19T12:00:00+00:00",
    )


def test_cache_and_reasoning_are_subsets_never_extra_tokens() -> None:
    usage = normalize_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 800},
            "completion_tokens_details": {"reasoning_tokens": 70},
        }
    )
    assert usage["total"] == 1100
    assert usage["cache_read_input"] == 800 and usage["reasoning_output"] == 70
    assert usage["complete"]
    assert usage["cache_write_input"] is None


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"prompt_tokens": True},
        {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 10},
        {"prompt_tokens": 10, "completion_tokens": 2, "prompt_cache_hit_tokens": 11},
    ],
)
def test_absent_or_contradictory_usage_is_not_a_complete_free_request(raw) -> None:
    assert not normalize_usage(raw)["complete"]


def test_raw_sdk_observation_survives_response_parsing_failure_and_failed_request() -> None:
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise TimeoutError("SECRET REQUEST CONTENT")
        return {"model": "gpt-5.6-luna", "usage": {"prompt_tokens": 4, "completion_tokens": 6}}

    model = SimpleNamespace(
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    )
    observe_model_requests(model, {"requested_model": "gpt-5.6-luna", "provider": "openai"})
    response = model.client.chat.completions.create(messages=["SECRET"])
    assert response["usage"]["prompt_tokens"] == 4
    with pytest.raises(TimeoutError):
        model.client.chat.completions.create(messages=["SECRET"])
    total = summarize_usage(model.provider_attempts)
    assert total["observed"]["total"] == 10 and total["requests"] == 2
    assert not total["usage_complete"]
    assert "SECRET" not in json.dumps(model.provider_attempts)
    assert model.provider_attempts[1]["status"] == "failed"


def test_failed_observation_cannot_replace_the_provider_response() -> None:
    class Bad:
        @property
        def usage(self):
            raise RuntimeError("invalid metadata")

    record = usage_event(Bad(), config={}, stage="critique", started_at="now", elapsed=1)
    assert record["observation_error"] and not record["usage"]["complete"]


def test_long_context_cache_write_pricing_and_reasoning_do_not_double_count() -> None:
    e = event(
        prompt_tokens=300000,
        completion_tokens=10000,
        total_tokens=310000,
        prompt_tokens_details={"cached_tokens": 100000, "cache_creation_tokens": 50000},
        completion_tokens_details={"reasoning_tokens": 9000},
    )
    price = estimate_request(e)
    expected = (100000 * 0.02 + 150000 * 0.2 + 50000 * 0.2 * 1.25) * 2 / 1e6
    expected += 10000 * 1.2 * 1.5 / 1e6
    assert price["usd_min"] == pytest.approx(expected)
    assert price["usd_max"] == price["usd_min"]


def test_deepseek_weekend_offpeak_and_unknown_holiday_peak_are_explicit() -> None:
    e = event("deepseek-flash", prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=1000)
    assert estimate_request(e)["usd_max"] == pytest.approx(0.00021)
    e["started_at"] = "2026-09-21T02:00:00+00:00"
    unknown = estimate_request(e)
    assert unknown["usd_min"] == pytest.approx(0.00021)
    assert unknown["usd_max"] == pytest.approx(0.00042)
    assert estimate_request(e, chinese_public_holiday=False)["usd_min"] == unknown["usd_max"]


def test_every_failed_rejected_attempt_counts_once_and_missing_usage_prevents_total_bill() -> None:
    first = event()
    second = event()
    second.update(stage="critique", selection="rejected")
    unknown = event()
    unknown.update(status="failed", usage=normalize_usage(None))
    summary = summarize_usage([first, second, first, unknown])
    assert summary["observed"]["total"] == 2200 and summary["requests"] == 3
    costs = estimate_all_attempts([first, second, first, unknown])
    assert costs["model_usd_min"] > 0 and costs["model_usd_max"] is None
    assert costs["unpriced_requests"] == 1 and costs["retrieval_usd"] is None


def test_compiler_rejected_json_and_success_both_record_usage() -> None:
    from pydantic import BaseModel

    from deepsearch.llm.structured_extractor import JSONInPromptStructuredExtractor

    class Schema(BaseModel):
        value: int

    class Model:
        calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            return SimpleNamespace(
                content="bad" if self.calls == 1 else '{"value": 7}',
                response_metadata={
                    "model_name": "deepseek-flash",
                    "finish_reason": "stop",
                    "token_usage": {"prompt_tokens": 10, "completion_tokens": 2},
                },
            )

    extractor = JSONInPromptStructuredExtractor(
        Model(), effective_config={"requested_model": "deepseek-flash"}
    )
    assert (
        asyncio.run(
            extractor.extract(system_instruction="JSON", prompt="value", schema=Schema)
        ).value
        == 7
    )
    assert len(extractor.diagnostics) == 2
    assert summarize_usage(extractor.diagnostics)["observed"]["total"] == 24


@pytest.mark.parametrize("provider", ["deepseek", "openai"])
def test_real_sdk_stage_observations_reconcile_without_double_counting(
    monkeypatch, provider
) -> None:
    from dataclasses import replace

    import httpx
    from openai import OpenAI
    from smolagents import OpenAIServerModel

    from deepsearch.agent.codeagent_finalize_node import _build_attempt_accounting
    from deepsearch.agent.codeagent_runner import build_default_model, run_codeagent_synthesis
    from deepsearch.agent.critique import CRITIQUE_OUTPUT_DELIMITER, run_critique
    from deepsearch.config import DeepSearchSettings

    calls = []
    model_id = "deepseek-flash" if provider == "deepseek" else "gpt-5.6-luna"
    # The runner requires a retrieval credential before constructing its tools.
    # This test replaces every network client, so give that boundary an explicit
    # inert value instead of borrowing a developer's local .env file.
    monkeypatch.setenv("LINKUP_API_KEY", "offline")

    def respond(request):
        calls.append(request)
        content = (
            '```python\nfinal_answer("A report [cite:https://example.org].")\n```'
            if len(calls) == 1
            else CRITIQUE_OUTPUT_DELIMITER + '\n{"annotations": [], "claim_count": 0}'
        )
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 1,
                "model": model_id + "-returned",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "prompt_tokens_details": {"cached_tokens": 30},
                    "completion_tokens_details": {"reasoning_tokens": 20},
                },
            },
        )

    client = OpenAI(
        api_key="offline",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        max_retries=0,
    )
    monkeypatch.setattr(OpenAIServerModel, "create_client", lambda self: client)
    settings = replace(
        DeepSearchSettings.load(),
        llm_backend=provider,
        deepseek_model="deepseek-flash",
        openai_model="gpt-5.6-luna",
        deepseek_api_key="offline",
        openai_api_key="offline",
        llm_reasoning_effort=None,
    )
    model = build_default_model(settings=settings)
    synthesis = run_codeagent_synthesis(
        system_instructions="Write a report.", user_directive="Finish.", model=model, max_steps=1
    )
    assert synthesis.self_terminated and len(synthesis.provider_attempts) == 1
    critic = run_critique(synthesis.output_markdown, [], {}, model=model)
    assert len(critic.provider_attempts) == 1 and critic.provider_attempts[0]["stage"] == "critique"
    assert critic.provider_attempts[0]["returned_model"] == model_id + "-returned"
    compiler = event(model_id)
    compiler["stage"] = "compiler"
    accounting = _build_attempt_accounting(
        [synthesis], [critic], token_usage_complete=True, compiler_events=[compiler]
    )
    assert accounting["model_accounting"]["observed"]["total"] == 1400
    assert accounting["model_accounting"]["reconciliation_delta"] == 0
    assert accounting["token_usage"]["total"] == 300


def test_volatile_usage_does_not_change_semantic_contract_identity() -> None:
    from deepsearch.mission.contract import _contract_id_for_payload

    baseline = {
        "title": "Fixed mission",
        "entity_extraction_decisions": [
            {"entity": "Example", "decision": "keep", "reason": "named entity"},
        ],
    }
    observed = {
        **baseline,
        "entity_extraction_decisions": [
            *baseline["entity_extraction_decisions"],
            {"entity": "", "decision": "provider_usage", "reason": json.dumps(event())},
        ],
    }
    assert _contract_id_for_payload(baseline) == _contract_id_for_payload(observed)
    assert _contract_id_for_payload(
        {**observed, "title": "Changed mission"}
    ) != _contract_id_for_payload(baseline)


def test_prior_lease_and_rejected_pass_observations_survive_cost_collection_once() -> None:
    from deepsearch.eval.cost import collect_attempt_events

    compiler, failed, selected = event(), event(), event()
    compiler["stage"] = "compiler"
    failed["usage"] = normalize_usage(None)
    metadata = {
        "prior_execution_attempts": [
            {
                "synthesis_telemetry": {
                    "compiler_provider_attempts": [compiler],
                    "provider_attempts": [failed],
                }
            }
        ],
        "synthesis_telemetry": {
            "provider_attempts": [selected],
            "recovery": {
                "attempt_accounting": {"provider_attempts": [selected]},
            },
        },
    }
    events = collect_attempt_events(metadata)
    assert len(events) == 3
    assert estimate_all_attempts(events)["unpriced_requests"] == 1


def test_failed_before_synthesis_still_persists_compiler_observation() -> None:
    from deepsearch.worker.writer import build_execution_metadata

    compiler = event()
    metadata = build_execution_metadata(
        {
            "mission_contract": {
                "entity_extraction_decisions": [
                    {"decision": "provider_usage", "reason": json.dumps(compiler)},
                ]
            }
        },
        duration_seconds=1,
    )
    assert metadata.synthesis_telemetry["compiler_provider_attempts"] == [compiler]


def test_absent_attempt_history_cannot_establish_zero_cost() -> None:
    assert estimate_all_attempts([])["model_usd_max"] is None
    assert summarize_usage([])["usage_complete"] is False
