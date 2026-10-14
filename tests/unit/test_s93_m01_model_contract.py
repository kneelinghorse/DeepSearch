"""Offline wire contracts: provider selection must reach every paid stage."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from deepsearch.agent import codeagent_runner as runner
from deepsearch.agent.critique import run_critique
from deepsearch.config import DeepSearchSettings
from deepsearch.llm.runtime import ModelConfigurationError, resolve_effective_model
from deepsearch.mission.entity_extractor_llm import (
    run_declared_entity_disambiguation_sync as real_disambiguate,
)
from deepsearch.mission.entity_extractor_llm import (
    run_llm_entity_extraction_sync as real_extract,
)
from deepsearch.worker.execution import _configure_attempt_environment


class Extracted(BaseModel):
    answer: str


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """No local dotenv credentials or settings may affect these wire contracts."""
    monkeypatch.setattr("deepsearch.config.load_dotenv", lambda: None)
    for name in (
        "LLM_BACKEND", "DEEPSEEK_MODEL", "DEEPSEEK_BASE_URL", "DEEPSEEK_MAX_TOKENS",
        "OPENAI_MODEL", "OPENAI_BASE_URL", "OPENAI_MAX_TOKENS",
        "LLM_REASONING_EFFORT", "LLM_REQUEST_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-offline-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-offline-key")
    monkeypatch.setenv("LINKUP_API_KEY", "linkup-offline-key")


def _response(request: httpx.Request, content: str, finish: str = "stop") -> httpx.Response:
    return httpx.Response(200, request=request, json={
        "id": "offline", "object": "chat.completion", "created": 0,
        "model": json.loads(request.content)["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    })


@pytest.mark.parametrize("provider,model,host,token_field,context", [
    ("deepseek", "deepseek-flash", "api.deepseek.com", "max_tokens", 1_000_000),
    ("openai", "gpt-5.6-luna", "api.openai.com", "max_completion_tokens", 1_050_000),
])
def test_env_reaches_real_compiler_synthesis_critique_and_final_answer_requests(
    monkeypatch, tmp_path, provider, model, host, token_field, context,
) -> None:
    """Real SDK serialization catches LangChain's unconditional token rename."""
    monkeypatch.setenv("LLM_BACKEND", provider)
    monkeypatch.setenv(f"{provider.upper()}_MAX_TOKENS", "9000")
    for name in ("DEEPSEARCH_CONTRACT_DIR", "DEEPSEARCH_REPORT_DIR",
                 "DEEPSEARCH_TELEMETRY_DIR", "CHECKPOINT_DB_PATH"):
        monkeypatch.setenv(name, "offline")
    _configure_attempt_environment(tmp_path)
    settings = DeepSearchSettings.load()
    effective = resolve_effective_model(settings)
    assert effective.context_window == context
    assert effective.model == model
    requests = []

    def respond(request):
        requests.append(request)
        return _response(request, '{"answer":"ok"}')

    extractor = settings.build_structured_extractor()
    chat = extractor.chat_model
    client = AsyncOpenAI(
        api_key=effective.api_key, base_url=effective.base_url,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        max_retries=0,
    )
    chat.async_client = client.chat.completions
    assert asyncio.run(extractor.extract(
        system_instruction="Extract JSON", prompt="answer", schema=Extracted,
    )) == Extracted(answer="ok")

    bound = runner.build_default_model(settings=settings)
    bound.client = OpenAI(
        api_key=effective.api_key, base_url=effective.base_url,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)), max_retries=0,
    )
    bound.generate(messages=[{"role": "user", "content": "draft"}], stop_sequences=["STOP"])
    # The same factory's actual critic call must not inject a DeepSeek token parameter.
    run_critique(
        draft_markdown="draft", citations=[], sources_collected=[], mission_context={}, model=bound,
    )
    from smolagents import CodeAgent
    agent = CodeAgent(tools=[], model=bound, max_steps=1, verbosity_level=-1)
    agent.provide_final_answer("Finish from retained memory")

    assert len(requests) == 4
    for request in requests:
        payload = json.loads(request.content)
        assert request.url.host == host
        assert request.headers["authorization"] == f"Bearer {provider}-offline-key"
        assert payload["model"] == model
        assert payload[token_field] == 9000
        assert ("max_tokens" if provider == "openai" else "max_completion_tokens") not in payload
        assert "temperature" not in payload
        assert payload["reasoning_effort"] == ("medium" if provider == "openai" else "high")
        if provider == "openai":
            assert "stop" not in payload
            assert "thinking" not in payload
        else:
            assert payload["thinking"] == {"type": "enabled"}
    assert bound.effective_config == effective.public_metadata()
    assert "offline-key" not in json.dumps(bound.effective_config)


@pytest.mark.parametrize("updates", [
    {"llm_backend": "typo"},
    {"deepseek_model": "deepseek-reasoner"},
    {"deepseek_model": "gpt-5.6-luna"},
    {"deepseek_base_url": "https://api.openai.com/v1"},
    {"llm_backend": "openai", "openai_base_url": "https://api.deepseek.com"},
    {"llm_backend": "openai", "openai_base_url": "https://proxy.example/v1"},
    {"llm_backend": "openai", "openai_model": "gpt-5.6"},
    {"llm_backend": "openai", "openai_api_key": None},
    {"deepseek_api_key": None},
    {"llm_backend": "openai", "openai_max_output_tokens": 128001},
    {"deepseek_max_output_tokens": 393217},
    {"llm_request_timeout_seconds": float("nan")},
    {"llm_reasoning_effort": "medium"},
])
def test_invalid_selection_never_routes_elsewhere(updates) -> None:
    with pytest.raises(ModelConfigurationError):
        resolve_effective_model(replace(DeepSearchSettings.load(), **updates))


@pytest.mark.parametrize("name,value", [
    ("DEEPSEEK_MAX_TOKENS", "bad"), ("LLM_REQUEST_TIMEOUT_SECONDS", "bad"),
    ("LLM_BACKEND", "deepsek"),
])
def test_invalid_environment_survives_load_for_preflight(monkeypatch, name, value) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ModelConfigurationError):
        resolve_effective_model(DeepSearchSettings.load())


@pytest.mark.parametrize("status,attempts", [(400, 1), (401, 1), (403, 1), (404, 1),
                                            (422, 1), (429, 3), (500, 3)])
def test_compiler_retries_only_transient_requests(monkeypatch, status, attempts, caplog) -> None:
    monkeypatch.setenv("LLM_BACKEND", "openai")
    extractor = DeepSearchSettings.load().build_structured_extractor()
    calls = []

    def fail(request):
        calls.append(request)
        return httpx.Response(status, request=request, json={"error": {
            "message": "SECRET SOURCE BODY", "type": "invalid_request_error",
            "code": "unsupported_parameter",
        }})

    client = AsyncOpenAI(api_key="unpaid", http_client=httpx.AsyncClient(
        transport=httpx.MockTransport(fail)), max_retries=0)
    extractor.chat_model.async_client = client.chat.completions
    assert asyncio.run(extractor.extract(
        system_instruction="JSON", prompt="answer", schema=Extracted,
    )) is None
    assert len(calls) == attempts
    failures = [d for d in extractor.diagnostics if d.get("kind") != "provider_usage"]
    assert len(failures) == attempts
    assert failures[-1]["http_status"] == status
    assert failures[-1]["requested_model"] == "gpt-5.6-luna"
    assert failures[-1]["code"] == "unsupported_parameter"
    assert "SECRET SOURCE BODY" not in json.dumps(extractor.diagnostics) + caplog.text


def test_context_pressure_uses_bound_model_and_unknown_is_unassessed() -> None:
    from deepsearch.agent.codeagent_finalize_node import _build_context_pressure

    # Real result type: model identity is measured at the bind, not guessed at finalize.
    result = runner.SynthesisResult(
        output_markdown="draft", step_count=1, self_terminated=True,
        hit_max_steps=False, runtime_seconds=1, token_usage={"input": 800000},
        tool_call_summary={}, authority_tier_distribution={}, pool_size=0,
        step_trace=[{"step_number": 1, "token_usage": {"input_tokens": 800000}}],
        effective_config={"context_window": 1_050_000},
    )
    assert _build_context_pressure(result)["pressure_pct"] == round(800000 / 1050000, 4)
    result.effective_config = {}
    assert _build_context_pressure(result)["pressure_pct"] is None


@pytest.mark.parametrize("declared", [[], ["OpenAI"]])
def test_compiler_fallback_persists_content_free_failure(monkeypatch, declared) -> None:
    from deepsearch.mission import entity_extractor_llm
    from deepsearch.mission.contract import MissionContract, compile_contract_from_state

    monkeypatch.setenv("LLM_BACKEND", "openai")
    extractor = DeepSearchSettings.load().build_structured_extractor()
    calls = []

    def reject(request):
        calls.append(request)
        return httpx.Response(400, request=request, json={"error": {
            "message": "DO NOT PERSIST THIS SECRET", "code": "unsupported_parameter",
        }})

    client = AsyncOpenAI(api_key="offline", http_client=httpx.AsyncClient(
        transport=httpx.MockTransport(reject)), max_retries=0)
    extractor.chat_model.async_client = client.chat.completions
    monkeypatch.setattr(DeepSearchSettings, "build_structured_extractor", lambda self: extractor)
    monkeypatch.setattr(entity_extractor_llm, "run_llm_entity_extraction_sync", real_extract)
    monkeypatch.setattr(
        entity_extractor_llm, "run_declared_entity_disambiguation_sync", real_disambiguate,
    )
    contract = compile_contract_from_state({
        "mission_id": "S93-OFFLINE",
        "mission_context": {"title": "API study", "objective": "Compare public API contracts"},
        "required_entities": declared,
    }, origin="s93-offline")
    persisted = MissionContract.model_validate_json(contract.model_dump_json())
    rows = [row for row in persisted.entity_extraction_decisions
            if row["decision"] == "provider_failure"]
    assert len(calls) == len(rows) == 1
    diagnostic = json.loads(rows[0]["reason"])
    assert diagnostic["requested_model"] == "gpt-5.6-luna"
    assert diagnostic["stage"] == "compiler"
    assert diagnostic["category"] == "configuration"
    assert diagnostic["http_status"] == 400
    assert "DO NOT PERSIST" not in contract.model_dump_json()


@pytest.mark.parametrize("finish", ["length", "content_filter", "tool_calls"])
def test_structured_extraction_rejects_nonterminal_json(finish) -> None:
    extractor = DeepSearchSettings.load().build_structured_extractor()
    calls = []

    def respond(request):
        calls.append(request)
        return _response(request, '{"answer":"looks valid but is incomplete"}', finish)

    client = AsyncOpenAI(api_key="offline", http_client=httpx.AsyncClient(
        transport=httpx.MockTransport(respond)), max_retries=0)
    extractor.chat_model.async_client = client.chat.completions
    assert asyncio.run(extractor.extract(
        system_instruction="JSON", prompt="answer", schema=Extracted,
    )) is None
    assert len(calls) == 1
    assert extractor.diagnostics[0]["finish_reason"] == finish


def test_cancelled_compiler_bridge_never_dispatches_or_falls_back(monkeypatch) -> None:
    import threading

    from deepsearch.agent.execution_control import (
        ExecutionCancelledError,
        bind_execution_cancel_event,
    )
    from deepsearch.mission.entity_extractor_llm import LLMEntityExtractor

    extractor = DeepSearchSettings.load().build_structured_extractor()
    event = threading.Event()
    event.set()
    with bind_execution_cancel_event(event), pytest.raises(ExecutionCancelledError):
        real_extract(LLMEntityExtractor(structured_extractor=extractor),
                     objectives=["Compare APIs"], deliverables=[], required_entities=[])
