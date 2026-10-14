"""Authored boundaries are deterministic admission policy, not model quality tests."""

from __future__ import annotations

import copy
import json
import socket
from pathlib import Path
from typing import Any

import httpx
import pytest

from deepsearch.agent.codeagent_runner import build_codeagent_tools
from deepsearch.agent.contract_to_prompt import build_codeagent_prompts
from deepsearch.agent.critique import _build_critique_messages, _GroundingVisibility
from deepsearch.agent.tools import scholar_search as scholar_module
from deepsearch.agent.tools import source_fetch as fetch_module
from deepsearch.agent.tools import url_liveness as probe_module
from deepsearch.agent.tools import web_search as search_module
from deepsearch.mission.contract import compile_contract_from_state
from deepsearch.mission.scope import (
    AuthoredScope,
    ScopeAdmission,
    compile_authored_scope,
    page_identity,
)

FIXTURE = Path(__file__).parents[1] / "fixtures/authored_scope_v1/postgresql-authored.json"
URL1 = "https://www.postgresql.org/docs/current/functions-datetime.html"
URL2 = "https://www.postgresql.org/docs/current/explicit-locking.html"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Scope fixtures must never connect or resolve DNS")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def authored_context() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def exact_scope() -> AuthoredScope:
    return compile_authored_scope(authored_context())


@pytest.mark.parametrize("enrichment", ["none", "configured"])
def test_contract_preserves_exact_policy_in_both_compiler_paths(enrichment: str) -> None:
    context = authored_context()
    state = {"mission_id": context["mission_id"], "mission_context": context}
    compiled = compile_contract_from_state(state, origin="fixture", enrichment_mode=enrichment)
    scope = compiled.authored_scope
    assert compiled.contract_schema_version == "1.2"
    assert compiled.compiler_revision == 3
    assert scope.allowed_urls == [URL1, URL2]
    assert (scope.min_words, scope.max_words, scope.max_sources) == (300, 500, 2)
    assert scope.restriction == "exact_pages"
    assert any(c.kind == "authored_scope" for c in compiled.acceptance_checks)
    state["mission_contract"] = compiled.model_dump(mode="json")
    system, _ = build_codeagent_prompts(state)
    assert URL1 in system and URL2 in system
    assert "2,500-6,000" not in system
    assert scope.prompt() in system
    changed = copy.deepcopy(state)
    changed["mission_context"]["references"][0]["url"] = URL1.replace("current", "18")
    other = compile_contract_from_state(changed, origin="fixture", enrichment_mode=enrichment)
    assert compiled.contract_id != other.contract_id
    changed["mission_context"]["success_criteria"][-1] = "Deliver a 300–501 word note."
    assert other.contract_id != compile_contract_from_state(
        changed, origin="fixture", enrichment_mode=enrichment,
    ).contract_id


def test_references_alone_are_seeds_and_leave_generic_report_target() -> None:
    context = {"objective": "Compare approaches", "references": [{"url": URL1}]}
    scope = compile_authored_scope(context)
    assert scope.reference_urls == [URL1]
    assert scope.restriction == "unrestricted"
    assert scope.allows("https://elsewhere.example/research")
    prompt, _ = build_codeagent_prompts({"mission_context": context})
    assert "2,500-6,000" in prompt and URL1 in prompt


@pytest.mark.parametrize("context", [
    {"authored_scope": {"min_words": 501, "max_words": 500}},
    {"authored_scope": {"max_sources": True}},
    {"authored_scope": {"restriction": "exact_pages", "allowed_urls": []}},
    {"authored_scope": {"restriction": "domains", "allowed_domains": ["https://x.com"]}},
    {"constraints": ["Use only requested pages."]},
    {"constraints": ["Use only example.com. Use only requested pages."],
     "references": [{"url": URL1}]},
    {"constraints": ["Use only requested pages."], "references": [{"url": URL1}],
     "excluded_domains": ["postgresql.org"]},
    {"success_criteria": ["Write a 500-300 word note."]},
])
def test_invalid_or_conflicting_scope_is_rejected_before_provider(context: dict) -> None:
    with pytest.raises(ValueError):
        compile_contract_from_state({"mission_context": context}, origin="fixture",
                                    enrichment_mode="none")


def test_page_identity_never_equates_www_apex_versions_or_queries() -> None:
    scope = exact_scope()
    assert scope.allows(URL1 + "#FUNCTIONS-DATETIME-CURRENT")
    for changed in [URL1.replace("www.", ""), URL1.replace("current", "18"), URL1 + "?v=2"]:
        assert not scope.allows(changed)
        assert page_identity(changed) != page_identity(URL1)


def test_exact_pages_refuse_both_searches_and_unlisted_fetch_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Prohibited action reached provider")
    monkeypatch.setattr(search_module, "web_search", unexpected)
    monkeypatch.setattr(scholar_module, "scholar_search", unexpected)
    monkeypatch.setattr(fetch_module, "source_fetch", unexpected)
    admission = ScopeAdmission(exact_scope())
    bundle = build_codeagent_tools(scope_admission=admission)
    assert bundle.tools[0]("postgres timestamps")[0]["error"] == "scope_exact_pages_no_search"
    assert bundle.tools[5]("postgres timestamps")[0]["error"] == "scope_exact_pages_no_search"
    assert bundle.tools[1](URL1.replace("current", "18"))["network_requested"] is False
    assert admission.consulted == {}
    assert len(admission.rejected) == 3


def test_domain_policy_filters_snippets_and_counts_them_even_without_fetch_or_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries: list[str] = []
    def search(query: str, *args: Any, **kwargs: Any) -> list[dict]:
        queries.append(query)
        return [{"url": u, "snippet": "Evidence", "title": "Source"} for u in
                [URL1, "https://postgrespro.com/docs/current/functions-datetime.html", URL2,
                 "https://www.postgresql.org/docs/18/ddl.html"]]
    monkeypatch.setattr(search_module, "web_search", search)
    admission = ScopeAdmission(AuthoredScope(restriction="domains",
                              allowed_domains=["postgresql.org"], max_sources=2))
    bundle = build_codeagent_tools(scope_admission=admission)
    results = bundle.tools[0]("timestamps")
    assert [r["url"] for r in results] == [URL1, URL2]
    assert "site:postgresql.org" in queries[0]
    assert len(admission.consulted) == 2
    assert bundle.sources_fetched == []
    assert len(bundle.sources_collected) == 2
    assert bundle.tools[0]("another query")[0]["error"] == "scope_source_limit"
    assert bundle.tools[5]("query")[0]["error"] == "scope_scholar_domain_filter_unavailable"
    assert bundle.tools[0]("site:postgrespro.com query")[0]["error"]
    assert len(queries) == 1


def test_scholar_snippets_obey_source_budget_without_domain_restriction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scholar_module, "scholar_search", lambda *a, **k: [
        {"url": u, "snippet": "Scholar evidence"} for u in [URL1, URL2]])
    admission = ScopeAdmission(AuthoredScope(max_sources=1))
    bundle = build_codeagent_tools(scope_admission=admission)
    assert [r["url"] for r in bundle.tools[5]("query")] == [URL1]
    assert admission.snapshot()["consulted_count"] == 1


def test_partial_fetch_and_stored_rereads_preserve_bound_and_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    def fetch(url: str, *args: Any, **kwargs: Any) -> dict:
        calls.append(url)
        return {"text": "Primary evidence " * 100 if url == URL1 else "",
                "status": "ok" if url == URL1 else "timeout", "status_code": 200,
                "final_url": url, "fetched_at": "2026-10-01T00:00:00Z"}
    monkeypatch.setattr(fetch_module, "source_fetch", fetch)
    admission = ScopeAdmission(exact_scope())
    bundle = build_codeagent_tools(scope_admission=admission)
    assert bundle.tools[1](URL1)["status"] == "ok"
    assert bundle.tools[1](URL1, start=20, length=50)["cache_hit"] is True
    assert bundle.tools[1](URL2)["status"] == "timeout"
    assert calls == [URL1, URL2]
    assert list(admission.consulted) == [URL1]
    assert admission.snapshot()["unavailable"] == [{"url": URL2, "reason": "timeout"}]
    assert bundle.sources_collected[0]["body"] == "Primary evidence " * 100


@pytest.mark.parametrize("tool", ["fetch", "probe"])
def test_redirect_scope_is_checked_before_dns_and_followup_request(
    monkeypatch: pytest.MonkeyPatch, tool: str,
) -> None:
    seen: list[str] = []
    dns: list[str] = []
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://postgrespro.com/other"})
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(
        **kw, transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(fetch_module, "_validate_public_target", lambda u: dns.append(u))
    monkeypatch.setattr(probe_module, "_validate_public_target", lambda u: dns.append(u))
    admission = ScopeAdmission(exact_scope())
    result = (fetch_module.source_fetch if tool == "fetch" else probe_module.url_liveness)(
        URL1, scope_admission=admission)
    assert result["error"] == "scope_source_not_allowed"
    assert seen == [URL1] and dns == [URL1]
    assert admission.consulted == {}


@pytest.mark.parametrize("prefiltered", [False, True])
def test_22_grounding_rows_cannot_enter_two_page_critique_view(prefiltered: bool) -> None:
    corpus = [{"url": u, "body": "Collected evidence", "snippet": "Snippet",
               "body_source": "source_fetch", "content_quality": "readable_body"}
              for u in [URL1, URL2, *[f"https://postgrespro.com/page{i}" for i in range(20)]]]
    admission = ScopeAdmission(exact_scope())
    if prefiltered:
        corpus = admission.filter_sources(corpus, "pre_critique")
    telemetry: dict = {}
    visibility = _GroundingVisibility()
    messages = _build_critique_messages(
        "No third source was consulted. [1][2]", corpus, authored_context(),
        grounding_telemetry=telemetry, grounding_visibility=visibility,
        scope_admission=admission,
    )
    assert "postgrespro.com" not in messages[1]["content"]
    assert exact_scope().prompt() in messages[1]["content"]
    assert telemetry["scope_sources_rejected"] == 20
    assert telemetry["grounding_context_complete"] == 0
    assert visibility.corpus_complete is False
    assert len(admission.rejected) == 20
    assert len(admission.consulted) == 2


def test_typed_scope_survives_execute_and_preview_boundaries() -> None:
    from deepsearch.api.models.requests import (
        ExecuteMissionRequest,
        MissionContractPreviewRequest,
        build_mission_context_from_execute_request,
    )
    from deepsearch.api.routes.missions import _build_preview_state
    policy = exact_scope().model_dump(mode="json")
    execute = ExecuteMissionRequest.model_validate({
        "mission_id": "typed", "title": "Typed", "objectives": ["Explain timing"],
        "authoredScope": policy,
    })
    preview = MissionContractPreviewRequest.model_validate({
        "mission_id": "typed", "title": "Typed", "objective": "Explain timing",
        "authoredScope": policy,
    })
    execute_context = build_mission_context_from_execute_request(execute)
    assert compile_authored_scope(execute_context) == exact_scope()
    preview_context = _build_preview_state(preview)["mission_context"]
    assert compile_authored_scope(preview_context) == exact_scope()


def test_retained_memory_finalization_keeps_the_exact_scope() -> None:
    from types import SimpleNamespace

    from deepsearch.agent.codeagent_runner import _configure_max_steps_final_answer_prompt
    system, _ = build_codeagent_prompts({"mission_context": authored_context()})
    agent = SimpleNamespace(prompt_templates={})
    _configure_max_steps_final_answer_prompt(agent, system)
    text = json.dumps(agent.prompt_templates)
    assert URL1 in text and URL2 in text and "2,500-6,000" not in text


def test_cross_service_package_matches_frozen_contract_actions_and_outputs() -> None:
    """TraceLab can detect compiler, prompt or output drift without a provider."""
    from scripts.check_authored_scope_parity import verify

    result = verify(FIXTURE.parent)
    assert result["actions_checked"] == 8
    assert result["outputs_checked"] == 4
    assert result["schema"] == "1.2"


def test_long_allowed_page_identity_survives_fetch_and_stored_reread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Payload bounding must never turn an allowed exact URL into another identity."""
    url = URL1 + "?view=" + "x" * 2500
    monkeypatch.setattr(fetch_module, "source_fetch", lambda *a, **kw: {
        "status": "ok", "text": "Evidence " * 100, "final_url": url,
        "status_code": 200,
    })
    admission = ScopeAdmission(AuthoredScope(
        restriction="exact_pages", allowed_urls=[url], max_sources=1,
    ))
    bundle = build_codeagent_tools(scope_admission=admission)
    assert bundle.tools[1](url)["final_url"] == url
    assert bundle.tools[1](url, start=10, length=20)["final_url"] == url
    assert list(admission.consulted) == [url]


@pytest.mark.parametrize("domain", ["bad host.org", "postgresql..org", "-bad.org"])
def test_invalid_domain_permissions_fail_before_retrieval(domain: str) -> None:
    with pytest.raises(ValueError, match="lowercase hostnames"):
        AuthoredScope(restriction="domains", allowed_domains=[domain])
