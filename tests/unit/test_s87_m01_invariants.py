"""Sprint 87 m01 — contract fidelity and mission-scoped authority invariants."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from deepsearch.agent.codeagent_runner import (
    CodeAgentToolBundle,
    derive_authority_tier,
)
from deepsearch.agent.contract_to_prompt import build_codeagent_prompts
from deepsearch.agent.tools import rerank as rerank_mod
from deepsearch.agent.tools.source_fetch import source_fetch
from deepsearch.agent.tools.url_liveness import url_liveness
from deepsearch.mission import contract as contract_mod


def _compiled_state() -> dict:
    return {
        "mission_id": "S87-CONTRACT",
        "mission_context": {
            "title": "UNCOMPILED DECOY",
            "constraints": ["UNCOMPILED CONSTRAINT"],
        },
        "mission_contract": {
            "title": "Compiled mission title",
            "objective": "Compiled mission objective",
            "background": "Compiled mission background",
            "constraints": ["Keep quantitative claims dated"],
            "excluded_domains": ["blocked.example"],
            "reference_titles": ["Canonical primary study"],
            "authority_seed_domains": ["primary.example"],
            "required_entities": ["Declared Entity"],
            "excluded_entities": ["Excluded Candidate"],
            "named_entities": ["Declared Entity"],
            "named_entities_query_surfaces": {
                "Declared Entity": ["retrieval alias unique"]
            },
            "objectives": [
                {"text": "Research objective unique", "required": True},
                {"text": "Optional objective absent", "required": False},
            ],
            "deliverables": ["Deliverable unique"],
        },
    }


def test_contract_compiler_preserves_runtime_prompt_and_authority_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compiler must not discard fields that shape retrieval or drafting."""

    monkeypatch.setattr(
        contract_mod,
        "_extract_entities_llm_first",
        lambda **_: ([], {}, {}, "test", []),
    )
    monkeypatch.setattr(
        contract_mod,
        "_disambiguate_declared_entities",
        lambda **_: ({}, {"Baymard Institute": ["Baymard UX research"]}, "test"),
    )
    state = {
        "mission_id": "S87-COMPILE",
        "mission_objectives": ["Compare the evidence"],
        "mission_context": {
            "title": "Authority contract",
            "objective": "Compare the evidence",
            "background": "Mission-specific background",
            "constraints": ["Prefer primary evidence"],
            "excluded_domains": ["HTTPS://WWW.Blocked.Example/path"],
            "reference_titles": ["Explicit Reference"],
            "references": [
                {
                    "title": "Baymard methodology",
                    "url": "https://www.baymard.com/research/methodology",
                }
            ],
            "primary_domains": ["NnGroup.com"],
            "required_entities": ["Baymard Institute"],
            "excluded_entities": ["SEO summaries"],
        },
    }

    compiled = contract_mod.compile_contract_from_state(state, origin="test")

    assert compiled.contract_version == "1.2"
    assert compiled.background == "Mission-specific background"
    assert compiled.constraints == ["Prefer primary evidence"]
    assert compiled.excluded_domains == ["blocked.example"]
    assert compiled.reference_titles == ["Explicit Reference", "Baymard methodology"]
    assert compiled.authority_seed_domains == ["nngroup.com", "baymard.com"]
    assert compiled.required_entities == ["Baymard Institute"]
    assert compiled.excluded_entities == ["SEO summaries"]
    assert compiled.named_entities_query_surfaces == {
        "Baymard Institute": ["Baymard UX research"]
    }


def test_prompt_reads_compiled_contract_and_renders_each_field_section_once() -> None:
    """A decoy mission_context cannot contradict the compiled runtime contract."""

    system, _ = build_codeagent_prompts(_compiled_state())

    assert "UNCOMPILED DECOY" not in system
    assert "UNCOMPILED CONSTRAINT" not in system
    for heading in (
        "# Mission Constraints",
        "# Required Entities",
        "# Out of scope",
        "# Excluded Domains",
        "# Reference Titles",
        "# Mission-Scoped High-Authority Domains",
        "# Named-Entity Query Surfaces",
        "# Research Objectives",
        "# Required Deliverables",
    ):
        assert system.count(heading) == 1
    for unique_value in (
        "Keep quantitative claims dated",
        "blocked.example",
        "Canonical primary study",
        "primary.example",
        "Excluded Candidate",
        "retrieval alias unique",
        "Research objective unique",
        "Deliverable unique",
    ):
        assert system.count(unique_value) == 1
    assert "Optional objective absent" not in system


@pytest.mark.parametrize(
    "domain",
    [
        "baymard.com",
        "nngroup.com",
        "mcmaster.com",
        "digikey.com",
        "tech.walmart.com",
        "autocare.org",
    ],
)
def test_cat_primary_domains_are_high_only_for_the_declaring_mission(domain: str) -> None:
    url = f"https://sub.{domain}/primary-evidence"

    assert derive_authority_tier(url, high_authority_domains=[domain]) == "high"
    assert derive_authority_tier(url) == "low"


@pytest.mark.parametrize(
    "mission_seed,seeded_url,unrelated_url",
    [
        ("kernel.org", "https://kernel.org/process/", "https://retaildive.com/a"),
        ("baymard.com", "https://baymard.com/research", "https://poetryfoundation.org/a"),
        ("poetryfoundation.org", "https://poetryfoundation.org/poems", "https://kernel.org/a"),
    ],
)
def test_authority_seeding_generalizes_without_cross_mission_leakage(
    mission_seed: str,
    seeded_url: str,
    unrelated_url: str,
) -> None:
    assert derive_authority_tier(
        seeded_url, high_authority_domains=[mission_seed]
    ) == "high"
    assert derive_authority_tier(unrelated_url) == "low"
    assert derive_authority_tier(seeded_url) == "low"


def test_authority_multiplier_changes_order_under_rrf_score_compression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mission seeding must affect final order, not just output labels."""

    monkeypatch.setattr(
        rerank_mod,
        "_bm25_scores",
        lambda query, docs: np.asarray([2.0, 1.0]),
    )
    monkeypatch.setattr(
        rerank_mod,
        "_dense_scores",
        lambda query, docs: np.asarray([2.0, 1.0]),
    )
    results = [
        {"title": "Generic summary", "url": "https://summary.example/a", "snippet": "same"},
        {"title": "Primary evidence", "url": "https://primary.example/a", "snippet": "same"},
    ]

    without_seed = rerank_mod.rerank("query", results, top_k=2)
    with_seed = rerank_mod.rerank(
        "query",
        results,
        top_k=2,
        high_authority_domains=["primary.example"],
    )

    assert without_seed[0]["url"] == "https://summary.example/a"
    assert with_seed[0]["url"] == "https://primary.example/a"
    assert with_seed[0]["authority_tier"] == "high"
    assert with_seed[0]["score"] > with_seed[1]["score"]


def test_scholar_default_is_applied_before_score_multiplier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rerank_mod,
        "_bm25_scores",
        lambda query, docs: np.asarray([1.0, 1.0]),
    )
    monkeypatch.setattr(
        rerank_mod,
        "_dense_scores",
        lambda query, docs: np.asarray([1.0, 1.0]),
    )
    results = [
        {"title": "Forum", "url": "https://reddit.com/r/papers", "snippet": "paper"},
        {"title": "Publisher", "url": "https://publisher.example/paper", "snippet": "paper"},
    ]

    ranked = rerank_mod.rerank(
        "paper",
        results,
        top_k=2,
        default_authority_tier="high",
    )

    assert [item["authority_tier"] for item in ranked] == ["high", "low"]
    assert ranked[0]["score"] > ranked[1]["score"]


def test_authority_telemetry_uses_scored_sources_not_legacy_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rerank_mod,
        "_bm25_scores",
        lambda query, docs: np.asarray([1.0, 1.0]),
    )
    monkeypatch.setattr(
        rerank_mod,
        "_dense_scores",
        lambda query, docs: np.asarray([1.0, 1.0]),
    )
    scored = rerank_mod.rerank(
        "evidence",
        [
            {"url": "https://primary.example/a", "title": "A", "snippet": "evidence"},
            {"url": "https://generic.example/b", "title": "B", "snippet": "evidence"},
        ],
        top_k=2,
        high_authority_domains=["primary.example"],
    )
    bundle = CodeAgentToolBundle(tools=[], pool=[], sources_collected=scored)

    assert bundle.pool_size == 0
    assert bundle.authority_tier_summary() == {"high": 1, "medium": 0, "low": 1}
    assert {item["authority_tier"] for item in bundle.sources_collected} == {
        "high",
        "low",
    }


def test_source_fetch_blocks_excluded_redirect_before_target_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manual redirect boundary rejects exclusions before transport."""

    requested_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        if request.url.host == "allowed.example":
            return httpx.Response(
                302,
                headers={"location": "https://child.blocked.example/secret"},
                request=request,
            )
        pytest.fail("excluded redirect target reached the transport")

    real_client = httpx.Client

    def client_factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    import deepsearch.agent.tools.source_fetch as source_fetch_mod

    monkeypatch.setattr(source_fetch_mod.httpx, "Client", client_factory)
    monkeypatch.setattr(
        source_fetch_mod,
        "_resolve_host_addresses",
        lambda host, port: {"93.184.216.34"},
    )
    result = source_fetch(
        "https://allowed.example/start",
        excluded_domains=["BLOCKED.EXAMPLE"],
    )

    assert result["status"] == "blocked"
    assert result["error"] == "excluded_domain"
    assert result["final_url"] == "https://child.blocked.example/secret"
    assert requested_hosts == ["allowed.example"]


def test_web_search_filters_excluded_subdomains_before_reranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deepsearch.agent.tools.web_search as web_search_mod

    class FakeClient:
        async def search(self, *, query: str, max_results: int):
            return [
                SimpleNamespace(
                    title="Blocked",
                    url="https://news.blocked.example/article",
                    snippet="relevant evidence",
                ),
                SimpleNamespace(
                    title="Allowed",
                    url="https://allowed.example/article",
                    snippet="relevant evidence",
                ),
            ]

    reranked_inputs: list[list[dict]] = []

    def fake_rerank(query, results, top_k, **kwargs):
        reranked_inputs.append(list(results))
        return [
            {**item, "score": 1.0, "authority_tier": "low"}
            for item in results
        ]

    monkeypatch.setattr(web_search_mod, "LinkupSearchClient", FakeClient)
    monkeypatch.setattr(web_search_mod, "rerank", fake_rerank)

    output = web_search_mod.web_search(
        "evidence",
        excluded_domains=["https://www.blocked.example/path"],
    )

    assert [item["url"] for item in output] == ["https://allowed.example/article"]
    assert [item["url"] for item in reranked_inputs[0]] == [
        "https://allowed.example/article"
    ]


def test_url_liveness_blocks_excluded_redirect_before_target_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        if request.url.host == "allowed.example":
            return httpx.Response(
                302,
                headers={"location": "https://blocked.example/probe"},
                request=request,
            )
        pytest.fail("excluded liveness redirect target reached the transport")

    real_client = httpx.Client

    def client_factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    import deepsearch.agent.tools.source_fetch as source_fetch_mod
    import deepsearch.agent.tools.url_liveness as url_liveness_mod

    monkeypatch.setattr(url_liveness_mod.httpx, "Client", client_factory)
    monkeypatch.setattr(
        source_fetch_mod,
        "_resolve_host_addresses",
        lambda host, port: {"93.184.216.34"},
    )
    result = url_liveness(
        "https://allowed.example/start",
        excluded_domains=["blocked.example"],
    )

    assert result == {
        "alive": False,
        "status_code": 0,
        "final_url": "https://blocked.example/probe",
        "error": "excluded_domain",
    }
    assert requested_hosts == ["allowed.example"]
