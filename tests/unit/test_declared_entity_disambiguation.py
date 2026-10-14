"""Unit tests for the S60.1 declared-entity disambiguation pass."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pydantic")

from deepsearch.mission.entity_extractor_llm import (
    DECLARED_ENTITY_SYSTEM_INSTRUCTION,
    DeclaredEntityDisambiguationOutcome,
    DeclaredEntityDisambiguationResult,
    DeclaredEntityDisambiguator,
    DisambiguatedEntity,
    run_declared_entity_disambiguation_sync,
)


# --------------------------------------------------------------------------- #
# DeclaredEntityDisambiguator + run_declared_entity_disambiguation_sync
# --------------------------------------------------------------------------- #


class _FakeStructuredExtractor:
    def __init__(self, result: DeclaredEntityDisambiguationResult | None) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def extract(self, *, system_instruction: str, prompt: str, schema: type) -> Any:
        self.calls.append(
            {
                "system_instruction": system_instruction,
                "prompt": prompt,
                "schema": schema,
            }
        )
        return self.result


def test_system_instruction_covers_disambiguation_and_query_surfaces() -> None:
    """The disambiguation prompt instructs both disambiguating_context and query_surfaces."""

    assert "DISAMBIGUATION CONTEXT" in DECLARED_ENTITY_SYSTEM_INSTRUCTION
    assert "QUERY SURFACES" in DECLARED_ENTITY_SYSTEM_INSTRUCTION
    assert "AUTHORITATIVE" in DECLARED_ENTITY_SYSTEM_INSTRUCTION
    # The Railway homonym example is the canonical motivating case.
    assert "Railway" in DECLARED_ENTITY_SYSTEM_INSTRUCTION
    assert "railroad" in DECLARED_ENTITY_SYSTEM_INSTRUCTION.lower()
    assert "Fly.io" in DECLARED_ENTITY_SYSTEM_INSTRUCTION


def test_skipped_no_declared_returns_outcome_with_empty_maps() -> None:
    """When the declared list is empty, the LLM is not invoked."""

    fake = _FakeStructuredExtractor(
        DeclaredEntityDisambiguationResult(entities=[])
    )
    disambiguator = DeclaredEntityDisambiguator(structured_extractor=fake)

    outcome = run_declared_entity_disambiguation_sync(
        disambiguator,
        declared_entities=[],
        objective="x",
        background="",
        deliverables=[],
    )

    assert isinstance(outcome, DeclaredEntityDisambiguationOutcome)
    assert outcome.disambiguation == {}
    assert outcome.query_surfaces == {}
    assert outcome.path == "skipped_no_declared"
    assert fake.calls == []  # LLM not invoked when declared list is empty


def test_declared_disambiguation_happy_path() -> None:
    """LLM-emitted disambiguation + query_surfaces flow through to the outcome."""

    result = DeclaredEntityDisambiguationResult(
        entities=[
            DisambiguatedEntity(
                name="Railway",
                disambiguating_context=(
                    "Railway.app — the deployment platform, not the railroad industry"
                ),
                query_surfaces=["Railway app hosting", "railway.app deployment platform"],
            ),
            DisambiguatedEntity(
                name="Fly.io",
                disambiguating_context="Fly.io — the application hosting platform",
                query_surfaces=["Fly.io hosting platform", "Fly.io edge deployment"],
            ),
            DisambiguatedEntity(
                name="AWS Lambda",
                disambiguating_context="",
                query_surfaces=[],
            ),
        ]
    )
    fake = _FakeStructuredExtractor(result)
    disambiguator = DeclaredEntityDisambiguator(structured_extractor=fake)

    outcome = run_declared_entity_disambiguation_sync(
        disambiguator,
        declared_entities=["AWS Lambda", "Railway", "Fly.io"],
        objective="Compare backend hosting platforms for a Figma plugin.",
        background="Burst-and-idle workloads with multi-minute browser automation.",
        deliverables=["A 5-row classification matrix"],
    )

    assert outcome is not None
    assert outcome.path == "llm_declared"
    assert outcome.disambiguation == {
        "Railway": "Railway.app — the deployment platform, not the railroad industry",
        "Fly.io": "Fly.io — the application hosting platform",
    }
    assert outcome.query_surfaces == {
        "Railway": ["Railway app hosting", "railway.app deployment platform"],
        "Fly.io": ["Fly.io hosting platform", "Fly.io edge deployment"],
    }
    # Background and deliverables reach the prompt body.
    prompt = fake.calls[0]["prompt"]
    assert "MISSION OBJECTIVE" in prompt
    assert "Burst-and-idle workloads" in prompt
    assert "5-row classification matrix" in prompt
    for entity in ("AWS Lambda", "Railway", "Fly.io"):
        assert entity in prompt


def test_declared_disambiguation_drops_entities_not_in_declared_list() -> None:
    """The declared list is authoritative; the LLM cannot smuggle in new entities."""

    result = DeclaredEntityDisambiguationResult(
        entities=[
            DisambiguatedEntity(
                name="Railway",
                disambiguating_context="Railway.app deployment platform",
                query_surfaces=["Railway app hosting"],
            ),
            DisambiguatedEntity(
                name="Heroku",  # not declared — must be dropped
                disambiguating_context="Heroku PaaS",
                query_surfaces=["Heroku platform"],
            ),
        ]
    )
    disambiguator = DeclaredEntityDisambiguator(
        structured_extractor=_FakeStructuredExtractor(result)
    )

    outcome = run_declared_entity_disambiguation_sync(
        disambiguator,
        declared_entities=["Railway"],
        objective="x",
        background="",
        deliverables=[],
    )

    assert outcome is not None
    assert outcome.path == "llm_declared"
    assert outcome.disambiguation == {"Railway": "Railway.app deployment platform"}
    assert outcome.query_surfaces == {"Railway": ["Railway app hosting"]}
    assert "Heroku" not in outcome.disambiguation
    assert "Heroku" not in outcome.query_surfaces


def test_declared_disambiguation_canonical_match_is_case_insensitive() -> None:
    """LLM may echo casing differently; canonical declared-list spelling wins."""

    result = DeclaredEntityDisambiguationResult(
        entities=[
            DisambiguatedEntity(
                name="vercel functions",
                disambiguating_context="Vercel Functions serverless runtime",
                query_surfaces=["Vercel Functions serverless"],
            ),
        ]
    )
    disambiguator = DeclaredEntityDisambiguator(
        structured_extractor=_FakeStructuredExtractor(result)
    )

    outcome = run_declared_entity_disambiguation_sync(
        disambiguator,
        declared_entities=["Vercel Functions"],
        objective="x",
        background="",
        deliverables=[],
    )

    assert outcome is not None
    # Output is keyed by the declared canonical spelling, not the LLM echo.
    assert "Vercel Functions" in outcome.disambiguation
    assert "vercel functions" not in outcome.disambiguation
    assert outcome.query_surfaces["Vercel Functions"] == ["Vercel Functions serverless"]


def test_declared_disambiguation_filters_canonical_from_query_surfaces() -> None:
    """The canonical name itself must not appear in query_surfaces — planner uses it as primary query."""

    result = DeclaredEntityDisambiguationResult(
        entities=[
            DisambiguatedEntity(
                name="Railway",
                disambiguating_context="",
                query_surfaces=["Railway", "Railway app", "railway.app", "railway"],
            ),
        ]
    )
    disambiguator = DeclaredEntityDisambiguator(
        structured_extractor=_FakeStructuredExtractor(result)
    )

    outcome = run_declared_entity_disambiguation_sync(
        disambiguator,
        declared_entities=["Railway"],
        objective="x",
        background="",
        deliverables=[],
    )

    assert outcome is not None
    # "Railway" (canonical) and the case-different "railway" are both filtered;
    # remaining surfaces preserve order and dedup.
    assert outcome.query_surfaces["Railway"] == ["Railway app", "railway.app"]


def test_declared_disambiguation_returns_none_on_extractor_failure() -> None:
    """When the structured extractor returns None, the outcome is None (caller falls back)."""

    fake = _FakeStructuredExtractor(None)
    disambiguator = DeclaredEntityDisambiguator(structured_extractor=fake)
    outcome = run_declared_entity_disambiguation_sync(
        disambiguator,
        declared_entities=["Railway"],
        objective="x",
        background="",
        deliverables=[],
    )
    assert outcome is None


def test_declared_disambiguation_skips_blank_only_declared_list() -> None:
    """Whitespace-only declared entries are treated as empty."""

    fake = _FakeStructuredExtractor(
        DeclaredEntityDisambiguationResult(entities=[])
    )
    disambiguator = DeclaredEntityDisambiguator(structured_extractor=fake)
    outcome = run_declared_entity_disambiguation_sync(
        disambiguator,
        declared_entities=["", "  ", None],  # type: ignore[list-item]
        objective="x",
        background="",
        deliverables=[],
    )
    assert outcome is not None
    assert outcome.path == "skipped_no_declared"
    assert fake.calls == []


# --------------------------------------------------------------------------- #
# Contract compiler integration
# --------------------------------------------------------------------------- #


def _stub_disambiguation(
    monkeypatch, outcome: DeclaredEntityDisambiguationOutcome | None
) -> None:
    """Patch the contract module's disambiguation entry point.

    Stubs both the structured-extractor build and the sync wrapper so the
    contract path returns ``outcome`` deterministically without an LLM call.
    """

    from deepsearch.mission import contract as contract_module
    from deepsearch.mission import entity_extractor_llm

    class _StubSettings:
        @classmethod
        def load(cls):
            return cls()

        def build_structured_extractor(self):
            return object()

    monkeypatch.setattr(contract_module, "DeepSearchSettings", _StubSettings, raising=False)
    import deepsearch.config as config_module

    monkeypatch.setattr(config_module, "DeepSearchSettings", _StubSettings, raising=False)

    monkeypatch.setattr(
        entity_extractor_llm,
        "run_declared_entity_disambiguation_sync",
        lambda *args, **kwargs: outcome,
    )


def _declared_state(*, objective: str, declared: list[str], background: str = "") -> dict[str, Any]:
    return {
        "mission_id": "S60-TEST",
        "mission_context": {
            "title": "S60 Declared-Entity Disambiguation Test",
            "objective": objective,
            "background": background,
            "deliverables": [],
            "success_criteria": [],
            "required_entities": declared,
        },
    }


def test_compile_contract_runs_disambiguation_for_declared_entities(monkeypatch) -> None:
    """When required_entities is non-empty, the disambiguation outcome populates the contract."""

    from deepsearch.mission.contract import compile_contract_from_state

    outcome = DeclaredEntityDisambiguationOutcome(
        disambiguation={
            "Railway": "Railway.app deployment platform, not the railroad industry",
            "Fly.io": "Fly.io application hosting",
        },
        query_surfaces={
            "Railway": ["Railway app hosting", "railway.app deployment platform"],
            "Fly.io": ["Fly.io hosting platform"],
        },
        path="llm_declared",
    )
    _stub_disambiguation(monkeypatch, outcome)

    state = _declared_state(
        objective="Compare backend hosting platforms for a Figma plugin.",
        declared=["AWS Lambda", "Google Cloud Run", "Vercel Functions", "Fly.io", "Railway"],
        background="Burst-and-idle workloads with browser automation.",
    )

    contract = compile_contract_from_state(state, origin="test")

    # Declared list is the entity slot list (S59.2 short-circuit unchanged).
    declared_lower = {e.lower() for e in contract.named_entities}
    for entity in ("aws lambda", "google cloud run", "vercel functions", "fly.io", "railway"):
        assert entity in declared_lower, f"missing declared entity {entity}"

    # S60.1 surfaces — disambiguation + query_surfaces populated by the pass.
    assert contract.named_entities_disambiguation == {
        "Railway": "Railway.app deployment platform, not the railroad industry",
        "Fly.io": "Fly.io application hosting",
    }
    assert contract.named_entities_query_surfaces == {
        "Railway": ["Railway app hosting", "railway.app deployment platform"],
        "Fly.io": ["Fly.io hosting platform"],
    }

    # Telemetry path tagged llm_declared (replaces the prior
    # "skipped_declared_authoritative" tag from the S59.2 short-circuit).
    paths = [
        row.get("reason")
        for row in contract.entity_extraction_decisions
        if row.get("decision") == "extraction_path"
    ]
    assert paths == ["llm_declared"]


def test_compile_contract_disambiguation_failure_falls_back_to_regex_path(
    monkeypatch,
) -> None:
    """When the disambiguation LLM call fails, path tag is regex_fallback and maps stay empty."""

    from deepsearch.mission.contract import compile_contract_from_state

    _stub_disambiguation(monkeypatch, None)

    state = _declared_state(
        objective="Compare backend hosting platforms.",
        declared=["AWS Lambda", "Railway"],
    )
    contract = compile_contract_from_state(state, origin="test")

    assert contract.named_entities_disambiguation == {}
    assert contract.named_entities_query_surfaces == {}
    paths = [
        row.get("reason")
        for row in contract.entity_extraction_decisions
        if row.get("decision") == "extraction_path"
    ]
    assert paths == ["regex_fallback"]


def test_compile_contract_skips_disambiguation_when_no_declared_entities(monkeypatch) -> None:
    """When required_entities is empty, the disambiguation pass does not run.

    The existing prose extractor's path tag (llm_primary or regex_fallback)
    is preserved end-to-end — S60.1 is a no-op on the prose-extraction path.
    """

    from deepsearch.mission.contract import compile_contract_from_state
    from deepsearch.mission import entity_extractor_llm
    from deepsearch.mission.entity_extractor_llm import EntityExtractionOutcome

    # Stub the prose extractor to return a deterministic llm_primary outcome.
    prose_outcome = EntityExtractionOutcome(
        entity_names=["axe-core"],
        disambiguation={},
        extracted=[],
        rejected=[],
        path="llm_primary",
        query_surfaces={},
    )

    class _StubSettings:
        @classmethod
        def load(cls):
            return cls()

        def build_structured_extractor(self):
            return object()

    from deepsearch.mission import contract as contract_module
    import deepsearch.config as config_module

    monkeypatch.setattr(contract_module, "DeepSearchSettings", _StubSettings, raising=False)
    monkeypatch.setattr(config_module, "DeepSearchSettings", _StubSettings, raising=False)
    monkeypatch.setattr(
        entity_extractor_llm,
        "run_llm_entity_extraction_sync",
        lambda *args, **kwargs: prose_outcome,
    )

    # Disambiguation should NOT be called — sentinel raises if it is.
    def _disambiguation_must_not_run(*args, **kwargs):  # pragma: no cover - sentinel
        raise AssertionError("disambiguation pass must not run when declared list is empty")

    monkeypatch.setattr(
        entity_extractor_llm,
        "run_declared_entity_disambiguation_sync",
        _disambiguation_must_not_run,
    )

    state = _declared_state(
        objective="Catalogue accessibility tools such as axe-core.",
        declared=[],
    )

    contract = compile_contract_from_state(state, origin="test")

    paths = [
        row.get("reason")
        for row in contract.entity_extraction_decisions
        if row.get("decision") == "extraction_path"
    ]
    assert paths == ["llm_primary"]
