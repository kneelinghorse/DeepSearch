"""Regression tests for named-entity preservation in targeted query generation."""

from __future__ import annotations

import asyncio

from deepsearch.agent import refinement


def _gaps() -> list[dict[str, object]]:
    return [
        {"objective_id": "OBJ01", "current_score": 0.4, "reasoning": "needs deeper evidence"},
    ]


def test_generate_targeted_queries_preserves_author_year_citations() -> None:
    """Author-year citations from the mission objective should survive query rewriting."""

    async def run() -> None:
        queries = refinement.generate_targeted_queries(
            _gaps(),
            {
                "summary": "llm evaluation methods",
                "objective": (
                    "Compare semantic entropy from Farquhar et al. 2024 with "
                    "self-evaluation from Kadavath et al. 2022 for silent-failure detection."
                ),
            },
            objective_lookup=["Identify the best silent-failure metric"],
        )

        assert queries
        assert any("Farquhar et al. 2024" in query for query in queries)
        assert any("Kadavath et al. 2022" in query for query in queries)

    asyncio.run(run())


def test_generate_targeted_queries_preserves_all_caps_acronyms() -> None:
    """All-caps acronyms from the mission objective should appear verbatim in queries."""

    async def run() -> None:
        queries = refinement.generate_targeted_queries(
            _gaps(),
            {
                "summary": "agent probing methods",
                "objective": "Compare internal activation probes SAPLMA and CCS for calibration.",
            },
            objective_lookup=["Survey activation-based calibration methods"],
        )

        assert queries
        assert any("SAPLMA" in query for query in queries)
        assert any("CCS" in query for query in queries)

    asyncio.run(run())


def test_generate_targeted_queries_preserves_named_technique_phrases() -> None:
    """Proper-noun technique phrases should be appended to the generated query set."""

    async def run() -> None:
        queries = refinement.generate_targeted_queries(
            _gaps(),
            {
                "summary": "compression-based evaluation",
                "objective": (
                    'Compare "Language Modeling Is Compression" with Minimum '
                    "Description Length approaches for model evaluation."
                ),
            },
            objective_lookup=["Find compression-based complexity metrics"],
        )

        assert queries
        assert any("Language Modeling Is Compression" in query for query in queries)
        assert any("Minimum Description Length" in query for query in queries)

    asyncio.run(run())


def test_generate_targeted_queries_preserves_camel_case_backend_names() -> None:
    """CamelCase backend names from the objective should survive query rewriting."""

    async def run() -> None:
        queries = refinement.generate_targeted_queries(
            _gaps(),
            {
                "summary": "langgraph persistence",
                "objective": (
                    "Explain when to use MemorySaver, SqliteSaver, "
                    "AsyncSqliteSaver, and PostgresSaver in practice."
                ),
            },
            objective_lookup=["Explain when to use each backend in practice"],
        )

        assert queries
        assert any("MemorySaver" in query for query in queries)
        assert any("SqliteSaver" in query for query in queries)
        assert any("AsyncSqliteSaver" in query for query in queries)
        assert any("PostgresSaver" in query for query in queries)

    asyncio.run(run())


def test_generate_targeted_queries_without_named_entities_is_unchanged() -> None:
    """Objectives without named entities should produce content-token queries.

    Sprint 47 (`_objective_query_terms` fix): instruction-lead verbs like
    `study` are now stripped from the objective term tokenizer, so the
    resulting query carries content tokens only.
    """

    async def run() -> None:
        queries = refinement.generate_targeted_queries(
            _gaps(),
            {
                "summary": "agent reliability",
                "objective": "compare generic calibration methods for agents",
            },
            objective_lookup=["study failure detection tradeoffs"],
        )

        assert queries == ["agent reliability failure detection tradeoffs"]

    asyncio.run(run())
