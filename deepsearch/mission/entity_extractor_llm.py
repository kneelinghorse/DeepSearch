"""LLM-first named-entity extraction for mission contracts.

Sprint 54 L1. Wraps the provider-portable ``StructuredExtractor`` port with a
contract-specific schema and prompt. Output:

- A list of entity names (``ExtractedEntity.name``) suitable for entity slot
  compilation.
- A ``disambiguating_context`` map (entity → expanded form) that S54.3's
  query planner uses to defuse acronym homonym pollution upstream of
  retrieval admission.
- The full ``ExtractedEntity`` record list for telemetry / audit.

This module sits BETWEEN ``deepsearch.llm.structured_extractor`` (which
knows about LLM calls and Pydantic schemas) and
``deepsearch.mission.contract`` (which knows about entity slots and
contract semantics). It is the ONLY module in ``deepsearch.mission`` that
imports the structured-extractor port, by design.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Sequence

from pydantic import BaseModel, Field

from deepsearch.agent.execution_control import ExecutionCancelledError

logger = logging.getLogger(__name__)


class ExtractedEntity(BaseModel):
    """A single entity emitted by the LLM extractor.

    Notes:
    - ``name`` is the canonical entity string used to build evidence slots.
    - ``disambiguating_context`` is non-empty for ambiguous acronyms only
      (e.g., "Object-Oriented Design System" for OODS in a design-systems
      objective). Empty for unambiguous entities (e.g., "axe-core").
    - ``query_surfaces`` (S56.3) is a list of 2-4 retrieval-friendly variants
      of the entity name that the LLM emits alongside the canonical surface.
      The query planner appends these as additional retrieval queries so
      compound-phrase entities like "automated ARIA validation" get useful
      surface coverage (e.g., ["ARIA validation", "ARIA", "axe-core"]) without
      the planner growing regex-era decomposition rules.
    - ``rejected=True`` lets the extractor surface participial-fragment /
      non-entity candidates explicitly with ``rejection_reason``. These rows
      are filtered out before entity slots are built but appear in
      audit telemetry.
    """

    name: str = Field(default="")
    disambiguating_context: str = Field(default="")
    query_surfaces: list[str] = Field(default_factory=list)
    is_acronym: bool = Field(default=False)
    rejected: bool = Field(default=False)
    rejection_reason: str = Field(default="")


class EntityExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)


@dataclass(slots=True)
class EntityExtractionOutcome:
    """Sync-friendly outcome of an LLM-first extraction attempt.

    ``path`` is one of ``"llm_primary"`` (LLM call succeeded) or
    ``"llm_failed"`` (LLM call returned None — caller should run regex
    fallback). When the feature flag is OFF, ``LLMEntityExtractor`` is not
    invoked at all and this dataclass is not produced.

    ``query_surfaces`` (S56.3) maps each accepted entity name to the LLM-
    supplied retrieval-friendly variants (deduplicated against the canonical
    name; canonical is intentionally NOT included here so the planner can
    keep canonical-first ordering of its own).
    """

    entity_names: list[str]
    disambiguation: dict[str, str]
    extracted: list[ExtractedEntity]
    rejected: list[ExtractedEntity]
    path: str
    query_surfaces: dict[str, list[str]] = field(default_factory=dict)


SYSTEM_INSTRUCTION = (
    "You extract NAMED ENTITIES (specific tools, methods, products, organizations, "
    "concepts) from research mission objectives. Output ONLY entities that name a "
    "specific thing.\n\n"
    "Rules:\n"
    "1. REJECT participial phrases like 'including review workflows', 'spanning "
    "initial assessment', 'based on token analysis'. Mark them with rejected=true "
    "and a brief rejection_reason.\n"
    "2. For ambiguous acronyms (3-5 uppercase letters that could match multiple "
    "real-world referents — e.g. OODS, CCS, SAPLMA), supply disambiguating_context "
    "from the surrounding objective text: the words that disambiguate the acronym "
    "from common homonyms. For OODS in a design-systems mission, that is "
    "'Object-Oriented Design System'. For unambiguous entities (axe-core, "
    "eslint-plugin-jsx-a11y, t-SNE), leave disambiguating_context empty.\n"
    "3. For enumerated lists like 'tools such as axe-core, eslint-plugin-jsx-a11y, "
    "Pa11y, automated ARIA validators, ...', extract EVERY named item as a separate "
    "entity. Do not collapse the list to the first item or to an umbrella phrase. "
    "If the list is long (>5 items), still extract every item.\n"
    "4. Preserve every entity in the user-supplied 'required entities' list as "
    "authored — these are explicit user declarations.\n"
    "5. Set is_acronym=true only for actual acronyms (3-5 uppercase letters). "
    "Hyphenated or mixed-case names like 't-SNE' or 'axe-core' are NOT acronyms.\n"
    "6. Do not extract generic categories ('tools', 'methods', 'workflows') unless "
    "they are part of a compound proper-noun phrase.\n"
    "7. For each accepted entity, populate query_surfaces with 2-4 retrieval-"
    "friendly variants that a researcher would actually type into a search "
    "engine to find evidence about that entity. Rules of thumb: "
    "(a) include the acronym alone when the entity contains one (e.g., "
    "for 'OODS (Object-Oriented Design System)', include 'OODS'); "
    "(b) include the head noun phrase for compound surfaces (e.g., "
    "for 'automated ARIA validation', include 'ARIA validation'); "
    "(c) include a well-known concrete tool name when the entity describes "
    "a category that has a canonical implementation (e.g., for 'automated "
    "ARIA validation', include 'axe-core'); "
    "(d) prefer fewer better surfaces over many noisy ones — if no useful "
    "shorter surface exists, leave query_surfaces empty for that entity. "
    "Do NOT include the canonical entity name itself in query_surfaces; the "
    "planner already uses the canonical name as the primary query."
)


@dataclass(slots=True)
class LLMEntityExtractor:
    """Bridges the StructuredExtractor port to mission-contract entity extraction."""

    structured_extractor: object  # deepsearch.llm.structured_extractor.StructuredExtractor

    async def extract(
        self,
        *,
        objectives: Sequence[str],
        deliverables: Sequence[str],
        required_entities: Sequence[str],
    ) -> EntityExtractionResult | None:
        prompt = _build_user_prompt(
            objectives=list(objectives),
            deliverables=list(deliverables),
            required_entities=list(required_entities),
        )
        return await self.structured_extractor.extract(  # type: ignore[union-attr]
            system_instruction=SYSTEM_INSTRUCTION,
            prompt=prompt,
            schema=EntityExtractionResult,
        )


def _build_user_prompt(
    *,
    objectives: list[str],
    deliverables: list[str],
    required_entities: list[str],
) -> str:
    parts: list[str] = []
    if objectives:
        parts.append("OBJECTIVES:")
        for index, text in enumerate(objectives, start=1):
            cleaned = (text or "").strip()
            if cleaned:
                parts.append(f"{index}. {cleaned}")
    if deliverables:
        parts.append("\nDELIVERABLES:")
        for index, text in enumerate(deliverables, start=1):
            cleaned = (text or "").strip()
            if cleaned:
                parts.append(f"{index}. {cleaned}")
    if required_entities:
        parts.append("\nREQUIRED ENTITIES (user-declared, must appear verbatim in output):")
        for text in required_entities:
            cleaned = (text or "").strip()
            if cleaned:
                parts.append(f"- {cleaned}")
    parts.append(
        "\nReturn a JSON object matching the EntityExtractionResult schema with "
        "an 'entities' array. Include both extracted and rejected candidates."
    )
    return "\n".join(parts)


def run_llm_entity_extraction_sync(
    extractor: LLMEntityExtractor,
    *,
    objectives: Sequence[str],
    deliverables: Sequence[str],
    required_entities: Sequence[str],
) -> EntityExtractionOutcome | None:
    """Run the async LLM extractor from a synchronous context.

    Spawns a single-worker thread that runs ``asyncio.run`` so this is safe
    whether the caller is already inside an event loop (FastAPI route) or
    not (CLI runner). Returns None on hard failure; otherwise returns an
    ``EntityExtractionOutcome`` with the extractor's classification.
    """

    async def _coro() -> EntityExtractionResult | None:
        return await extractor.extract(
            objectives=objectives,
            deliverables=deliverables,
            required_entities=required_entities,
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(copy_context().run, asyncio.run, _coro())
            result = future.result()
    except ExecutionCancelledError:
        raise
    except Exception:
        logger.exception("LLM entity extraction call raised; falling back to regex extractor")
        return None

    if result is None:
        return None

    extracted: list[ExtractedEntity] = []
    rejected: list[ExtractedEntity] = []
    entity_names: list[str] = []
    disambiguation: dict[str, str] = {}
    query_surfaces: dict[str, list[str]] = {}
    seen_names: set[str] = set()

    for entry in result.entities or []:
        name = (entry.name or "").strip()
        if not name:
            continue
        if entry.rejected:
            rejected.append(entry)
            continue
        key = name.lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        extracted.append(entry)
        entity_names.append(name)
        context = (entry.disambiguating_context or "").strip()
        if context:
            disambiguation[name] = context
        cleaned_surfaces: list[str] = []
        seen_surfaces: set[str] = {key}
        for raw_surface in entry.query_surfaces or []:
            surface = (raw_surface or "").strip()
            if not surface:
                continue
            surface_key = surface.lower()
            if surface_key in seen_surfaces:
                continue
            seen_surfaces.add(surface_key)
            cleaned_surfaces.append(surface)
        if cleaned_surfaces:
            query_surfaces[name] = cleaned_surfaces

    return EntityExtractionOutcome(
        entity_names=entity_names,
        disambiguation=disambiguation,
        extracted=extracted,
        rejected=rejected,
        path="llm_primary",
        query_surfaces=query_surfaces,
    )


# ---------------------------------------------------------------------------
# S60.1 — Declared-Entity Disambiguation Pass
#
# When the author declares ``required_entities``, the prose extractor is
# short-circuited (S59.2) — but ``named_entities_disambiguation`` and
# ``named_entities_query_surfaces`` (S54.3 + S56.3 surfaces consumed by the
# query planner) become empty. Single-word brand entities with strong homonyms
# (Railway, Fly.io) get retrieval pollution because the entity-lane query is
# the bare entity name with no disambiguating qualifier.
#
# This pass runs a single focused LLM call when ``required_entities`` is
# non-empty: input = the declared list + objective + background + deliverables;
# output = (disambiguation, query_surfaces) for each declared entity. The
# declared list itself is authoritative — the LLM does NOT add or remove
# entities. Per planning-pass amendment §1, no ``disambiguation_hints``
# author-override field; the existing ``background`` field is the author-side
# surface for nudging context.
# ---------------------------------------------------------------------------


class DisambiguatedEntity(BaseModel):
    """A single entry in the declared-entity disambiguation result.

    - ``name`` matches one of the user's declared entities verbatim (case-
      insensitive matching is applied downstream).
    - ``disambiguating_context`` is a short phrase (≤120 chars) distinguishing
      the entity from common homonyms in the mission's domain. Empty when
      unambiguous.
    - ``query_surfaces`` is 2-4 retrieval-friendly variants the planner
      appends to the entity-lane query. Empty when the bare entity name
      retrieves cleanly.
    """

    name: str = Field(default="")
    disambiguating_context: str = Field(default="")
    query_surfaces: list[str] = Field(default_factory=list)


class DeclaredEntityDisambiguationResult(BaseModel):
    entities: list[DisambiguatedEntity] = Field(default_factory=list)


@dataclass(slots=True)
class DeclaredEntityDisambiguationOutcome:
    """Sync-friendly outcome of the declared-entity disambiguation pass.

    ``path`` is one of:
    - ``"llm_declared"`` — LLM call succeeded, disambiguation populated.
    - ``"regex_fallback"`` — LLM build/call failed; caller emits empty maps.
    - ``"skipped_no_declared"`` — declared list is empty; pass skipped.
    """

    disambiguation: dict[str, str]
    query_surfaces: dict[str, list[str]]
    path: str


DECLARED_ENTITY_SYSTEM_INSTRUCTION = (
    "You generate retrieval-friendly DISAMBIGUATION CONTEXT and QUERY "
    "SURFACES for a fixed list of named entities the author has declared "
    "as required for a research mission. The entity list is AUTHORITATIVE — "
    "do NOT add, remove, rename, or split entities. Your job is to help "
    "retrieval find the RIGHT thing for each entity given the surrounding "
    "mission context.\n\n"
    "For each declared entity, produce:\n"
    "1. disambiguating_context — a short phrase (≤120 chars) that "
    "distinguishes the entity from common homonyms in the mission domain. "
    "Leave empty when the entity is unambiguous in context. Examples in a "
    "hosting/deployment-platform mission:\n"
    "   - 'Railway' → 'Railway.app — the deployment platform, not the "
    "railroad industry'\n"
    "   - 'Fly.io' → 'Fly.io — the application hosting platform, not the "
    "insect or aviation'\n"
    "   - 'AWS Lambda' → '' (unambiguous in a serverless-hosting context)\n"
    "2. query_surfaces — 2-4 retrieval-friendly variants a researcher would "
    "type into a search engine to find documentation about the RIGHT entity. "
    "Append a disambiguation qualifier when the bare entity name is "
    "dominated by a homonym in general web search. Examples:\n"
    "   - 'Railway' → ['Railway app hosting', 'railway.app deployment "
    "platform']\n"
    "   - 'Fly.io' → ['Fly.io hosting platform', 'Fly.io edge deployment']\n"
    "   - 'Vercel Functions' → ['Vercel Functions serverless', 'Vercel "
    "Edge Functions']\n"
    "Do NOT include the canonical entity name itself in query_surfaces; the "
    "planner uses the canonical name as the primary query. If no useful "
    "variant exists for an entity (already unambiguous and well-indexed), "
    "leave query_surfaces empty for that entity.\n\n"
    "Output ONE entry per declared entity, in the order received. Use the "
    "exact spelling and casing from the declared list."
)


@dataclass(slots=True)
class DeclaredEntityDisambiguator:
    """Bridges the StructuredExtractor port to declared-entity disambiguation."""

    structured_extractor: object  # deepsearch.llm.structured_extractor.StructuredExtractor

    async def disambiguate(
        self,
        *,
        declared_entities: Sequence[str],
        objective: str,
        background: str,
        deliverables: Sequence[str],
    ) -> DeclaredEntityDisambiguationResult | None:
        prompt = _build_disambiguation_prompt(
            declared_entities=list(declared_entities),
            objective=objective,
            background=background,
            deliverables=list(deliverables),
        )
        return await self.structured_extractor.extract(  # type: ignore[union-attr]
            system_instruction=DECLARED_ENTITY_SYSTEM_INSTRUCTION,
            prompt=prompt,
            schema=DeclaredEntityDisambiguationResult,
        )


def _build_disambiguation_prompt(
    *,
    declared_entities: list[str],
    objective: str,
    background: str,
    deliverables: list[str],
) -> str:
    parts: list[str] = []
    objective_text = (objective or "").strip()
    if objective_text:
        parts.append("MISSION OBJECTIVE:")
        parts.append(objective_text)
    background_text = (background or "").strip()
    if background_text:
        parts.append("\nMISSION BACKGROUND:")
        parts.append(background_text)
    if deliverables:
        parts.append("\nDELIVERABLES:")
        for index, text in enumerate(deliverables, start=1):
            cleaned = (text or "").strip()
            if cleaned:
                parts.append(f"{index}. {cleaned}")
    parts.append("\nDECLARED REQUIRED ENTITIES (authoritative — do not modify):")
    for entity in declared_entities:
        cleaned = (entity or "").strip()
        if cleaned:
            parts.append(f"- {cleaned}")
    parts.append(
        "\nReturn a JSON object matching the DeclaredEntityDisambiguationResult "
        "schema with an 'entities' array. Emit one entry per declared entity, "
        "preserving exact spelling and casing."
    )
    return "\n".join(parts)


def run_declared_entity_disambiguation_sync(
    disambiguator: DeclaredEntityDisambiguator,
    *,
    declared_entities: Sequence[str],
    objective: str,
    background: str,
    deliverables: Sequence[str],
) -> DeclaredEntityDisambiguationOutcome | None:
    """Run the async disambiguation call from a synchronous context.

    Spawns a single-worker thread that runs ``asyncio.run`` so this is safe
    whether the caller is already inside an event loop (FastAPI route) or
    not (CLI runner). Returns ``None`` on hard failure (caller should treat
    as ``regex_fallback``); returns an outcome with ``path="skipped_no_declared"``
    when the declared list is empty.
    """

    declared_clean = [
        str(entity or "").strip() for entity in declared_entities if str(entity or "").strip()
    ]
    if not declared_clean:
        return DeclaredEntityDisambiguationOutcome(
            disambiguation={},
            query_surfaces={},
            path="skipped_no_declared",
        )

    async def _coro() -> DeclaredEntityDisambiguationResult | None:
        return await disambiguator.disambiguate(
            declared_entities=declared_clean,
            objective=objective,
            background=background,
            deliverables=deliverables,
        )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(copy_context().run, asyncio.run, _coro())
            result = future.result()
    except ExecutionCancelledError:
        raise
    except Exception:
        logger.exception(
            "LLM declared-entity disambiguation call raised; "
            "downstream emits empty disambiguation maps"
        )
        return None

    if result is None:
        return None

    declared_lookup = {entity.lower(): entity for entity in declared_clean}
    disambiguation: dict[str, str] = {}
    query_surfaces: dict[str, list[str]] = {}

    for entry in result.entities or []:
        name = (entry.name or "").strip()
        if not name:
            continue
        canonical = declared_lookup.get(name.lower())
        if canonical is None:
            # LLM returned an entity not in the declared list — drop it; the
            # declared list is authoritative.
            continue
        context = (entry.disambiguating_context or "").strip()
        if context:
            disambiguation[canonical] = context
        cleaned_surfaces: list[str] = []
        seen_surfaces: set[str] = {canonical.lower()}
        for raw_surface in entry.query_surfaces or []:
            surface = (raw_surface or "").strip()
            if not surface:
                continue
            surface_key = surface.lower()
            if surface_key in seen_surfaces:
                continue
            seen_surfaces.add(surface_key)
            cleaned_surfaces.append(surface)
        if cleaned_surfaces:
            query_surfaces[canonical] = cleaned_surfaces

    return DeclaredEntityDisambiguationOutcome(
        disambiguation=disambiguation,
        query_surfaces=query_surfaces,
        path="llm_declared",
    )


__all__ = [
    "DECLARED_ENTITY_SYSTEM_INSTRUCTION",
    "DeclaredEntityDisambiguationOutcome",
    "DeclaredEntityDisambiguationResult",
    "DeclaredEntityDisambiguator",
    "DisambiguatedEntity",
    "EntityExtractionOutcome",
    "EntityExtractionResult",
    "ExtractedEntity",
    "LLMEntityExtractor",
    "SYSTEM_INSTRUCTION",
    "run_declared_entity_disambiguation_sync",
    "run_llm_entity_extraction_sync",
]
