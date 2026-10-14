"""S49.2 — per-slot alias families for concept-like entity slots.

When a required entity in the mission contract is a concept-like phrase
(e.g. ``multi-agent coordination protocols``) rather than an exact named
entity, registered acronym, or canonical method name, the literature
typically names the family members rather than the literal phrase. The
admission layer consults this registry as a SECOND-PASS match in
``_augment_target_slots_from_contract`` and ``_contract_filter_reason``:
when a retrieval's title/abstract does not contain the canonical phrase
but does contain an alias-family term, the slot binds anyway.

Per-slot lookup, NOT a global relaxation. Only entries below participate.
Exact named entities, acronyms, and canonical-name slots are unaffected
because they have no entry here.

The surface below mirrors the documented alias families in
``tests/test_entity_slot_alias_admission_diagnostic`` (S49.1 diagnostic).
A regression test verifies the two surfaces stay aligned.
"""

from __future__ import annotations

from typing import Mapping, Tuple


# Keys are normalized to lowercased canonical entity text. Values are
# ordered tuples of alias terms; alias matching uses word-boundary
# regex so multi-word terms are treated as phrases.
_ALIAS_FAMILIES: Mapping[str, Tuple[str, ...]] = {
    "multi-agent coordination protocols": (
        "contract net protocol",
        "agent communication language",
        "FIPA-ACL",
        "coordination mechanism",
        "distributed task allocation",
        "multi-agent communication",
    ),
    "treaty or agreement": (
        "electronic institution",
        "social law",
        "normative system",
        "multi-agent governance",
        "agent contract",
        "cooperative game theory",
    ),
    "anti-gaming": (
        "specification gaming",
        "reward hacking",
        "anti-collusion",
        "manipulation-resistant mechanism",
        "robust mechanism design",
        "Goodhart",
    ),
}


def lookup_alias_family(entity: str) -> Tuple[str, ...]:
    """Return the alias family for ``entity``, or ``()`` if no entry registered.

    Match key is the lower-cased, stripped canonical entity text. Returns
    an empty tuple for any entity that is not a registered concept-like
    slot — that case must NOT alter admission behavior, since exact named
    entities and acronyms are explicitly out of scope for this registry.
    """

    key = str(entity or "").strip().lower()
    return _ALIAS_FAMILIES.get(key, ())


def registered_concept_like_entities() -> Tuple[str, ...]:
    """Return the list of concept-like entities with a registered family."""

    return tuple(_ALIAS_FAMILIES.keys())
