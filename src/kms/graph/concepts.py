"""
Concept identity — the global ``:Concept`` hub's naming and dedup scheme.

**This layer is currently DARK: nothing writes a ``:Concept`` node or an ``:INSTANCE_OF``
edge.** Its only source was the entity's ``field`` attribute, drawn from AutoMathKG's closed
seven-value mathematical-field taxonomy, and that taxonomy is deleted. What survives here is
the half that is not AutoMathKG at all — concepts, conceptualization and the ``φ`` mapping are
AutoSchemaKG's contribution — namely the identity and dedup scheme the real conceptualization
pass will build on (see ``docs/CONCEPT-LAYER.md``).

A concept is its OWN kind (``:Concept``), not an ``:Entity``, and it is **born canonical** — a
corpus-level hub with a **global** (not source-scoped) uuid, so the same concept induced from
book A and book B converges on ONE node with no cross-book pass needed. That convergence is the
whole point: with block-to-block relations gone, concepts are the only connective tissue between
blocks.

Normalization lowercases and collapses whitespace, giving cheap exact-name clustering. It merges
spacing and case variants only; genuine paraphrases ("linear algebra" vs "vector spaces") stay
distinct until the embedding/fusion tier merges them and re-points the ``:INSTANCE_OF`` edges.
The induced phrase rides on that edge as a property rather than becoming a node of its own — a
concept mention has no content beyond the string, so nodehood would multiply the graph by
(entities + procedures + acts) x ~3 for no gain.
"""

from uuid import NAMESPACE_URL, uuid5

CONCEPT_LABEL = 'Concept'


def normalize_concept(name: str) -> str:
    """The clustering key for a concept name: lowercased, whitespace-collapsed. Trivial spacing/case
    variants share a concept; genuine paraphrases stay distinct until a semantic dedup tier merges."""
    return ' '.join(name.split()).lower()


def concept_uuid(name: str) -> str:
    """Stable, deterministic vertex key for a concept: uuid5 over the normalized name. Global on
    purpose — NO ``source`` prefix — so the same concept from different books/entities resolves to
    one node. The ``concept#`` segment keeps it disjoint from every other uuid namespace."""
    return uuid5(NAMESPACE_URL, f'concept#{normalize_concept(name)}').hex


def concept_properties(name: str) -> dict:
    """The Neo4j property map for one concept: its global uuid and the ``name`` as written. No
    ``source``: a concept is corpus-level (born canonical), not book-scoped."""
    return {'uuid': concept_uuid(name), 'name': name.strip()}


def concept_batch(names: list[str]) -> list[dict]:
    """The unique concept property maps for a run of induced names, de-duplicated by uuid.
    Concepts carry a single ``:Concept`` label, so one batched MERGE writes them all."""
    seen: dict[str, dict] = {}
    for name in names:
        if name and name.strip():
            props = concept_properties(name)
            seen[props['uuid']] = props
    return list(seen.values())
