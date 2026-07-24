"""
Graph representation of the concept layer — the ``:Concept`` nodes and their ``:INSTANCE_OF`` edges
(the ``φ`` conceptualization axis; see ``docs/UNIFIED-KG.md``).

A concept is an abstract category an entity instantiates. This module maps them onto Neo4j, mirroring
``graph.nodes``/``graph.entities``: pure mapping (no neo4j driver — that lives in ``graph.db``, the
writes in ``graph.writer``). A concept is its OWN kind (``:Concept``), not an ``:Entity``, and it is
**born canonical** — a corpus-level hub with a **global** (not source-scoped) uuid, so an entity's
field in book A and the same field in book B converge on ONE concept node. That convergence is the
whole point: it is the connective tissue the retrieval/curriculum queries traverse.

Scope (math-first, step 2 substrate): the only concept source today is the entity's ``field``
attribute — a *field*-concept (algebra / analysis / …). It reifies into a ``:Concept:Field`` node with
an ``(:Entity)-[:INSTANCE_OF]->(:Concept)`` edge, no new extraction needed. Richer per-entity concepts
(the specific mathematical concept a definition is *about*) and the ``:BROADER`` concept taxonomy
(MSC-anchored) are a later stage that adds concept *sources* here without changing this substrate —
``_entity_concepts`` returns a list precisely so more sources slot in.

Identity: deterministic uuid5 over ``(concept type, normalized name)``, ``concept#`` segment, GLOBAL
(no source) — like the reference ``canonical`` but its own kind. Normalization lowercases and collapses
whitespace, giving cheap exact-name clustering.
"""

from collections import defaultdict
from uuid import NAMESPACE_URL, uuid5

from kms.core.models import Entity
from kms.graph.entities import entity_uuid

CONCEPT_LABEL = "Concept"
FIELD_CONCEPT = "field"  # the one concept type sourced today (the entity's `field` attribute)


def normalize_concept(name: str) -> str:
    """The clustering key for a concept name: lowercased, whitespace-collapsed. Trivial spacing/case
    variants share a concept; genuine paraphrases stay distinct until a semantic dedup tier merges."""
    return " ".join(name.split()).lower()


def concept_uuid(concept_type: str, name: str) -> str:
    """Stable, deterministic vertex key for a concept: uuid5 over ``(concept type, normalized name)``.
    Global on purpose — NO ``source`` prefix — so the same concept from different books/entities
    resolves to one node. The ``concept#`` segment keeps it disjoint from every other uuid namespace."""
    return uuid5(
        NAMESPACE_URL, f"concept#{concept_type.strip().lower()}#{normalize_concept(name)}"
    ).hex


def concept_type_label(concept_type: str) -> str:
    """The per-type label for a concept (``"field"`` -> ``"Field"``), applied alongside the base
    ``:Concept`` label. Concept types are single lowercase words, so capitalizing is a valid label."""
    return concept_type.strip().lower().capitalize()


def concept_properties(concept_type: str, name: str) -> dict:
    """The Neo4j property map for one concept: its global uuid, its ``type`` (field / …), and the
    ``name`` as written. No ``source``: a concept is corpus-level (born canonical), not book-scoped."""
    return {
        "uuid": concept_uuid(concept_type, name),
        "type": concept_type.strip().lower(),
        "name": name.strip(),
    }


def _entity_concepts(entity: Entity) -> list[tuple[str, str]]:
    """The ``(concept_type, name)`` concepts an entity instantiates. Today just its ``field`` (a
    field-concept), if set; richer concept sources are added here later without touching callers."""
    return [(FIELD_CONCEPT, entity.field)] if entity.field else []


def concept_batches(entities: list[Entity]) -> dict[str, list[dict]]:
    """The unique concept property maps across the overlay — de-duplicated by uuid and grouped by
    per-type label, so each label is one batched MERGE (mirrors ``canonical_batches``)."""
    seen: dict[str, tuple[str, dict]] = {}
    for entity in entities:
        for concept_type, name in _entity_concepts(entity):
            props = concept_properties(concept_type, name)
            seen[props["uuid"]] = (concept_type_label(concept_type), props)
    batches: dict[str, list[dict]] = defaultdict(list)
    for label, props in seen.values():
        batches[label].append(props)
    return dict(batches)


def instance_rows(entities: list[Entity], source: str) -> list[dict]:
    """The ``{entity, concept}`` uuid pairs for the ``:INSTANCE_OF`` edges — one per (entity, concept)
    it instantiates. The entity uuid is source-scoped (a mention); the concept uuid is global."""
    return [
        {"entity": entity_uuid(source, entity.id), "concept": concept_uuid(concept_type, name)}
        for entity in entities
        for concept_type, name in _entity_concepts(entity)
    ]
