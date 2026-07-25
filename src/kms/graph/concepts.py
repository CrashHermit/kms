"""
Graph representation of the concept layer — the ``:Concept`` nodes and their ``:INSTANCE_OF`` edges
(the ``φ`` conceptualization axis; see ``docs/UNIFIED-KG.md`` and ``docs/GENERALIZATION.md``).

A concept is an abstract category an element instantiates. This module maps them onto Neo4j, mirroring
``graph.nodes``/``graph.entities``: pure mapping (no neo4j driver — that lives in ``graph.db``, the
writes in ``graph.writer``). A concept is its OWN kind (``:Concept``), not an ``:Entity``, and it is
**born canonical** — a corpus-level hub with a **global** (not source-scoped) uuid, so a concept in
book A and the same concept in book B converge on ONE concept node. That convergence is the whole
point: it is the connective tissue the retrieval/curriculum queries traverse.

Sources (AutoSchemaKG-style conceptualization, ``docs/GENERALIZATION.md``, "Concept layer redesign"):

* **Entities** — the conceptualizer tags each entity with several flat concept phrases spanning
  specific → general ("normal subgroup", "group theory", "abstract algebra"). Multi-granularity comes
  from attaching several flat tags of differing generality, NOT from a concept tree: a coarse tag is
  shared by many entities, a fine tag by few, and the generality gradient falls out of the sharing.
  This replaced AutoMathKG's fixed seven-value ``field`` taxonomy, which could not survive contact
  with a physics or biology textbook.
* **Events** — a procedure's steps reify into ``:Event`` s, and the same conceptualization runs over
  them ("proof by contradiction", "algebraic manipulation"), giving the procedural half of the graph
  the same concept handles as the declarative half.

Both use ONE concept type (``topic``) on purpose: an entity concept and an event concept that name
the same idea must land on the same vertex, which a per-axis type would prevent.

Identity: deterministic uuid5 over ``(concept type, normalized name)``, ``concept#`` segment, GLOBAL
(no source) — like the reference ``canonical`` but its own kind. Normalization lowercases and collapses
whitespace, giving cheap exact-name clustering; genuine paraphrases ("linear algebra" vs "vector
spaces") stay distinct until the embedding/fusion tier merges them.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph.entities import entity_uuid
from kms.graph.procedures import procedure_events

CONCEPT_LABEL = 'Concept'
# The one concept type minted today: an induced topic tag (from an entity or an event). Kept as a
# named constant because the uuid folds it in — a second type would be a second concept space.
TOPIC_CONCEPT = 'topic'


def normalize_concept(name: str) -> str:
    """The clustering key for a concept name: lowercased, whitespace-collapsed. Trivial spacing/case
    variants share a concept; genuine paraphrases stay distinct until a semantic dedup tier merges."""
    return ' '.join(name.split()).lower()


def concept_uuid(concept_type: str, name: str) -> str:
    """Stable, deterministic vertex key for a concept: uuid5 over ``(concept type, normalized name)``.
    Global on purpose — NO ``source`` prefix — so the same concept from different books/entities
    resolves to one node. The ``concept#`` segment keeps it disjoint from every other uuid namespace."""
    return uuid5(
        NAMESPACE_URL,
        f'concept#{concept_type.strip().lower()}#{normalize_concept(name)}',
    ).hex


def concept_properties(concept_type: str, name: str) -> dict:
    """The Neo4j property map for one concept: its global uuid, its ``type``, and the ``name`` as
    written. No ``source``: a concept is corpus-level (born canonical), not book-scoped."""
    return {
        'uuid': concept_uuid(concept_type, name),
        'type': concept_type.strip().lower(),
        'name': name.strip(),
    }


def _names(names: list[str]) -> list[str]:
    """The usable concept phrases in a raw tag list: non-empty, de-duplicated on the normalized key,
    first spelling wins. Guards the conceptualizer's output, which is free text from a model."""
    kept: dict[str, str] = {}
    for name in names:
        key = normalize_concept(name or '')
        if key and key not in kept:
            kept[key] = name.strip()
    return list(kept.values())


def concept_rows(entities: list[models.Entity], source: str) -> list[dict]:
    """The unique concept property maps across the overlay — every entity tag and every event tag,
    de-duplicated by uuid. One flat list: concepts carry a single ``:Concept`` label (their ``type``
    is a property), so one batched MERGE writes them all."""
    seen: dict[str, dict] = {}
    for _, name in _entity_concepts(entities):
        props = concept_properties(TOPIC_CONCEPT, name)
        seen.setdefault(props['uuid'], props)
    for _, name in _event_concepts(entities, source):
        props = concept_properties(TOPIC_CONCEPT, name)
        seen.setdefault(props['uuid'], props)
    return list(seen.values())


def _entity_concepts(
    entities: list[models.Entity],
) -> list[tuple[models.Entity, str]]:
    """The ``(entity, concept name)`` pairs the overlay's entities instantiate, tags cleaned."""
    return [
        (entity, name)
        for entity in entities
        for name in _names(entity.concepts)
    ]


def _event_concepts(
    entities: list[models.Entity], source: str
) -> list[tuple[str, str]]:
    """The ``(event uuid, concept name)`` pairs the overlay's procedure steps instantiate. The event
    uuid comes from ``procedures.procedure_events``, so it matches the ``:Event`` the procedural
    layer wrote."""
    return [
        (event, name)
        for entity in entities
        for event, step in procedure_events(entity, source)
        for name in _names(step.concepts)
    ]


def entity_instance_rows(
    entities: list[models.Entity], source: str
) -> list[dict]:
    """The ``{entity, concept}`` uuid pairs for the entity ``:INSTANCE_OF`` edges — one per (entity,
    concept) it instantiates. The entity uuid is source-scoped (a mention); the concept uuid is
    global, which is what makes two books' mentions of the same idea meet on one vertex."""
    return [
        {
            'entity': entity_uuid(source, entity.id),
            'concept': concept_uuid(TOPIC_CONCEPT, name),
        }
        for entity, name in _entity_concepts(entities)
    ]


def event_instance_rows(
    entities: list[models.Entity], source: str
) -> list[dict]:
    """The ``{event, concept}`` uuid pairs for the step-level ``:INSTANCE_OF`` edges — the procedural
    half of the same conceptualization. Kept apart from the entity rows so each write MATCHes on its
    own label rather than scanning both kinds."""
    return [
        {'event': event, 'concept': concept_uuid(TOPIC_CONCEPT, name)}
        for event, name in _event_concepts(entities, source)
    ]
