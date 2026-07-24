"""
Graph representation of the math-semantic entity overlay — the Anchor tier over the node stream.

The per-type finders build a sparse overlay of Definition / Theorem / Problem entities on the flat
node stream, and their attributors fill the self-contained AutoMathKG attributes. This module maps a
``core.Entity`` onto its Neo4j form, mirroring ``graph.nodes`` for the structural layer: the
vocabulary is ``core.EntityType`` (it invents no new types), and it stays free of the neo4j driver
(pure mapping) — the driver lives in ``graph.db``.

Representation: every entity carries the shared ``:Entity`` label AND its per-type label
(``:Entity:Problem``, ``:Entity:Definition``, ``:Entity:Theorem`` — Neo4j nodes hold multiple
labels), so ``MATCH (e:Theorem)`` is a native label scan with no property index. It roots under its
book via ``(:Source)-[:HAS_ENTITY]->(:Entity)`` and points back at the structural chunks it was built
from via ``(:Entity)-[:DERIVED_FROM]->(:Node)`` (the entity's ``members`` are node ids, resolved to the
same deterministic node uuids the ``:Node`` layer wrote). Cross-entity reference edges
(refs / references_tactics) and the step-level event layer are later graph-tier work; this layer is
entity-grain only.

Identity: the stable vertex key is a DETERMINISTIC uuid5 over ``(source, entity id)`` — the id is the
entity's document-order position, assigned when the three overlays are flattened (see
``core.flatten_entities``) — so re-persisting a book MERGEs onto the same vertices instead of
duplicating them, and ``source`` disambiguates the same id across different books. The ``entity#``
segment keeps these uuids disjoint from the node uuids (which key on ``source#index``).

Structured attributes: the scalar attributes (label, number, title, field, instruction) and
``contents`` (a string array) map onto native Neo4j properties. The entity's own ``bodylist`` — its
*statement* structure — stays here as a JSON-string property (declarative: it belongs to the claim).
The *derivations* — ``proofs`` and ``solutions`` — do NOT: they are the procedural half of the graph
and are reified into ``:Procedure`` + ``:Event`` structure by ``graph.procedures`` (see
``docs/UNIFIED-KG.md``), so they are absent from the entity's own property map.
"""

import json
from uuid import NAMESPACE_URL, uuid5

from kms.core.models import BodySegment, Entity
from kms.graph.nodes import source_uuid

ENTITY_LABEL = "Entity"


def entity_uuid(source: str, entity_id: int) -> str:
    """Stable, deterministic vertex key for an entity: uuid5 over the book ``source`` and the
    entity's document-order ``id``. The ``entity#`` segment keeps it disjoint from node uuids
    (which key on ``source#index``); ``source`` keeps the same id in two different books distinct."""
    return uuid5(NAMESPACE_URL, f"{source}#entity#{entity_id}").hex


def entity_label(entity: Entity) -> str:
    """The per-type label for an entity (``EntityType.THEOREM`` -> ``"Theorem"``). Applied ALONGSIDE
    the base ``:Entity`` label, never instead of it. Every entity is typed, so this is never None —
    unlike ``node_label``. The EntityType values are single lowercase words, so capitalizing yields a
    valid Neo4j label."""
    return entity.type.value.capitalize()


def _segment(segment: BodySegment) -> dict:
    """A bodylist segment as a plain dict (no pydantic ``.model_dump()``, so this stays importable
    under the test stubs — same reason the pipeline's JSON path unpacked it by hand)."""
    return {"description": segment.description, "action": segment.action}


def entity_properties(entity: Entity, source: str) -> dict:
    """The Neo4j property map for one entity: its stable uuid, the source link, the math type, the
    self-contained scalar attributes, ``contents`` as a native string array, and the entity's own
    (statement) ``bodylist`` as a JSON string (see the module docstring). Derivations (``proofs`` /
    ``solutions``) are NOT here — they reify into ``:Procedure`` / ``:Event`` structure via
    ``graph.procedures``. None and empty attributes are omitted rather than written as nulls, mirroring
    how the finder/attributor layer leaves them unset. Precondition: ``entity.id`` is set
    (true post-flatten)."""
    props = {
        "uuid": entity_uuid(source, entity.id),
        "source": source_uuid(source),  # links back to the :Source node
        "type": entity.type.value,
        "label": entity.label,
        "number": entity.number,
        "title": entity.title,
        "field": entity.field,
        "instruction": entity.instruction,
        "contents": entity.contents or None,
        "bodylist": (
            json.dumps([_segment(s) for s in entity.bodylist], ensure_ascii=False)
            if entity.bodylist
            else None
        ),
    }
    return {key: value for key, value in props.items() if value is not None}
