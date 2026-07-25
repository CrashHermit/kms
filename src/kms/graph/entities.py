"""
Graph representation of the entity overlay — the Anchor tier over the node stream.

The finders build a sparse overlay of entities on the flat node stream, and their attributors fill
the self-contained AutoMathKG attributes. This module maps a ``core.models.Entity`` onto its Neo4j
form, mirroring ``graph.nodes`` for the structural layer, and stays free of the neo4j driver (pure
mapping) — the driver lives in ``graph.db``.

Representation: **kind is the label, type is a property** (``docs/GENERALIZATION.md``). Every entity
carries the shared ``:Entity`` label plus the ``:Mention`` role label (a book-specific entity,
distinct from a reference ``:Entity:Canonical``), and its ``type`` — definition / theorem / law /
mechanism / … — rides as a property, NOT as a per-type label. That is a deliberate reversal of the
math-first schema, which minted ``:Entity:Theorem``: now that the type is *induced* per book rather
than drawn from a three-value enum, per-type labels would grow without bound and every query would
have to know the label set in advance. ``MATCH (e:Entity {type: 'theorem'})`` is a property lookup
backed by the ``entity_type`` index instead of a label scan — a bounded cost for an unbounded
vocabulary.

An entity roots under its book via ``(:Source)-[:HAS_ENTITY]->(:Entity)`` and points back at the
structural chunks it was built from via ``(:Entity)-[:DERIVED_FROM]->(:Node)`` (the entity's
``members`` are node ids, resolved to the same deterministic node uuids the ``:Node`` layer wrote).
Its derivations reify into the procedural layer (``graph.procedures``), its ``concepts`` into the
concept layer (``graph.concepts``), and its ``refs`` into reference edges onto canonicals
(``graph.references``).

Identity: the stable vertex key is a DETERMINISTIC uuid5 over ``(source, entity id)`` — the id is the
entity's document-order position, assigned when the overlays are flattened (see
``core.models.flatten_entities``) — so re-persisting a book MERGEs onto the same vertices instead of
duplicating them, and ``source`` disambiguates the same id across different books. The ``entity#``
segment keeps these uuids disjoint from the node uuids (which key on ``source#index``).

Structured attributes: the scalar attributes (label, number, title, type, instruction) and
``contents`` (a string array) map onto native Neo4j properties. The entity's own ``bodylist`` — its
*statement* structure — stays here as a JSON-string property (declarative: it belongs to the claim).
The *derivations* — its ``procedures`` — do NOT: they are the procedural half of the graph and are
reified into ``:Procedure`` + ``:Event`` structure by ``graph.procedures`` (see
``docs/UNIFIED-KG.md``), so they are absent from the entity's own property map. Nor are its
``concepts``: they become ``:Concept`` vertices with ``:INSTANCE_OF`` edges (``graph.concepts``).
"""

import json
from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph.nodes import source_uuid

ENTITY_LABEL = 'Entity'
# Role labels, applied ALONGSIDE the base :Entity label (see docs/UNIFIED-KG.md). A book-specific
# mention and a corpus-level canonical are both :Entity — the role label distinguishes them (and
# keeps `MATCH (:Entity {type: …})` seeing both), with no implicit "absence of label ⇒ mention" rule.
MENTION_LABEL = (
    'Mention'  # a book-specific entity (carries :DERIVED_FROM provenance)
)
CANONICAL_LABEL = 'Canonical'  # a corpus-level identity hub (minted from refs; see graph.references)


def entity_uuid(source: str, entity_id: int) -> str:
    """Stable, deterministic vertex key for an entity: uuid5 over the book ``source`` and the
    entity's document-order ``id``. The ``entity#`` segment keeps it disjoint from node uuids
    (which key on ``source#index``); ``source`` keeps the same id in two different books distinct."""
    return uuid5(NAMESPACE_URL, f'{source}#entity#{entity_id}').hex


def _segment(segment: models.BodySegment) -> dict:
    """A bodylist segment as a plain dict (no pydantic ``.model_dump()``, so this stays importable
    under the test stubs — same reason the pipeline's JSON path unpacked it by hand)."""
    return {'description': segment.description, 'action': segment.action}


def entity_properties(entity: models.Entity, source: str) -> dict:
    """The Neo4j property map for one entity: its stable uuid, the source link, the induced
    ``type``, the self-contained scalar attributes, ``contents`` as a native string array, and the
    entity's own (statement) ``bodylist`` as a JSON string (see the module docstring). Derivations
    (``procedures``) and ``concepts`` are NOT here — they reify into ``:Procedure`` / ``:Event`` and
    ``:Concept`` structure via ``graph.procedures`` / ``graph.concepts``. None and empty attributes
    are omitted rather than written as nulls, mirroring how the finder/attributor layer leaves them
    unset — including an untyped entity, whose vertex simply carries no ``type``. Precondition:
    ``entity.id`` is set (true post-flatten)."""
    props = {
        'uuid': entity_uuid(source, entity.id),
        'source': source_uuid(source),  # links back to the :Source node
        'type': entity.type,
        'label': entity.label,
        'number': entity.number,
        'title': entity.title,
        'instruction': entity.instruction,
        'contents': entity.contents or None,
        'bodylist': (
            json.dumps(
                [_segment(s) for s in entity.bodylist], ensure_ascii=False
            )
            if entity.bodylist
            else None
        ),
    }
    return {key: value for key, value in props.items() if value is not None}
