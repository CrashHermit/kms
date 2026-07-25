"""
Graph representation of the pedagogical block overlay — the semantic tier over the node stream.

The group finder builds a sparse overlay of blocks on the flat node stream, and the statement
extractor fills their self-contained attributes. This module maps a ``core.models.Entity`` onto
its Neo4j form, mirroring ``graph.nodes`` for the structural layer: pure mapping, free of the
neo4j driver (that lives in ``graph.db``, the writes in ``graph.writer``).

Representation: every entity carries the bare ``:Entity`` label and nothing else. There is NO
per-type label: ``type`` is an OPEN, induced property (definition / theorem / example / law / …)
and an open vocabulary would explode the label set — ``kind = label, type = property`` (see
``docs/SCHEMA.md``). There is likewise no ``:Mention``/``:Canonical`` role split: canonical hubs
existed only as reference targets, and the reference layer is gone. An entity roots under its
book via ``(:Source)-[:HAS_ENTITY]->(:Entity)`` and points back at the structural chunks it was
built from via ``(:Entity)-[:DERIVED_FROM]->(:Node)`` (its ``members`` are node ids, resolved to
the same deterministic node uuids the ``:Node`` layer wrote). Its derivations reify into the
procedural layer (``graph.procedures``).

Identity: the stable vertex key is a DETERMINISTIC uuid5 over ``(source, entity id)`` — the id is
the entity's document-order position, assigned when the overlay is flattened (see
``core.models.flatten_entities``) — so re-persisting a book MERGEs onto the same vertices instead
of duplicating them, and ``source`` disambiguates the same id across different books. The
``entity#`` segment keeps these uuids disjoint from the node uuids (which key on ``source#index``).

Structured attributes: the scalar attributes (type, label, number, title, instruction) and
``contents`` (a string array) map onto native Neo4j properties. The derivations do NOT: they are
the procedural half of the graph and reify into ``:Procedure`` + ``:Act`` structure via
``graph.procedures``, so they are absent from the entity's own property map.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes

ENTITY_LABEL = 'Entity'


def entity_uuid(source: str, entity_id: int) -> str:
    """Stable, deterministic vertex key for an entity: uuid5 over the book ``source`` and the
    entity's document-order ``id``. The ``entity#`` segment keeps it disjoint from node uuids
    (which key on ``source#index``); ``source`` keeps the same id in two different books distinct."""
    return uuid5(NAMESPACE_URL, f'{source}#entity#{entity_id}').hex


def entity_properties(entity: models.Entity, source: str) -> dict:
    """The Neo4j property map for one entity: its stable uuid, the source link, the open ``type``,
    the self-contained scalar attributes, and ``contents`` as a native string array. Derivations
    are NOT here — they reify into ``:Procedure`` / ``:Act`` structure via ``graph.procedures``.
    None and empty attributes are omitted rather than written as nulls, mirroring how the finder /
    extractor layer leaves them unset. Precondition: ``entity.id`` is set (true post-flatten)."""
    props = {
        'uuid': entity_uuid(source, entity.id),
        'source': nodes.source_uuid(source),  # links back to the :Source node
        'type': entity.type,
        'label': entity.label,
        'number': entity.number,
        'title': entity.title,
        'instruction': entity.instruction,
        'contents': entity.contents or None,
    }
    return {key: value for key, value in props.items() if value is not None}
