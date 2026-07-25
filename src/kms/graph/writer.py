"""
Persist the structural node stream and the entity overlay into Neo4j — the I/O half of the graph.

``persist_nodes`` writes the ``:Node`` layer: one ``:Source`` vertex for the book, one vertex per
``models.ASTNode`` (base ``:Node`` label + its per-type label), all MERGEd on their deterministic uuids so
re-running a book is idempotent, then wires them up — ``(:Source)-[:HEAD]->`` the first node and
``:NEXT`` edges threading the rest in document order so the stream hangs off the source and is
walkable in Cypher. ``persist_entities`` writes the ``:Entity`` overlay on top: one vertex per pedagogical block (a bare
``:Entity`` label — ``type`` is an open property, never a label), rooted under the book via
``:HAS_ENTITY`` and linked back to the structural chunks it was built from via ``:DERIVED_FROM``.
``persist_procedures`` writes the procedural layer: one ``:Procedure`` per derivation hung off its
entity via ``:HAS_PROCEDURE`` and linked to its own source chunks via ``:DERIVED_FROM``, and one
``:Act`` per step threaded ``:FIRST``/``:THEN`` (see ``graph.procedures``).

The concept layer is currently dark — its only source was the deleted ``field`` taxonomy — so there
is no ``persist_concepts`` here yet; ``graph.concepts`` keeps the hub identity scheme the
conceptualization pass will write through. The reference/canonical layers are gone entirely.

Writes are batched: Cypher can't parameterize a label, but the structural node labels come from a
closed enum (``models.NodeType``), so grouping by label and interpolating it is safe and turns the
whole stream into one MERGE per label plus a couple for the source/edges — no per-vertex
round-trips. Entities, procedures and acts each carry a single fixed label, so each is one batched
MERGE. The pure planning (grouping, edge pairs, head) is factored out and unit-tested; only the
``session.run`` calls need a live database.
"""

from collections import defaultdict
from typing import Any

from kms.core import models
from kms.graph.db import database, driver
from kms.graph.entities import ENTITY_LABEL, entity_properties, entity_uuid
from kms.graph.nodes import (
    NODE_LABEL,
    SOURCE_LABEL,
    node_label,
    node_properties,
    node_uuid,
    source_properties,
    source_uuid,
)
from kms.graph.procedures import (
    ACT_LABEL,
    PROCEDURE_LABEL,
    act_rows,
    first_pairs,
    has_procedure_pairs,
    procedure_member_pairs,
    procedure_rows,
    then_pairs,
)


def node_batches(
    nodes: list[models.ASTNode], source: str
) -> dict[str | None, list[dict]]:
    """Group the nodes' property maps by their per-type label, so each label is one batched
    MERGE. The ``None`` bucket holds any typeless node (base ``:Node`` label only)."""
    batches: dict[str | None, list[dict]] = defaultdict(list)
    for node in nodes:
        batches[node_label(node)].append(node_properties(node, source))
    return dict(batches)


def next_pairs(nodes: list[models.ASTNode], source: str) -> list[dict]:
    """The ``{from, to}`` uuid pairs for the ``:NEXT`` chain: consecutive nodes in the
    document-ordered stream. Empty for a stream of fewer than two nodes."""
    return [
        {'from': node_uuid(source, a.id), 'to': node_uuid(source, b.id)}
        for a, b in zip(
            nodes, nodes[1:], strict=False
        )  # deliberately uneven: consecutive pairs
    ]


def head_uuid(nodes: list[models.ASTNode], source: str) -> str | None:
    """The uuid of the stream's first node — the ``:HEAD`` the source hangs off — or None if the
    stream is empty."""
    return node_uuid(source, nodes[0].id) if nodes else None


async def persist_nodes(
    nodes: list[models.ASTNode],
    source: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Upsert the book's ``:Source`` root, its structural node stream, the ``:HEAD`` link, and the
    ``:NEXT`` chain. Idempotent: every MERGE keys on a deterministic uuid, so re-persisting the same
    ``source`` updates in place. A no-op for an empty stream. ``source`` is the book identity and
    ``metadata`` its optional attributes; every node's id must be assigned (post-flatten)."""
    if not nodes:
        return
    source_props = source_properties(source, metadata)
    batches = node_batches(nodes, source)
    pairs = next_pairs(nodes, source)
    head = head_uuid(nodes, source)

    async with driver().session(database=database()) as session:
        await session.run(
            f'MERGE (s:{SOURCE_LABEL} {{uuid: $uuid}}) SET s += $props',
            uuid=source_props['uuid'],
            props=source_props,
        )
        for label, rows in batches.items():
            query = f'UNWIND $rows AS row MERGE (n:{NODE_LABEL} {{uuid: row.uuid}}) SET n += row'
            if label:
                query += f' SET n:{label}'
            await session.run(query, rows=rows)

        await session.run(
            f'MATCH (s:{SOURCE_LABEL} {{uuid: $src}}), (n:{NODE_LABEL} {{uuid: $head}}) '
            f'MERGE (s)-[:HEAD]->(n)',
            src=source_props['uuid'],
            head=head,
        )
        if pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (a:{NODE_LABEL} {{uuid: pair.from}}), (b:{NODE_LABEL} {{uuid: pair.to}}) '
                f'MERGE (a)-[:NEXT]->(b)',
                pairs=pairs,
            )


def entity_rows(entities: list[models.Entity], source: str) -> list[dict]:
    """Every entity's property map, one flat list. Entities carry a single ``:Entity`` label —
    ``type`` is an open induced property, never a label — so one batched MERGE writes them all."""
    return [entity_properties(entity, source) for entity in entities]


def member_pairs(entities: list[models.Entity], source: str) -> list[dict]:
    """The ``{entity, node}`` uuid pairs for the ``:DERIVED_FROM`` edges: one per (entity, member) so
    an entity links to every source chunk it was built from. A member id resolves to the ``:Node``
    layer's own deterministic ``node_uuid``, so the edge lands on the real provenance chunk."""
    return [
        {
            'entity': entity_uuid(source, entity.id),
            'node': node_uuid(source, member),
        }
        for entity in entities
        for member in entity.members
    ]


async def persist_entities(entities: list[models.Entity], source: str) -> None:
    """Upsert the book's ``:Entity`` overlay: one vertex per pedagogical block (a bare ``:Entity``
    label), rooted under the already-persisted ``:Source`` via ``:HAS_ENTITY``, and linked to its
    structural chunks via ``:DERIVED_FROM``. Idempotent — every MERGE keys on a deterministic uuid,
    so re-persisting the same ``source`` updates in place. A no-op for an empty overlay. The
    ``:Source`` and ``:Node`` vertices are expected to already exist (the node persister runs first);
    the MATCHes here attach to them rather than creating them. Every entity's id must be assigned
    (post-flatten)."""
    if not entities:
        return
    rows = entity_rows(entities, source)
    pairs = member_pairs(entities, source)
    source_key = source_uuid(source)
    uuids = [entity_uuid(source, entity.id) for entity in entities]

    async with driver().session(database=database()) as session:
        await session.run(
            f'UNWIND $rows AS row MERGE (e:{ENTITY_LABEL} {{uuid: row.uuid}}) SET e += row',
            rows=rows,
        )
        await session.run(
            f'MATCH (s:{SOURCE_LABEL} {{uuid: $src}}) '
            f'UNWIND $uuids AS uuid '
            f'MATCH (e:{ENTITY_LABEL} {{uuid: uuid}}) '
            f'MERGE (s)-[:HAS_ENTITY]->(e)',
            src=source_key,
            uuids=uuids,
        )
        if pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (e:{ENTITY_LABEL} {{uuid: pair.entity}}), (n:{NODE_LABEL} {{uuid: pair.node}}) '
                f'MERGE (e)-[:DERIVED_FROM]->(n)',
                pairs=pairs,
            )


async def persist_procedures(
    entities: list[models.Entity], source: str
) -> None:
    """Upsert the procedural layer: one ``:Procedure`` per derivation (a bare ``:Procedure``
    label), hung off its entity via ``:HAS_PROCEDURE`` and linked to its own source chunks via
    ``:DERIVED_FROM``; one ``:Act`` per step, threaded ``:FIRST`` from the procedure and ``:THEN``
    along the steps. Idempotent — every MERGE keys on a deterministic uuid. A no-op when no entity
    carries a derivation. The ``:Entity`` and ``:Node`` vertices are expected to already exist (the
    node and entity persisters run first); the MATCHes attach to them. Every entity's id must be
    assigned (post-flatten)."""
    procedure_batch = procedure_rows(entities, source)
    if not procedure_batch:
        return
    acts = act_rows(entities, source)
    owners = has_procedure_pairs(entities, source)
    members = procedure_member_pairs(entities, source)
    firsts = first_pairs(entities, source)
    thens = then_pairs(entities, source)

    async with driver().session(database=database()) as session:
        await session.run(
            f'UNWIND $rows AS row MERGE (p:{PROCEDURE_LABEL} {{uuid: row.uuid}}) SET p += row',
            rows=procedure_batch,
        )
        if acts:
            await session.run(
                f'UNWIND $rows AS row MERGE (a:{ACT_LABEL} {{uuid: row.uuid}}) SET a += row',
                rows=acts,
            )
        await session.run(
            f'UNWIND $pairs AS pair '
            f'MATCH (e:{ENTITY_LABEL} {{uuid: pair.entity}}), '
            f'(p:{PROCEDURE_LABEL} {{uuid: pair.procedure}}) '
            f'MERGE (e)-[:HAS_PROCEDURE]->(p)',
            pairs=owners,
        )
        if members:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (p:{PROCEDURE_LABEL} {{uuid: pair.procedure}}), '
                f'(n:{NODE_LABEL} {{uuid: pair.node}}) '
                f'MERGE (p)-[:DERIVED_FROM]->(n)',
                pairs=members,
            )
        if firsts:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (p:{PROCEDURE_LABEL} {{uuid: pair.procedure}}), '
                f'(a:{ACT_LABEL} {{uuid: pair.act}}) '
                f'MERGE (p)-[:FIRST]->(a)',
                pairs=firsts,
            )
        if thens:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (a:{ACT_LABEL} {{uuid: pair.from}}), (b:{ACT_LABEL} {{uuid: pair.to}}) '
                f'MERGE (a)-[:THEN]->(b)',
                pairs=thens,
            )
