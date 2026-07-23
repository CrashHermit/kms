"""
Persist the structural node stream and the entity overlay into Neo4j — the I/O half of the graph.

``persist_nodes`` writes the ``:Node`` layer: one ``:Source`` vertex for the book, one vertex per
``ASTNode`` (base ``:Node`` label + its per-type label), all MERGEd on their deterministic uuids so
re-running a book is idempotent, then wires them up — ``(:Source)-[:HEAD]->`` the first node and
``:NEXT`` edges threading the rest in document order so the stream hangs off the source and is
walkable in Cypher. ``persist_entities`` writes the ``:Entity`` overlay on top: one vertex per
Definition / Theorem / Problem (base ``:Entity`` label + its per-type label), rooted under the book
via ``:HAS_ENTITY`` and linked back to the structural chunks it was built from via ``:HAS_MEMBER``.

Writes are batched: Cypher can't parameterize a label, but the label comes from a closed enum
(``NodeType`` / ``EntityType``), so grouping by label and interpolating it is safe and turns the
whole stream into one MERGE per label plus a couple for the source/edges — no per-vertex
round-trips. The pure planning (grouping, edge pairs, head) is factored out and unit-tested; only the
``session.run`` calls need a live database.
"""

from collections import defaultdict
from typing import Any

from kms.core.models import ASTNode, Entity
from kms.graph.db import database, driver
from kms.graph.entities import (
    ENTITY_LABEL,
    entity_label,
    entity_properties,
    entity_uuid,
    member_uuid,
)
from kms.graph.nodes import (
    NODE_LABEL,
    SOURCE_LABEL,
    node_label,
    node_properties,
    node_uuid,
    source_properties,
    source_uuid,
)


def node_batches(nodes: list[ASTNode], source: str) -> dict[str | None, list[dict]]:
    """Group the nodes' property maps by their per-type label, so each label is one batched
    MERGE. The ``None`` bucket holds any typeless node (base ``:Node`` label only)."""
    batches: dict[str | None, list[dict]] = defaultdict(list)
    for node in nodes:
        batches[node_label(node)].append(node_properties(node, source))
    return dict(batches)


def next_pairs(nodes: list[ASTNode], source: str) -> list[dict]:
    """The ``{from, to}`` uuid pairs for the ``:NEXT`` chain: consecutive nodes in the
    document-ordered stream. Empty for a stream of fewer than two nodes."""
    return [
        {"from": node_uuid(source, a.id), "to": node_uuid(source, b.id)}
        for a, b in zip(nodes, nodes[1:], strict=False)  # deliberately uneven: consecutive pairs
    ]


def head_uuid(nodes: list[ASTNode], source: str) -> str | None:
    """The uuid of the stream's first node — the ``:HEAD`` the source hangs off — or None if the
    stream is empty."""
    return node_uuid(source, nodes[0].id) if nodes else None


async def persist_nodes(
    nodes: list[ASTNode], source: str, metadata: dict[str, Any] | None = None
) -> None:
    """Upsert the book's ``:Source`` root, its structural node stream, the ``:HEAD`` link, and the
    ``:NEXT`` chain. Idempotent: every MERGE keys on a deterministic uuid, so re-persisting the same
    ``source`` updates in place. A no-op for an empty stream. ``source`` is the book identity and
    ``metadata`` its optional attributes; every node's id must be assigned (post-flatten)."""
    if not nodes:
        return
    src = source_properties(source, metadata)
    batches = node_batches(nodes, source)
    pairs = next_pairs(nodes, source)
    head = head_uuid(nodes, source)

    async with driver().session(database=database()) as session:
        await session.run(
            f"MERGE (s:{SOURCE_LABEL} {{uuid: $uuid}}) SET s += $props",
            uuid=src["uuid"],
            props=src,
        )
        for label, rows in batches.items():
            query = f"UNWIND $rows AS row MERGE (n:{NODE_LABEL} {{uuid: row.uuid}}) SET n += row"
            if label:
                query += f" SET n:{label}"
            await session.run(query, rows=rows)

        await session.run(
            f"MATCH (s:{SOURCE_LABEL} {{uuid: $src}}), (n:{NODE_LABEL} {{uuid: $head}}) "
            f"MERGE (s)-[:HEAD]->(n)",
            src=src["uuid"],
            head=head,
        )
        if pairs:
            await session.run(
                f"UNWIND $pairs AS pair "
                f"MATCH (a:{NODE_LABEL} {{uuid: pair.from}}), (b:{NODE_LABEL} {{uuid: pair.to}}) "
                f"MERGE (a)-[:NEXT]->(b)",
                pairs=pairs,
            )


def entity_batches(entities: list[Entity], source: str) -> dict[str, list[dict]]:
    """Group the entities' property maps by their per-type label, so each label is one batched
    MERGE. Every entity is typed, so there is no ``None`` bucket (unlike ``node_batches``)."""
    batches: dict[str, list[dict]] = defaultdict(list)
    for entity in entities:
        batches[entity_label(entity)].append(entity_properties(entity, source))
    return dict(batches)


def member_pairs(entities: list[Entity], source: str) -> list[dict]:
    """The ``{entity, node}`` uuid pairs for the ``:HAS_MEMBER`` edges: one per (entity, member) so
    an entity links to every source chunk it was built from. Members are node ids resolved to the
    ``:Node`` layer's deterministic uuids."""
    return [
        {"entity": entity_uuid(source, entity.id), "node": member_uuid(source, member)}
        for entity in entities
        for member in entity.members
    ]


async def persist_entities(entities: list[Entity], source: str) -> None:
    """Upsert the book's ``:Entity`` overlay: one vertex per entity (base ``:Entity`` label + its
    per-type label), rooted under the already-persisted ``:Source`` via ``:HAS_ENTITY``, and linked
    to its structural chunks via ``:HAS_MEMBER``. Idempotent — every MERGE keys on a deterministic
    uuid, so re-persisting the same ``source`` updates in place. A no-op for an empty overlay. The
    ``:Source`` and ``:Node`` vertices are expected to already exist (the node persister runs first);
    the MATCHes here attach to them rather than creating them. Every entity's id must be assigned
    (post-flatten)."""
    if not entities:
        return
    batches = entity_batches(entities, source)
    pairs = member_pairs(entities, source)
    src = source_uuid(source)
    uuids = [entity_uuid(source, entity.id) for entity in entities]

    async with driver().session(database=database()) as session:
        for label, rows in batches.items():
            await session.run(
                f"UNWIND $rows AS row MERGE (e:{ENTITY_LABEL} {{uuid: row.uuid}}) "
                f"SET e += row SET e:{label}",
                rows=rows,
            )
        await session.run(
            f"MATCH (s:{SOURCE_LABEL} {{uuid: $src}}) "
            f"UNWIND $uuids AS uuid "
            f"MATCH (e:{ENTITY_LABEL} {{uuid: uuid}}) "
            f"MERGE (s)-[:HAS_ENTITY]->(e)",
            src=src,
            uuids=uuids,
        )
        if pairs:
            await session.run(
                f"UNWIND $pairs AS pair "
                f"MATCH (e:{ENTITY_LABEL} {{uuid: pair.entity}}), (n:{NODE_LABEL} {{uuid: pair.node}}) "
                f"MERGE (e)-[:HAS_MEMBER]->(n)",
                pairs=pairs,
            )
