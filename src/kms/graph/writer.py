"""
Persist the structural node stream into Neo4j — the I/O half of the ``:Node`` layer.

Writes one ``:Source`` vertex for the book, one vertex per ``ASTNode`` (base ``:Node`` label + its
per-type label), all MERGEd on their deterministic uuids so re-running a book is idempotent, then
wires them up: ``(:Source)-[:HEAD]->`` the first node, and ``:NEXT`` edges threading the rest in
document order so the stream hangs off the source and is walkable in Cypher.

Writes are batched: Cypher can't parameterize a label, but the label comes from the closed
``NodeType`` enum, so grouping nodes by label and interpolating it is safe and turns the whole
stream into one MERGE per label plus a couple for the source/edges — no per-node round-trips. The
pure planning (grouping, edge pairs, head) is factored out and unit-tested; only the ``session.run``
calls need a live database.
"""

from collections import defaultdict
from typing import Any

from kms.core.models import ASTNode
from kms.graph.db import database, driver
from kms.graph.nodes import (
    NODE_LABEL,
    SOURCE_LABEL,
    node_label,
    node_properties,
    node_uuid,
    source_properties,
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
