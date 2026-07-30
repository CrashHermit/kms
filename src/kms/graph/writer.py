"""
Persist the structural node stream and the statement/procedure overlay into
Neo4j.

``persist_nodes`` upserts ``:Source`` and ``:Node`` vertices (no edges).

``persist_chain`` writes the merged walkable chain: ``:HEAD`` from
``:Source`` to the first element, then ``:NEXT`` threading non-absorbed
prose :Node and :Statement vertices in document order. Absorbed raw
nodes are persisted but skipped in the chain.

``persist_statements`` writes the ``:Statement`` overlay and
``:HAS_STATEMENT`` edges from ``:Source``.

``persist_procedures`` writes ``:Procedure`` vertices and
``:HAS_PROCEDURE`` edges from their statements. ``:Act`` step chains are
declared but not yet written.

Writes are batched: structural node labels are grouped by their per-type
label and each batch is one MERGE. Statements, procedures and acts each
carry a single fixed label, so each is one batched MERGE.
"""

from collections import defaultdict
from typing import Any

from kms.core import models
from kms.graph.db import database, driver
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
    procedure_rows,
    then_pairs,
)
from kms.graph.statements import (
    STATEMENT_LABEL,
    statement_properties,
    statement_uuid,
)


def node_batches(
    nodes: list[models.ASTNode], source: str
) -> dict[str | None, list[dict]]:
    """Group the nodes' property maps by their per-type label, so each
    label is one batched MERGE."""
    batches: dict[str | None, list[dict]] = defaultdict(list)
    for node in nodes:
        batches[node_label(node)].append(node_properties(node, source))
    return dict(batches)


async def persist_nodes(
    nodes: list[models.ASTNode],
    source: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Upsert the book's ``:Source`` root and its ``:Node`` vertices.

    Vertices only — no ``:NEXT`` or ``:HEAD`` edges (those are written
    by ``persist_chain`` with the merged statement/prose ordering).
    """
    if not nodes:
        return
    source_props = source_properties(source, metadata)
    batches = node_batches(nodes, source)

    async with driver().session(database=database()) as session:
        await session.run(
            f'MERGE (s:{SOURCE_LABEL} {{uuid: $uuid}}) SET s += $props',
            uuid=source_props['uuid'],
            props=source_props,
        )
        for label, rows in batches.items():
            query = (
                f'UNWIND $rows AS row '
                f'MERGE (n:{NODE_LABEL} {{uuid: row.uuid}}) SET n += row'
            )
            if label:
                query += f' SET n:{label}'
            await session.run(query, rows=rows)


def _chain_elements(
    nodes: list[models.ASTNode],
    statements: list[models.Statement],
    source: str,
) -> list[dict]:
    """The merged chain's elements, in document order.

    Walks the raw node stream, slotting each statement in at its first
    member's place and skipping the members it absorbed — the absorbed nodes
    are still persisted for provenance, they just aren't in the walkable
    chain.

    Each element carries its ``label`` as well as its uuid. The label is not
    decoration: node and statement uuids are disjoint but a lookup by uuid
    alone still has to scan, and the per-label uniqueness constraints are what
    make the MATCHes in ``persist_chain`` indexed.

    Args:
        nodes: The flat node stream, in document order.
        statements: The statement overlay.
        source: The stable book identity.

    Returns:
        One ``{label, uuid}`` per chain element, in document order.
    """
    absorbed: set[int] = set()
    statement_by_first_node: dict[int, models.Statement] = {}
    for statement in statements:
        absorbed.update(statement.statement_of)
        # A statement's id IS its first member's id, so it keys both its place
        # in the stream and its own uuid — one identity, read one way.
        if statement.id is not None:
            statement_by_first_node[statement.id] = statement

    elements: list[dict] = []
    for node in nodes:
        node_id = node.id
        if node_id is None:
            continue
        statement = statement_by_first_node.get(node_id)
        if statement is not None:
            elements.append(
                {
                    'label': STATEMENT_LABEL,
                    'uuid': statement_uuid(source, statement.id),
                }
            )
        elif node_id not in absorbed:
            elements.append(
                {'label': NODE_LABEL, 'uuid': node_uuid(source, node_id)}
            )
    return elements


def _merged_chain(
    nodes: list[models.ASTNode],
    statements: list[models.Statement],
    source: str,
) -> list[dict]:
    """The consecutive element pairs of the merged ``:NEXT`` chain.

    Args:
        nodes: The flat node stream, in document order.
        statements: The statement overlay.
        source: The stable book identity.

    Returns:
        One ``{from, from_label, to, to_label}`` per ``:NEXT`` edge.
    """
    elements = _chain_elements(nodes, statements, source)
    return [
        {
            'from': current['uuid'],
            'from_label': current['label'],
            'to': following['uuid'],
            'to_label': following['label'],
        }
        for current, following in zip(elements, elements[1:], strict=False)
    ]


def _merged_head(
    nodes: list[models.ASTNode],
    statements: list[models.Statement],
    source: str,
) -> dict | None:
    """The merged chain's first element, or None if the chain is empty.

    Args:
        nodes: The flat node stream, in document order.
        statements: The statement overlay.
        source: The stable book identity.

    Returns:
        Its ``{label, uuid}``, or None.
    """
    elements = _chain_elements(nodes, statements, source)
    return elements[0] if elements else None


async def persist_chain(
    nodes: list[models.ASTNode],
    statements: list[models.Statement],
    source: str,
) -> None:
    """Write the merged ``:NEXT`` chain and ``:HEAD`` edge.

    The chain mixes non-absorbed prose ``:Node`` vertices with
    ``:Statement`` vertices in document order. Absorbed raw nodes are
    skipped — they still exist in the graph for provenance but are not
    in the walkable chain.

    Both tiers are matched BY LABEL. Cypher cannot parameterise a label, so
    the pairs are bucketed by their ``(from_label, to_label)`` combination —
    at most four — and each bucket is one query with its labels written in. A
    label-free ``MATCH (a {uuid: …})`` would scan every vertex in the database
    and, worse, would silently do the wrong thing the moment two tiers shared
    a uuid.
    """
    if not nodes:
        return
    pairs = _merged_chain(nodes, statements, source)
    head = _merged_head(nodes, statements, source)
    source_key = source_uuid(source)

    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for pair in pairs:
        buckets[(pair['from_label'], pair['to_label'])].append(pair)

    async with driver().session(database=database()) as session:
        if head:
            head_label = head['label']
            await session.run(
                f'MATCH (s:{SOURCE_LABEL} {{uuid: $source}}), '
                f'(n:{head_label} {{uuid: $head}}) '
                f'MERGE (s)-[:HEAD]->(n)',
                source=source_key,
                head=head['uuid'],
            )
        for (from_label, to_label), rows in buckets.items():
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (a:{from_label} {{uuid: pair.from}}), '
                f'(b:{to_label} {{uuid: pair.to}}) '
                f'MERGE (a)-[:NEXT]->(b)',
                pairs=rows,
            )


def statement_rows(
    statements: list[models.Statement], source: str
) -> list[dict]:
    """Every statement's property map, one flat list."""
    return [statement_properties(statement, source) for statement in statements]


async def persist_statements(
    statements: list[models.Statement], source: str
) -> None:
    """Upsert the book's ``:Statement`` overlay, rooted under the
    already-persisted ``:Source`` via ``:HAS_STATEMENT``."""
    if not statements:
        return
    rows = statement_rows(statements, source)

    async with driver().session(database=database()) as session:
        await session.run(
            f'UNWIND $rows AS row '
            f'MERGE (s:{STATEMENT_LABEL} {{uuid: row.uuid}}) '
            f'SET s += row',
            rows=rows,
        )


async def persist_procedures(
    statements: list[models.Statement], source: str
) -> None:
    """Upsert the procedural layer: one ``:Procedure`` per derivation hung
    off its statement via ``:HAS_PROCEDURE``."""
    procedure_batch = procedure_rows(statements, source)
    if not procedure_batch:
        return
    acts = act_rows(statements, source)
    owners = has_procedure_pairs(statements, source)
    firsts = first_pairs(statements, source)
    thens = then_pairs(statements, source)

    async with driver().session(database=database()) as session:
        await session.run(
            f'UNWIND $rows AS row '
            f'MERGE (p:{PROCEDURE_LABEL} {{uuid: row.uuid}}) '
            f'SET p += row',
            rows=procedure_batch,
        )
        if acts:
            await session.run(
                f'UNWIND $rows AS row '
                f'MERGE (a:{ACT_LABEL} {{uuid: row.uuid}}) '
                f'SET a += row',
                rows=acts,
            )
        await session.run(
            f'UNWIND $pairs AS pair '
            f'MATCH (s:{STATEMENT_LABEL} {{uuid: pair.statement}}), '
            f'(p:{PROCEDURE_LABEL} {{uuid: pair.procedure}}) '
            f'MERGE (s)-[:HAS_PROCEDURE]->(p)',
            pairs=owners,
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
                f'MATCH (a:{ACT_LABEL} {{uuid: pair.from}}), '
                f'(b:{ACT_LABEL} {{uuid: pair.to}}) '
                f'MERGE (a)-[:THEN]->(b)',
                pairs=thens,
            )
