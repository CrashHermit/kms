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


def _merged_chain(
    nodes: list[models.ASTNode],
    statements: list[models.StatementNode],
    source: str,
) -> list[dict]:
    """Build the ``{from, to}`` uuid pairs for the merged ``:NEXT`` chain.

    Walks the raw node stream in document order, skipping nodes absorbed
    into statements, and slotting ``:Statement`` nodes in their place.
    Returns uuid pairs for consecutive elements in the merged chain.
    """
    # Which node ids are absorbed into a statement.
    absorbed: set[int] = set()
    for statement in statements:
        for member_id in statement.statement_of or []:
            absorbed.add(member_id)

    # Map statement first-node id -> statement entity (for its graph uuid).
    statement_by_first_node: dict[int, models.StatementNode] = {}
    for statement in statements:
        first = (statement.statement_of or [None])[0]
        if first is not None:
            statement_by_first_node[first] = statement

    # Walk nodes, emit chain elements.
    chain: list[dict] = []
    for node in nodes:
        node_id = node.id
        if node_id is None:
            continue
        if node_id in statement_by_first_node:
            chain.append(
                {
                    'kind': 'statement',
                    'uuid': statement_uuid(
                        source, statement_by_first_node[node_id].id
                    ),
                }
            )
            continue
        if node_id in absorbed:
            continue
        chain.append(
            {
                'kind': 'node',
                'uuid': node_uuid(source, node_id),
            }
        )

    return [
        {'from': current['uuid'], 'to': following['uuid']}
        for current, following in zip(chain, chain[1:], strict=False)
    ]


def _merged_head(
    nodes: list[models.ASTNode],
    statements: list[models.StatementNode],
    source: str,
) -> str | None:
    """The uuid of the first element in the merged chain, or None if empty."""
    absorbed: set[int] = set()
    for statement in statements:
        for member_id in statement.statement_of or []:
            absorbed.add(member_id)
    statement_by_first_node: dict[int, models.StatementNode] = {}
    for statement in statements:
        first = (statement.statement_of or [None])[0]
        if first is not None:
            statement_by_first_node[first] = statement

    for node in nodes:
        node_id = node.id
        if node_id is None:
            continue
        if node_id in statement_by_first_node:
            return statement_uuid(source, statement_by_first_node[node_id].id)
        if node_id not in absorbed:
            return node_uuid(source, node_id)
    return None


async def persist_chain(
    nodes: list[models.ASTNode],
    statements: list[models.StatementNode],
    source: str,
) -> None:
    """Write the merged ``:NEXT`` chain and ``:HEAD`` edge.

    The chain mixes non-absorbed prose ``:Node`` vertices with
    ``:Statement`` vertices in document order. Absorbed raw nodes are
    skipped — they still exist in the graph for provenance but are not
    in the walkable chain.
    """
    if not nodes:
        return
    pairs = _merged_chain(nodes, statements, source)
    head = _merged_head(nodes, statements, source)
    source_key = source_uuid(source)

    async with driver().session(database=database()) as session:
        if head:
            await session.run(
                f'MATCH (s:{SOURCE_LABEL} {{uuid: $source}}), '
                f'(n {{uuid: $head}}) '
                f'MERGE (s)-[:HEAD]->(n)',
                source=source_key,
                head=head,
            )
        if pairs:
            await session.run(
                'UNWIND $pairs AS pair '
                'MATCH (a {uuid: pair.from}), '
                '(b {uuid: pair.to}) '
                'MERGE (a)-[:NEXT]->(b)',
                pairs=pairs,
            )


def statement_rows(
    statements: list[models.StatementNode], source: str
) -> list[dict]:
    """Every statement's property map, one flat list."""
    return [statement_properties(statement, source) for statement in statements]


async def persist_statements(
    statements: list[models.StatementNode], source: str
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
    statements: list[models.StatementNode], source: str
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
