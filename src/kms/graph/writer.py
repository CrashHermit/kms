"""
Persist the structural node stream and the statement/procedure overlay into
Neo4j.

``persist_nodes`` upserts ``:Source`` and ``:Node`` vertices (no edges).

``persist_chain`` writes the pure provenance chain: ``:HEAD`` from
``:Source`` to the first ``:Node``, then ``:NEXT`` threading every node in
document order. Nothing is skipped and no statement is slotted in — the
chain is the verbatim stream.

``persist_statements`` writes the ``:Statement`` overlay as bare vertices,
plus ``:MEMBER_OF`` edges from each member node. No edge runs from
``:Source`` to them: a statement is reached from the raw blocks that informed
it, and scoped to its book by the indexed ``source`` property (see
``schema``). An edge per statement would duplicate that index and hang the
whole book off one supernode.

``persist_procedures`` writes ``:Procedure`` vertices and
``:MEMBER_OF`` edges from their member nodes. ``:Act`` step chains are
declared but not yet written.

``persist_equations`` writes ``:Equation`` vertices hung off their owning
``:Statement`` or ``:Procedure`` via ``:HAS_EQUATION`` — the semantic unit
reached from its member nodes — or off the plain ``:Node``, equally via
``:HAS_EQUATION``, when the source unit is outside any hub.

Writes are batched: structural node labels are grouped by their per-type
label and each batch is one MERGE. Statements, procedures and acts each
carry a single fixed label, so each is one batched MERGE.
"""

from collections import defaultdict
from typing import Any

from kms.core import models
from kms.graph.db import database, driver
from kms.graph.equations import (
    EQUATION_LABEL,
    equation_pairs,
    equation_rows,
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
from kms.graph.procedures import (
    ACT_LABEL,
    PROCEDURE_LABEL,
    act_rows,
    first_pairs,
    procedure_member_pairs,
    procedure_rows,
    then_pairs,
)
from kms.graph.statements import (
    STATEMENT_LABEL,
    statement_member_pairs,
    statement_properties,
)
from kms.graph.variables import (
    VARIABLE_LABEL,
    has_variable_pairs,
    variable_rows,
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
    by ``persist_chain`` in document order).
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


def _chain_nodes(nodes: list[models.ASTNode], source: str) -> list[str]:
    """Every node's uuid in document order — the pure provenance chain.

    The chain is the verbatim stream, nothing skipped: statements are not
    elements of it — they hang off their member nodes via
    ``:MEMBER_OF`` (see ``persist_statements``).

    Args:
        nodes: The flat node stream, in document order.
        source: The stable book identity.

    Returns:
        One uuid per node, in document order.
    """
    return [node_uuid(source, node.id) for node in nodes if node.id is not None]


def _chain_pairs(chain: list[str]) -> list[dict]:
    """The consecutive pairs of the provenance ``:NEXT`` chain.

    Args:
        chain: The ordered node uuids.

    Returns:
        One ``{from, to}`` per ``:NEXT`` edge.
    """
    return [
        {'from': current, 'to': following}
        for current, following in zip(chain, chain[1:], strict=False)
    ]


async def persist_chain(
    nodes: list[models.ASTNode],
    source: str,
) -> None:
    """Write the pure provenance ``:NEXT`` chain and ``:HEAD`` edge.

    ``:HEAD`` runs from ``:Source`` to the first ``:Node``, then ``:NEXT``
    threads every node in document order. Nothing is skipped and no statement
    is slotted in: the chain is the verbatim stream, and the statement
    overlay hangs off it via ``(:Node)-[:MEMBER_OF]->(:Statement)``
    (see ``persist_statements``).
    """
    if not nodes:
        return
    chain = _chain_nodes(nodes, source)
    if not chain:
        return
    pairs = _chain_pairs(chain)
    head = chain[0]

    async with driver().session(database=database()) as session:
        await session.run(
            f'MATCH (s:{SOURCE_LABEL} {{uuid: $source}}), '
            f'(n:{NODE_LABEL} {{uuid: $head}}) '
            f'MERGE (s)-[:HEAD]->(n)',
            source=source_uuid(source),
            head=head,
        )
        if pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (a:{NODE_LABEL} {{uuid: pair.from}}), '
                f'(b:{NODE_LABEL} {{uuid: pair.to}}) '
                f'MERGE (a)-[:NEXT]->(b)',
                pairs=pairs,
            )


def statement_rows(
    statements: list[models.Statement], source: str
) -> list[dict]:
    """Every statement's property map, one flat list."""
    return [statement_properties(statement, source) for statement in statements]


async def persist_statements(
    statements: list[models.Statement], source: str
) -> None:
    """Upsert the book's ``:Statement`` overlay as bare vertices, plus the
    ``:MEMBER_OF`` edges from each member node.

    Statements are deliberately out of the chain — the walkable ``:NEXT``
    spine is the pure provenance node stream — and each one points at the raw
    blocks that are its members: one ``(:Node)-[:MEMBER_OF]->(:Statement)``
    edge per member of the group. Book-scoped lookup goes through the
    ``statement_source`` index rather than a traversal.
    """
    if not statements:
        return
    rows = statement_rows(statements, source)
    pairs = statement_member_pairs(statements, source)

    async with driver().session(database=database()) as session:
        await session.run(
            f'UNWIND $rows AS row '
            f'MERGE (s:{STATEMENT_LABEL} {{uuid: row.uuid}}) '
            f'SET s += row',
            rows=rows,
        )
        if pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (n:{NODE_LABEL} {{uuid: pair.node}}), '
                f'(s:{STATEMENT_LABEL} {{uuid: pair.statement}}) '
                f'MERGE (n)-[:MEMBER_OF]->(s)',
                pairs=pairs,
            )


async def persist_procedures(
    procedures: list[models.Procedure],
    source: str,
) -> None:
    """Upsert the procedural layer: one ``:Procedure`` hub per derivation,
    pointing at its member nodes via ``:MEMBER_OF``."""
    procedure_batch = procedure_rows(procedures, source)
    if not procedure_batch:
        return
    acts = act_rows(procedures, source)
    members = procedure_member_pairs(procedures, source)
    firsts = first_pairs(procedures, source)
    thens = then_pairs(procedures, source)

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
        if members:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (n:{NODE_LABEL} {{uuid: pair.node}}), '
                f'(p:{PROCEDURE_LABEL} {{uuid: pair.procedure}}) '
                f'MERGE (n)-[:MEMBER_OF]->(p)',
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
                f'MATCH (a:{ACT_LABEL} {{uuid: pair.from}}), '
                f'(b:{ACT_LABEL} {{uuid: pair.to}}) '
                f'MERGE (a)-[:THEN]->(b)',
                pairs=thens,
            )


async def persist_variables(
    variables: list[tuple[str, list[int], list[models.Variable]]],
    procedures: list[models.Procedure],
    source: str,
) -> None:
    """Upsert the ``:Variable`` nodes and ``:HAS_VARIABLE`` edges.

    One ``:Variable`` per binding. A variable hangs off the unit it was
    extracted from: the ``:Equation`` when ``equation_index`` is set, its
    owning ``:Statement`` or ``:Procedure`` hub, or the plain ``:Node``
    otherwise — mirroring how equations resolve their container. Cypher
    cannot parameterise a label, so the pairs are split by
    ``container_label`` and each bucket is one query with its label written
    in.
    """
    variable_batch = variable_rows(variables, source)
    if not variable_batch:
        return
    pairs = has_variable_pairs(variables, procedures, source)
    equation_pairs = [p for p in pairs if p['container_label'] == 'equation']
    statement_pairs = [p for p in pairs if p['container_label'] == 'statement']
    procedure_pairs = [p for p in pairs if p['container_label'] == 'procedure']
    node_pairs = [p for p in pairs if p['container_label'] == 'node']

    async with driver().session(database=database()) as session:
        await session.run(
            f'UNWIND $rows AS row '
            f'MERGE (v:{VARIABLE_LABEL} {{uuid: row.uuid}}) '
            f'SET v += row',
            rows=variable_batch,
        )
        if equation_pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (e:{EQUATION_LABEL} {{uuid: pair.container}}), '
                f'(v:{VARIABLE_LABEL} {{uuid: pair.variable}}) '
                f'MERGE (e)-[:HAS_VARIABLE]->(v)',
                pairs=equation_pairs,
            )
        if statement_pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (s:{STATEMENT_LABEL} {{uuid: pair.container}}), '
                f'(v:{VARIABLE_LABEL} {{uuid: pair.variable}}) '
                f'MERGE (s)-[:HAS_VARIABLE]->(v)',
                pairs=statement_pairs,
            )
        if procedure_pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (p:{PROCEDURE_LABEL} {{uuid: pair.container}}), '
                f'(v:{VARIABLE_LABEL} {{uuid: pair.variable}}) '
                f'MERGE (p)-[:HAS_VARIABLE]->(v)',
                pairs=procedure_pairs,
            )
        if node_pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (n:{NODE_LABEL} {{uuid: pair.container}}), '
                f'(v:{VARIABLE_LABEL} {{uuid: pair.variable}}) '
                f'MERGE (n)-[:HAS_VARIABLE]->(v)',
                pairs=node_pairs,
            )


async def persist_equations(
    equations: list[tuple[str, list[int], list[models.Equation]]],
    procedures: list[models.Procedure],
    source: str,
) -> None:
    """Upsert the ``:Equation`` nodes and their attachment edges.

    One ``:Equation`` per extracted equation. An equation hangs off the unit
    it was extracted from: its owning ``:Statement``, ``:Procedure`` or plain
    ``:Node`` — every unit is an equally valid extraction source — via
    ``:HAS_EQUATION``. Cypher cannot parameterise a label, so the pairs are
    split by ``container_label`` and each bucket is one query with its label
    written in.
    """
    equation_batch = equation_rows(equations, source)
    if not equation_batch:
        return
    pairs = equation_pairs(equations, procedures, source)
    statement_pairs = [p for p in pairs if p['container_label'] == 'statement']
    procedure_pairs = [p for p in pairs if p['container_label'] == 'procedure']
    node_pairs = [p for p in pairs if p['container_label'] == 'node']

    async with driver().session(database=database()) as session:
        await session.run(
            f'UNWIND $rows AS row '
            f'MERGE (e:{EQUATION_LABEL} {{uuid: row.uuid}}) '
            f'SET e += row',
            rows=equation_batch,
        )
        if statement_pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (s:{STATEMENT_LABEL} {{uuid: pair.container}}), '
                f'(e:{EQUATION_LABEL} {{uuid: pair.equation}}) '
                f'MERGE (s)-[:HAS_EQUATION]->(e)',
                pairs=statement_pairs,
            )
        if procedure_pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (p:{PROCEDURE_LABEL} {{uuid: pair.container}}), '
                f'(e:{EQUATION_LABEL} {{uuid: pair.equation}}) '
                f'MERGE (p)-[:HAS_EQUATION]->(e)',
                pairs=procedure_pairs,
            )
        if node_pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (n:{NODE_LABEL} {{uuid: pair.container}}), '
                f'(e:{EQUATION_LABEL} {{uuid: pair.equation}}) '
                f'MERGE (n)-[:HAS_EQUATION]->(e)',
                pairs=node_pairs,
            )
