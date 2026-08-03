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

``persist_variables`` writes ``:Variable`` vertices hung off their
provenance ``:Node`` via ``:HAS_VARIABLE`` — statement and procedure hubs
inherit variables through ``:MEMBER_OF``.

Writes are batched: structural node labels are grouped by their per-type
label and each batch is one MERGE. Statements, procedures and acts each
carry a single fixed label, so each is one batched MERGE.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from kms.core import models
from kms.graph.facts import (
    FACT_LABEL,
    evidence_pairs,
    fact_rows,
)
from kms.graph.instructions import (
    INSTRUCTION_LABEL,
    governs_pairs,
    instruction_rows,
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
    *,
    session_factory: Callable,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Upsert the book's ``:Source`` root and its ``:Node`` vertices.

    Vertices only — no ``:NEXT`` or ``:HEAD`` edges (those are written
    by ``persist_chain`` in document order).

    Args:
        nodes: The flat node stream, in document order.
        source: The stable book identity.
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
        metadata: Optional ``{title, author}`` for the ``:Source`` node.
    """
    if not nodes:
        return
    source_props = source_properties(source, metadata)
    batches = node_batches(nodes, source)

    async with session_factory() as session:
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
    *,
    session_factory: Callable,
) -> None:
    """Write the pure provenance ``:NEXT`` chain and ``:HEAD`` edge.

    ``:HEAD`` runs from ``:Source`` to the first ``:Node``, then ``:NEXT``
    threads every node in document order. Nothing is skipped and no statement
    is slotted in: the chain is the verbatim stream, and the statement
    overlay hangs off it via ``(:Node)-[:MEMBER_OF]->(:Statement)``
    (see ``persist_statements``).

    Args:
        nodes: The flat node stream, in document order.
        source: The stable book identity.
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
    """
    if not nodes:
        return
    chain = _chain_nodes(nodes, source)
    if not chain:
        return
    pairs = _chain_pairs(chain)
    head = chain[0]

    async with session_factory() as session:
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
    statements: list[models.Statement],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    """Upsert the book's ``:Statement`` overlay as bare vertices, plus the
    ``:MEMBER_OF`` edges from each member node.

    Statements are deliberately out of the chain — the walkable ``:NEXT``
    spine is the pure provenance node stream — and each one points at the raw
    blocks that are its members: one ``(:Node)-[:MEMBER_OF]->(:Statement)``
    edge per member of the group. Book-scoped lookup goes through the
    ``statement_source`` index rather than a traversal.

    Args:
        statements: The statement hubs.
        source: The stable book identity.
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
    """
    if not statements:
        return
    rows = statement_rows(statements, source)
    pairs = statement_member_pairs(statements, source)

    async with session_factory() as session:
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


async def persist_instructions(
    instructions: list[models.Instruction],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    """Upsert the ``:Instruction`` hubs and their ``:GOVERNS`` edges.

    One hub per lead-in, carrying the page's own sentence, pointing at each
    exercise node it governs. The edge runs from the hub outward: governance
    is a claim the instruction makes about those nodes, not a grouping they
    belong to, so an exercise keeps its statement membership untouched.

    Args:
        instructions: The instruction hubs.
        source: The stable book identity.
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
    """
    if not instructions:
        return
    rows = instruction_rows(instructions, source)
    pairs = governs_pairs(instructions, source)

    async with session_factory() as session:
        await session.run(
            f'UNWIND $rows AS row '
            f'MERGE (i:{INSTRUCTION_LABEL} {{uuid: row.uuid}}) '
            f'SET i += row',
            rows=rows,
        )
        if pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (i:{INSTRUCTION_LABEL} {{uuid: pair.instruction}}), '
                f'(n:{NODE_LABEL} {{uuid: pair.node}}) '
                f'MERGE (i)-[:GOVERNS]->(n)',
                pairs=pairs,
            )


async def persist_procedures(
    procedures: list[models.Procedure],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    """Upsert the procedural layer: one ``:Procedure`` hub per derivation,
    pointing at its member nodes via ``:MEMBER_OF``.

    Args:
        procedures: The procedure hubs.
        source: The stable book identity.
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
    """
    procedure_batch = procedure_rows(procedures, source)
    if not procedure_batch:
        return
    acts = act_rows(procedures, source)
    members = procedure_member_pairs(procedures, source)
    firsts = first_pairs(procedures, source)
    thens = then_pairs(procedures, source)

    async with session_factory() as session:
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
    variables: list[tuple[int, list[models.Variable]]],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    """Upsert the ``:Variable`` nodes and ``:HAS_VARIABLE`` edges.

    One ``:Variable`` per binding, each hanging off the ``:Node`` it was
    extracted from via ``:HAS_VARIABLE`` — a single label, no bucketing
    needed.

    Args:
        variables: The ``(node_id, [Variable])`` extraction results.
        source: The stable book identity.
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
    """
    variable_batch = variable_rows(variables, source)
    if not variable_batch:
        return
    pairs = has_variable_pairs(variables, source)

    async with session_factory() as session:
        await session.run(
            f'UNWIND $rows AS row '
            f'MERGE (v:{VARIABLE_LABEL} {{uuid: row.uuid}}) '
            f'SET v += row',
            rows=variable_batch,
        )
        if pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (n:{NODE_LABEL} {{uuid: pair.container}}), '
                f'(v:{VARIABLE_LABEL} {{uuid: pair.variable}}) '
                f'MERGE (n)-[:HAS_VARIABLE]->(v)',
                pairs=pairs,
            )


async def persist_facts(
    facts: list[models.AtomicFact],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    """Upsert the ``:Fact`` nodes and their ``:EVIDENCE_FOR`` edges.

    One ``:Fact`` per atomic fact, carrying its text and — when computed —
    its embedding vector. Each provenance node the fact draws on points at
    it via ``:EVIDENCE_FOR``, the same raw-material → construct anchor the
    statement and procedure tiers use.

    Args:
        facts: The atomic facts, in document order, with embeddings filled.
        source: The stable book identity.
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
    """
    fact_batch = fact_rows(facts, source)
    if not fact_batch:
        return
    pairs = evidence_pairs(facts, source)

    async with session_factory() as session:
        await session.run(
            f'UNWIND $rows AS row '
            f'MERGE (f:{FACT_LABEL} {{uuid: row.uuid}}) '
            f'SET f += row',
            rows=fact_batch,
        )
        if pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (n:{NODE_LABEL} {{uuid: pair.node}}), '
                f'(f:{FACT_LABEL} {{uuid: pair.fact}}) '
                f'MERGE (n)-[:EVIDENCE_FOR]->(f)',
                pairs=pairs,
            )
