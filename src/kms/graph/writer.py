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

Every vertex and edge carries ``created_at`` and ``modified_at`` — ISO-8601
UTC stamps, set once when the element is first written and bumped on every
re-write. These are transaction-time bookkeeping (when we wrote the graph),
deliberately separate from any semantic time in the content.

Writes are batched: structural node labels are grouped by their per-type
label and each batch is one MERGE. Statements, procedures and acts each
carry a single fixed label, so each is one batched MERGE.
"""

from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
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


def utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string, for ``created_at`` stamps.

    Returns:
        The current time, e.g. ``2026-08-04T03:00:00+00:00``.
    """
    return datetime.now(UTC).isoformat(timespec='seconds')


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
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(
            f'MERGE (s:{SOURCE_LABEL} {{uuid: $uuid}}) '
            f'ON CREATE SET s.created_at = $now '
            f'SET s += $props, s.modified_at = $now',
            uuid=source_props['uuid'],
            props=source_props,
            now=now,
        )
        for label, rows in batches.items():
            query = (
                f'UNWIND $rows AS row '
                f'MERGE (n:{NODE_LABEL} {{uuid: row.uuid}}) '
                f'ON CREATE SET n.created_at = $now '
                f'SET n += row, n.modified_at = $now'
            )
            if label:
                query += f' SET n:{label}'
            await session.run(query, rows=rows, now=now)


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
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(
            f'MATCH (s:{SOURCE_LABEL} {{uuid: $source}}), '
            f'(n:{NODE_LABEL} {{uuid: $head}}) '
            f'MERGE (s)-[r:HEAD]->(n) '
            f'ON CREATE SET r.created_at = $now '
            f'SET r.modified_at = $now',
            source=source_uuid(source),
            head=head,
            now=now,
        )
        if pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (a:{NODE_LABEL} {{uuid: pair.from}}), '
                f'(b:{NODE_LABEL} {{uuid: pair.to}}) '
                f'MERGE (a)-[r:NEXT]->(b) '
                f'ON CREATE SET r.created_at = $now '
                f'SET r.modified_at = $now',
                pairs=pairs,
                now=now,
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
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(
            f'UNWIND $rows AS row '
            f'MERGE (s:{STATEMENT_LABEL} {{uuid: row.uuid}}) '
            f'ON CREATE SET s.created_at = $now '
            f'SET s += row, s.modified_at = $now',
            rows=rows,
            now=now,
        )
        if pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (n:{NODE_LABEL} {{uuid: pair.node}}), '
                f'(s:{STATEMENT_LABEL} {{uuid: pair.statement}}) '
                f'MERGE (n)-[r:MEMBER_OF]->(s) '
                f'ON CREATE SET r.created_at = $now '
                f'SET r.modified_at = $now',
                pairs=pairs,
                now=now,
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
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(
            f'UNWIND $rows AS row '
            f'MERGE (i:{INSTRUCTION_LABEL} {{uuid: row.uuid}}) '
            f'ON CREATE SET i.created_at = $now '
            f'SET i += row, i.modified_at = $now',
            rows=rows,
            now=now,
        )
        if pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (i:{INSTRUCTION_LABEL} {{uuid: pair.instruction}}), '
                f'(n:{NODE_LABEL} {{uuid: pair.node}}) '
                f'MERGE (i)-[r:GOVERNS]->(n) '
                f'ON CREATE SET r.created_at = $now '
                f'SET r.modified_at = $now',
                pairs=pairs,
                now=now,
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
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(
            f'UNWIND $rows AS row '
            f'MERGE (p:{PROCEDURE_LABEL} {{uuid: row.uuid}}) '
            f'ON CREATE SET p.created_at = $now '
            f'SET p += row, p.modified_at = $now',
            rows=procedure_batch,
            now=now,
        )
        if acts:
            await session.run(
                f'UNWIND $rows AS row '
                f'MERGE (a:{ACT_LABEL} {{uuid: row.uuid}}) '
                f'ON CREATE SET a.created_at = $now '
                f'SET a += row, a.modified_at = $now',
                rows=acts,
                now=now,
            )
        if members:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (n:{NODE_LABEL} {{uuid: pair.node}}), '
                f'(p:{PROCEDURE_LABEL} {{uuid: pair.procedure}}) '
                f'MERGE (n)-[r:MEMBER_OF]->(p) '
                f'ON CREATE SET r.created_at = $now '
                f'SET r.modified_at = $now',
                pairs=members,
                now=now,
            )
        if firsts:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (p:{PROCEDURE_LABEL} {{uuid: pair.procedure}}), '
                f'(a:{ACT_LABEL} {{uuid: pair.act}}) '
                f'MERGE (p)-[r:FIRST]->(a) '
                f'ON CREATE SET r.created_at = $now '
                f'SET r.modified_at = $now',
                pairs=firsts,
                now=now,
            )
        if thens:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (a:{ACT_LABEL} {{uuid: pair.from}}), '
                f'(b:{ACT_LABEL} {{uuid: pair.to}}) '
                f'MERGE (a)-[r:THEN]->(b) '
                f'ON CREATE SET r.created_at = $now '
                f'SET r.modified_at = $now',
                pairs=thens,
                now=now,
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
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(
            f'UNWIND $rows AS row '
            f'MERGE (f:{FACT_LABEL} {{uuid: row.uuid}}) '
            f'ON CREATE SET f.created_at = $now '
            f'SET f += row, f.modified_at = $now',
            rows=fact_batch,
            now=now,
        )
        if pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (n:{NODE_LABEL} {{uuid: pair.node}}), '
                f'(f:{FACT_LABEL} {{uuid: pair.fact}}) '
                f'MERGE (n)-[r:EVIDENCE_FOR]->(f) '
                f'ON CREATE SET r.created_at = $now '
                f'SET r.modified_at = $now',
                pairs=pairs,
                now=now,
            )
