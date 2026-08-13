from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from kms.core import models
from kms.graph import queries
from kms.graph.entity_hubs import (
    entity_hub_properties,
    entity_hub_uuid,
)
from kms.graph.instructions import (
    instruction_member_pairs,
    instruction_rows,
)
from kms.graph.nodes import (
    node_label,
    node_properties,
    node_uuid,
    source_properties,
    source_uuid,
)
from kms.graph.procedures import (
    first_pairs,
    procedure_member_pairs,
    procedure_rows,
    step_rows,
    then_pairs,
)
from kms.graph.statements import (
    has_procedure_pairs,
    statement_member_pairs,
    statement_properties,
)
from kms.graph.triplets import (
    evidence_pairs,
    triplet_rows,
)


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec='seconds')


def node_batches(
    nodes: list[models.ASTNode], source: str
) -> dict[str | None, list[dict]]:
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
    if not nodes:
        return
    source_props = source_properties(source, metadata)
    batches = node_batches(nodes, source)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(
            queries.MERGE_SOURCE,
            uuid=source_props['uuid'],
            props=source_props,
            now=now,
        )
        for label, rows in batches.items():
            await session.run(
                queries.merge_nodes_query(label), rows=rows, now=now
            )


def _chain_nodes(nodes: list[models.ASTNode], source: str) -> list[str]:
    return [node_uuid(source, node.id) for node in nodes if node.id is not None]


def _chain_pairs(chain: list[str]) -> list[dict]:
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
            queries.MERGE_HEAD,
            source=source_uuid(source),
            head=head,
            now=now,
        )
        if pairs:
            await session.run(queries.MERGE_NEXT, pairs=pairs, now=now)


def _statement_rows(
    statements: list[models.Statement], source: str
) -> list[dict]:
    return [statement_properties(s, source) for s in statements]


async def persist_statements(
    statements: list[models.Statement],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    if not statements:
        return
    rows = _statement_rows(statements, source)
    pairs = statement_member_pairs(statements, source)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(queries.MERGE_STATEMENTS, rows=rows, now=now)
        if pairs:
            await session.run(
                queries.MERGE_STATEMENT_MEMBERS, pairs=pairs, now=now
            )


async def persist_instructions(
    instructions: list[models.Instruction],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    if not instructions:
        return
    rows = instruction_rows(instructions, source)
    pairs = instruction_member_pairs(instructions, source)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(queries.MERGE_INSTRUCTIONS, rows=rows, now=now)
        if pairs:
            await session.run(
                queries.MERGE_INSTRUCTION_MEMBERS, pairs=pairs, now=now
            )


async def persist_procedures(
    procedures: list[models.Procedure],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    procedure_batch = procedure_rows(procedures, source)
    if not procedure_batch:
        return
    steps = step_rows(procedures, source)
    members = procedure_member_pairs(procedures, source)
    firsts = first_pairs(procedures, source)
    thens = then_pairs(procedures, source)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(
            queries.MERGE_PROCEDURES, rows=procedure_batch, now=now
        )
        if steps:
            await session.run(queries.MERGE_STEPS, rows=steps, now=now)
        if members:
            await session.run(
                queries.MERGE_PROCEDURE_MEMBERS, pairs=members, now=now
            )
        if firsts:
            await session.run(queries.MERGE_FIRST, pairs=firsts, now=now)
        if thens:
            await session.run(queries.MERGE_THEN, pairs=thens, now=now)


async def persist_statement_procedure_links(
    statements: list[models.Statement],
    procedures: list[models.Procedure],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    pairs = has_procedure_pairs(statements, procedures, source)
    if not pairs:
        return
    now = utcnow_iso()
    async with session_factory() as session:
        await session.run(queries.MERGE_HAS_PROCEDURE, pairs=pairs, now=now)


async def persist_triplets(
    triplets: list[models.Triplet],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    triplet_batch = triplet_rows(triplets, source)
    if not triplet_batch:
        return
    pairs = evidence_pairs(triplets, source)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(queries.MERGE_TRIPLETS, rows=triplet_batch, now=now)
        if pairs:
            await session.run(
                queries.MERGE_TRIPLET_EVIDENCE, pairs=pairs, now=now
            )


async def persist_entity_hubs(
    hubs: list[dict],
    *,
    session_factory: Callable,
) -> None:
    if not hubs:
        return
    hub_rows = [
        entity_hub_properties(
            source=h['source'],
            canonical_name=h['canonical_name'],
            description=h['description'],
            embedding=h.get('embedding'),
        )
        for h in hubs
    ]
    subject_pairs: list[dict] = []
    object_pairs: list[dict] = []
    for h in hubs:
        hub_uuid = entity_hub_uuid(h['source'], h['canonical_name'])
        for s in h.get('subject_spokes', []):
            subject_pairs.append(
                {'triplet': s['triplet_uuid'], 'hub': hub_uuid}
            )
        for s in h.get('object_spokes', []):
            object_pairs.append({'triplet': s['triplet_uuid'], 'hub': hub_uuid})
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(queries.MERGE_ENTITY_HUBS, rows=hub_rows, now=now)
        if subject_pairs:
            await session.run(
                queries.MERGE_HAS_SUBJECT, pairs=subject_pairs, now=now
            )
        if object_pairs:
            await session.run(
                queries.MERGE_HAS_OBJECT, pairs=object_pairs, now=now
            )
