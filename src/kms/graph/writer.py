from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from kms.core import models
from kms.graph import queries
from kms.graph.community import (
    community_rows,
)
from kms.graph.community import (
    evidence_pairs as community_evidence_pairs,
)
from kms.graph.community import (
    member_pairs as community_member_pairs,
)
from kms.graph.definitions import (
    definition_rows,
    has_definition_pairs,
)
from kms.graph.entities import (
    entity_rows,
)
from kms.graph.entity_hubs import (
    canonical_entity_pairs,
    entity_hub_rows,
)
from kms.graph.fact_hubs import (
    fact_hub_rows,
    has_fact_pairs,
)
from kms.graph.facts import (
    evidence_pairs,
    fact_rows,
)
from kms.graph.instructions import (
    governs_pairs,
    instruction_rows,
)
from kms.graph.nodes import (
    node_label,
    node_properties,
    node_uuid,
    source_properties,
    source_uuid,
)
from kms.graph.predicate_hubs import (
    canonical_predicate_pairs,
    predicate_hub_rows,
)
from kms.graph.predicates import (
    has_predicate_pairs,
    predicate_rows,
)
from kms.graph.procedures import (
    act_rows,
    first_pairs,
    procedure_member_pairs,
    procedure_rows,
    then_pairs,
)
from kms.graph.statements import (
    statement_member_pairs,
    statement_properties,
)
from kms.graph.triplet_hubs import (
    canonical_object_pairs,
    canonical_subject_pairs,
    supported_by_pairs,
)
from kms.graph.triplet_hubs import (
    canonical_predicate_pairs as th_canonical_predicate_pairs,
)
from kms.graph.triplets import (
    has_object_pairs,
    has_subject_pairs,
    triplet_rows,
    yields_pairs,
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


def statement_rows(
    statements: list[models.Statement], source: str
) -> list[dict]:
    return [statement_properties(statement, source) for statement in statements]


async def persist_statements(
    statements: list[models.Statement],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    if not statements:
        return
    rows = statement_rows(statements, source)
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
    pairs = governs_pairs(instructions, source)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(queries.MERGE_INSTRUCTIONS, rows=rows, now=now)
        if pairs:
            await session.run(queries.MERGE_GOVERNS, pairs=pairs, now=now)


async def persist_procedures(
    procedures: list[models.Procedure],
    source: str,
    *,
    session_factory: Callable,
) -> None:
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
            queries.MERGE_PROCEDURES, rows=procedure_batch, now=now
        )
        if acts:
            await session.run(queries.MERGE_ACTS, rows=acts, now=now)
        if members:
            await session.run(
                queries.MERGE_PROCEDURE_MEMBERS, pairs=members, now=now
            )
        if firsts:
            await session.run(queries.MERGE_FIRST, pairs=firsts, now=now)
        if thens:
            await session.run(queries.MERGE_THEN, pairs=thens, now=now)


async def persist_facts(
    facts: list[models.AtomicFact],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    fact_batch = fact_rows(facts, source)
    if not fact_batch:
        return
    pairs = evidence_pairs(facts, source)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(queries.MERGE_FACTS, rows=fact_batch, now=now)
        if pairs:
            await session.run(queries.MERGE_EVIDENCE, pairs=pairs, now=now)


async def persist_entities(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    node_entity_descriptions: dict[int, list[dict]],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    entity_batch = entity_rows(
        triplets, facts, source, node_entity_descriptions
    )
    if not entity_batch:
        return
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(queries.MERGE_ENTITIES, rows=entity_batch, now=now)


async def persist_triplets(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    triplet_batch = triplet_rows(triplets, facts, source)
    if not triplet_batch:
        return
    yields = yields_pairs(triplets, facts, source)
    subjects = has_subject_pairs(triplets, facts, source)
    objects = has_object_pairs(triplets, facts, source)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(queries.MERGE_TRIPLETS, rows=triplet_batch, now=now)
        if yields:
            await session.run(queries.MERGE_YIELDS, pairs=yields, now=now)
        if subjects:
            await session.run(
                queries.MERGE_HAS_SUBJECT, pairs=subjects, now=now
            )
        if objects:
            await session.run(queries.MERGE_HAS_OBJECT, pairs=objects, now=now)


async def persist_predicates(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
    *,
    session_factory: Callable,
    node_predicate_descriptions: dict[int, list[dict]] | None = None,
) -> None:
    node_predicate_descriptions = node_predicate_descriptions or {}
    predicate_batch = predicate_rows(
        triplets, facts, source, node_predicate_descriptions
    )
    if not predicate_batch:
        return
    pairs = has_predicate_pairs(triplets, facts, source)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(
            queries.MERGE_PREDICATES, rows=predicate_batch, now=now
        )
        if pairs:
            await session.run(queries.MERGE_HAS_PREDICATE, pairs=pairs, now=now)


async def persist_entity_hubs(
    entity_clusters: list[list[dict]],
    hub_definitions: list[dict],
    source: str,
    *,
    session_factory: Callable,
    definitions: list[dict] | None = None,
) -> None:
    hub_batch = entity_hub_rows(
        entity_clusters, source, definitions=definitions
    )
    if not hub_batch:
        return
    definition_batch = definition_rows(hub_definitions)
    canonical_pairs = canonical_entity_pairs(entity_clusters, source)
    has_definition_pairs_list = has_definition_pairs(hub_definitions)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(
            queries.MERGE_ENTITY_HUBS, rows=hub_batch, now=now
        )
        if definition_batch:
            await session.run(
                queries.MERGE_DEFINITIONS,
                rows=definition_batch,
                now=now,
            )
        if canonical_pairs:
            await session.run(
                queries.MERGE_CANONICAL_ENTITY,
                pairs=canonical_pairs,
                now=now,
            )
        if has_definition_pairs_list:
            await session.run(
                queries.MERGE_HAS_DEFINITION,
                pairs=has_definition_pairs_list,
                now=now,
            )


async def persist_predicate_hubs(
    predicate_clusters: list[list[dict]],
    hub_definitions: list[dict],
    source: str,
    *,
    session_factory: Callable,
    definitions: list[dict] | None = None,
) -> None:
    hub_batch = predicate_hub_rows(
        predicate_clusters, source, definitions=definitions
    )
    if not hub_batch:
        return
    definition_batch = definition_rows(hub_definitions)
    canonical_pairs = canonical_predicate_pairs(
        predicate_clusters, source
    )
    has_definition_pairs_list = has_definition_pairs(hub_definitions)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(
            queries.MERGE_PREDICATE_HUBS, rows=hub_batch, now=now
        )
        if definition_batch:
            await session.run(
                queries.MERGE_DEFINITIONS,
                rows=definition_batch,
                now=now,
            )
        if canonical_pairs:
            await session.run(
                queries.MERGE_CANONICAL_PREDICATE,
                pairs=canonical_pairs,
                now=now,
            )
        if has_definition_pairs_list:
            await session.run(
                queries.MERGE_HAS_DEFINITION,
                pairs=has_definition_pairs_list,
                now=now,
            )


async def persist_communities(
    communities: list[dict],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    if not communities:
        return
    rows = community_rows(communities, source)
    member_pairs_list = community_member_pairs(communities)
    evidence_pairs_list = community_evidence_pairs(communities)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(
            queries.MERGE_COMMUNITIES, rows=rows, now=now
        )
        if member_pairs_list:
            await session.run(
                queries.MERGE_COMMUNITY_MEMBERS,
                pairs=member_pairs_list,
                now=now,
            )
        if evidence_pairs_list:
            await session.run(
                queries.MERGE_COMMUNITY_EVIDENCE,
                pairs=evidence_pairs_list,
                now=now,
            )


async def persist_triplet_hubs(
    groups: list[dict],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    if not groups:
        return
    from kms.graph.triplet_hubs import triplet_hub_properties

    hub_rows = [
        triplet_hub_properties(
            source, g['subj_hub'], g['pred_hub'], g['obj_hub']
        )
        for g in groups
    ]
    fact_rows_list = fact_hub_rows(groups)
    subj_pairs = canonical_subject_pairs(groups)
    pred_pairs = th_canonical_predicate_pairs(groups)
    obj_pairs = canonical_object_pairs(groups)
    fact_pairs = has_fact_pairs(groups)
    support_pairs = supported_by_pairs(groups)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(
            queries.MERGE_TRIPLET_HUBS, rows=hub_rows, now=now
        )
        if fact_rows_list:
            await session.run(
                queries.MERGE_FACT_HUBS,
                rows=fact_rows_list,
                now=now,
            )
        if subj_pairs:
            await session.run(
                queries.MERGE_CANONICAL_SUBJECT,
                pairs=subj_pairs,
                now=now,
            )
        if pred_pairs:
            await session.run(
                queries.MERGE_CANONICAL_PREDICATE,
                pairs=pred_pairs,
                now=now,
            )
        if obj_pairs:
            await session.run(
                queries.MERGE_CANONICAL_OBJECT,
                pairs=obj_pairs,
                now=now,
            )
        if fact_pairs:
            await session.run(
                queries.MERGE_HAS_FACT,
                pairs=fact_pairs,
                now=now,
            )
        if support_pairs:
            await session.run(
                queries.MERGE_SUPPORTED_BY,
                pairs=support_pairs,
                now=now,
            )

