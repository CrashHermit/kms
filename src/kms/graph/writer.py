"""Idempotent persistence of extraction results into Neo4j."""

from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from kms.core import models
from kms.graph import (
    assertions,
    learning,
    local_entity_hubs,
    local_event_hubs,
    local_predicate_hubs,
    local_procedure_hubs,
    local_statement_hubs,
    names,
    queries,
)
from kms.graph.hubs import triplet_hub_properties
from kms.graph.instructions import (
    instruction_governance_pairs,
    instruction_member_pairs,
    instruction_rows,
)
from kms.graph.nodes import (
    node_label,
    node_properties,
    source_properties,
    source_uuid,
    visual_asset_pairs,
    visual_asset_rows,
)
from kms.graph.procedures import (
    procedure_enrichment_properties,
    procedure_member_pairs,
    procedure_rows,
)
from kms.graph.statements import (
    has_procedure_pairs,
    statement_enrichment_properties,
    statement_member_pairs,
    statement_properties,
)
from kms.graph.triplets import (
    evidence_pairs,
    triplet_rows,
)


def utcnow_iso() -> str:
    """Returns the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat(timespec='seconds')


def node_batches(
    nodes: list[models.SourceNode],
    source: str,
    embeddings: list[list[float] | None] | None = None,
) -> dict[str | None, list[dict]]:
    """Groups node rows by label for batched merges."""
    if embeddings is not None and len(embeddings) != len(nodes):
        raise ValueError('node embeddings must match node count')
    batches: dict[str | None, list[dict]] = defaultdict(list)
    for index, node in enumerate(nodes):
        embedding = embeddings[index] if embeddings is not None else None
        batches[node_label(node)].append(
            node_properties(node, source, embedding=embedding)
        )
    return dict(batches)


async def persist_nodes(
    nodes: list[models.SourceNode],
    source: str,
    *,
    session_factory: Callable,
    metadata: dict[str, Any] | None = None,
    embeddings: list[list[float] | None] | None = None,
) -> None:
    """Persists the source vertex and all AST nodes."""
    if not nodes:
        return
    if any(node.uuid is None for node in nodes):
        raise ValueError('cannot persist nodes without stable ids')
    source_props = source_properties(source, metadata)
    batches = node_batches(nodes, source, embeddings)
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
        asset_rows = visual_asset_rows(nodes, source)
        if asset_rows:
            await session.run(
                queries.MERGE_VISUAL_ASSETS,
                rows=asset_rows,
                now=now,
            )
            await session.run(
                queries.MERGE_NODE_ASSETS,
                pairs=visual_asset_pairs(nodes, source),
                now=now,
            )


def _chain_nodes(nodes: list[models.SourceNode], source: str) -> list[str]:
    """Returns the uuid of every node in document order for the NEXT chain."""
    missing = [index for index, node in enumerate(nodes) if node.uuid is None]
    if missing:
        raise ValueError(f'nodes are missing uuids at positions {missing}')
    return [node.uuid for node in nodes]


def _chain_pairs(chain: list[str]) -> list[dict]:
    """Builds consecutive from/to pairs for the NEXT edges."""
    return [
        {'from': current, 'to': following}
        for current, following in zip(chain, chain[1:], strict=False)
    ]


async def persist_chain(
    nodes: list[models.SourceNode],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    """Persists the HEAD and NEXT edges linking nodes in document order.

    Args:
        nodes: The AST nodes in document order.
        source: The source key.
        session_factory: Async callable returning a Neo4j session.
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
    """Builds the row dicts for all statements in a source."""
    return [statement_properties(statement, source) for statement in statements]


async def persist_statements(
    statements: list[models.Statement],
    nodes: list[models.SourceNode],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    """Persists statement vertices and their MEMBER_OF edges.

    Args:
        statements: The statements to persist.
        nodes: The node stream for resolving member positions.
        source: The source key.
        session_factory: Async callable returning a Neo4j session.
    """
    if not statements:
        return
    if any(statement.uuid is None for statement in statements):
        raise ValueError('cannot persist statements without assigned uuids')
    rows = _statement_rows(statements, source)
    pairs = statement_member_pairs(statements, nodes, source)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(queries.MERGE_STATEMENTS, rows=rows, now=now)
        if pairs:
            await session.run(
                queries.MERGE_STATEMENT_MEMBERS, pairs=pairs, now=now
            )


async def _persist_hubs(
    hubs: list[dict],
    *,
    session_factory: Callable,
    graph_module,
    merge_query,
    membership_query,
) -> None:
    if not hubs:
        return
    rows = [
        graph_module.hub_properties(
            hub['source'],
            hub['canonical_name'],
            hub['description'],
            hub['embedding'],
            hub['members'],
        )
        for hub in hubs
    ]
    pairs = [
        {'base': member, 'hub': hub['uuid']}
        for hub in hubs
        for member in hub['members']
    ]
    async with session_factory() as session:
        await session.run(merge_query(), rows=rows, now=utcnow_iso())
        await session.run(membership_query(), pairs=pairs, now=utcnow_iso())


async def persist_statement_hubs(
    hubs: list[dict], *, session_factory: Callable
) -> None:
    await _persist_hubs(
        hubs,
        session_factory=session_factory,
        graph_module=local_statement_hubs,
        merge_query=queries.merge_statement_hubs_query,
        membership_query=queries.merge_statement_hub_memberships_query,
    )


async def persist_procedure_hubs(
    hubs: list[dict], *, session_factory: Callable
) -> None:
    await _persist_hubs(
        hubs,
        session_factory=session_factory,
        graph_module=local_procedure_hubs,
        merge_query=queries.merge_procedure_hubs_query,
        membership_query=queries.merge_procedure_hub_memberships_query,
    )


async def clear_statement_hubs(
    source: str, *, session_factory: Callable
) -> None:
    async with session_factory() as session:
        await session.run(queries.delete_statement_hubs_query(), source=source)


async def clear_procedure_hubs(
    source: str, *, session_factory: Callable
) -> None:
    async with session_factory() as session:
        await session.run(queries.delete_procedure_hubs_query(), source=source)


async def _persist_global_hubs(
    hubs: list[dict],
    *,
    session_factory: Callable,
    graph_module,
    all_source_query,
    merge_query,
    alignment_query,
) -> None:
    if not hubs:
        return
    source_by_hub = await all_source_query(session_factory)
    known_sources = {
        record['uuid']: record['source'] for record in source_by_hub
    }
    for hub in hubs:
        members = hub.get('members', [])
        sources = {
            known_sources[member]
            for member in members
            if member in known_sources
        }
        if len(sources) < 2:
            raise ValueError(
                'meta learning hubs require two distinct source supports: '
                f'{hub.get("uuid")}'
            )
    rows = [
        graph_module.global_hub_properties(
            hub['canonical_name'],
            hub['description'],
            hub['embedding'],
            hub['members'],
        )
        | {'uuid': hub['uuid']}
        for hub in hubs
    ]
    pairs = [
        {'source_hub': member, 'meta_hub': hub['uuid']}
        for hub in hubs
        for member in hub.get('members', [])
    ]
    async with session_factory() as session:
        await session.run(
            merge_query(),
            rows=rows,
            now=utcnow_iso(),
        )
        if pairs:
            await session.run(
                alignment_query(),
                pairs=pairs,
                now=utcnow_iso(),
            )


async def persist_global_statement_hubs(
    hubs: list[dict], *, session_factory: Callable
) -> None:
    await _persist_global_hubs(
        hubs,
        session_factory=session_factory,
        graph_module=local_statement_hubs,
        all_source_query=queries.all_source_statement_hubs,
        merge_query=queries.merge_global_statement_hubs_query,
        alignment_query=queries.merge_global_statement_alignments_query,
    )


async def persist_global_procedure_hubs(
    hubs: list[dict], *, session_factory: Callable
) -> None:
    await _persist_global_hubs(
        hubs,
        session_factory=session_factory,
        graph_module=local_procedure_hubs,
        all_source_query=queries.all_source_procedure_hubs,
        merge_query=queries.merge_global_procedure_hubs_query,
        alignment_query=queries.merge_global_procedure_alignments_query,
    )


async def clear_global_statement_hubs(*, session_factory: Callable) -> None:
    async with session_factory() as session:
        await session.run(queries.delete_global_statement_hubs_query())


async def clear_global_procedure_hubs(*, session_factory: Callable) -> None:
    async with session_factory() as session:
        await session.run(queries.delete_global_procedure_hubs_query())


async def persist_statement_enrichment(
    enrichments: list[dict],
    *,
    session_factory: Callable,
) -> None:
    """Persists compiled statement content and derived embeddings."""
    if not enrichments:
        return
    rows = [
        statement_enrichment_properties(
            enrichment['uuid'],
            enrichment.get('statement', enrichment['description']),
            enrichment['description'],
            enrichment['embedding'],
        )
        for enrichment in enrichments
    ]
    async with session_factory() as session:
        await session.run(
            queries.MERGE_STATEMENT_ENRICHMENT,
            rows=rows,
            now=utcnow_iso(),
        )


async def persist_procedure_enrichment(
    enrichments: list[dict],
    *,
    session_factory: Callable,
) -> None:
    """Persists compiled procedure content and derived embeddings."""
    if not enrichments:
        return
    rows = [
        procedure_enrichment_properties(
            enrichment['uuid'],
            enrichment['procedure'],
            enrichment['embedding'],
        )
        for enrichment in enrichments
    ]
    async with session_factory() as session:
        await session.run(
            queries.MERGE_PROCEDURE_ENRICHMENT,
            rows=rows,
            now=utcnow_iso(),
        )


async def persist_instructions(
    instructions: list[models.Instruction],
    nodes: list[models.SourceNode],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    """Persists instruction vertices and their MEMBER_OF edges.

    Args:
        instructions: The instructions to persist.
        nodes: The node stream for resolving member positions.
        source: The source key.
        session_factory: Async callable returning a Neo4j session.
    """
    if not instructions:
        return
    rows = instruction_rows(instructions, source)
    pairs = instruction_member_pairs(instructions, nodes, source)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(queries.MERGE_INSTRUCTIONS, rows=rows, now=now)
        if pairs:
            await session.run(
                queries.MERGE_INSTRUCTION_MEMBERS, pairs=pairs, now=now
            )


async def persist_instruction_governance(
    statements: list[models.Statement],
    instructions: list[models.Instruction],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    """Persists instruction governance edges to statement nodes."""
    pairs = instruction_governance_pairs(statements, instructions, source)
    if not pairs:
        return
    async with session_factory() as session:
        await session.run(
            queries.MERGE_INSTRUCTION_GOVERNANCE,
            pairs=pairs,
            now=utcnow_iso(),
        )


async def persist_procedures(
    procedures: list[models.Procedure],
    doc_nodes: list[models.SourceNode],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    """Persists procedures and their source-node memberships.

    Args:
        procedures: The procedures to persist.
        doc_nodes: The document nodes in order (for member resolution).
        source: The source key.
        session_factory: Async callable returning a Neo4j session.
    """
    if any(procedure.uuid is None for procedure in procedures):
        raise ValueError('cannot persist procedures without assigned uuids')
    procedure_batch = procedure_rows(procedures, source)
    if not procedure_batch:
        return
    members = procedure_member_pairs(procedures, source, doc_nodes)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(
            queries.MERGE_PROCEDURES, rows=procedure_batch, now=now
        )
        if members:
            await session.run(
                queries.MERGE_PROCEDURE_MEMBERS, pairs=members, now=now
            )


async def persist_statement_procedure_links(
    statements: list[models.Statement],
    procedures: list[models.Procedure],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    """Links statements that are also procedures via HAS_PROCEDURE.

    Args:
        statements: The statements to link.
        procedures: The procedures to link against.
        source: The source key.
        session_factory: Async callable returning a Neo4j session.
    """
    pairs = has_procedure_pairs(statements, procedures, source)
    if not pairs:
        return
    now = utcnow_iso()
    async with session_factory() as session:
        await session.run(queries.MERGE_HAS_PROCEDURE, pairs=pairs, now=now)


async def persist_assertions(
    triplets: list[models.Triplet],
    source: str,
    doc_nodes: list[models.SourceNode],
    *,
    session_factory: Callable,
    entity_descriptions: dict[int, dict[str, str | None]] | None = None,
    predicate_descriptions: dict[int, dict[str, str | None]] | None = None,
    entity_embeddings: dict[int, dict[str, list[float]]] | None = None,
    predicate_embeddings: dict[int, dict[str, list[float]]] | None = None,
    event_descriptions: dict[int, dict[str, str | None]] | None = None,
    event_embeddings: dict[int, dict[str, list[float]]] | None = None,
) -> None:
    """Persists the assertion layer: triplets, entities, events, and predicates.

    Writes the :Triplet anchors, node-local :Entity and :Event components,
    and per-triplet :Predicate components, using the per-node descriptions
    and embeddings when available.
    """
    if not triplets:
        return
    missing = [
        index
        for index, triplet in enumerate(triplets)
        if any(
            triplet.occurrence_uuids.get(node_position) is None
            for node_position in triplet.evidence_positions
        )
    ]
    if missing:
        raise ValueError(
            f'cannot persist triplets without assigned occurrence uuids: '
            f'{missing}'
        )
    rows = assertions.assertion_rows(
        triplets,
        source,
        entity_descriptions or {},
        predicate_descriptions or {},
        entity_embeddings,
        predicate_embeddings,
        event_descriptions,
        event_embeddings,
    )
    pairs = evidence_pairs(triplets, source, doc_nodes)
    now = utcnow_iso()

    async with session_factory() as session:
        await session.run(
            queries.MERGE_TRIPLETS,
            rows=triplet_rows(triplets, source),
            now=now,
        )
        if pairs:
            await session.run(
                queries.MERGE_TRIPLET_EVIDENCE, pairs=pairs, now=now
            )
        if rows['events']:
            await session.run(
                queries.MERGE_EVENTS, rows=rows['events'], now=now
            )
        if rows['event_names']:
            await session.run(
                queries.MERGE_EVENT_NAMES,
                rows=rows['event_names'],
                now=now,
            )
        if rows['event_name_edges']:
            await session.run(
                queries.MERGE_HAS_EVENT_NAME,
                pairs=rows['event_name_edges'],
            )
        if rows['entities']:
            await session.run(
                queries.MERGE_ENTITIES, rows=rows['entities'], now=now
            )
            await session.run(
                queries.MERGE_ENTITY_NAMES,
                rows=rows['entity_names'],
                now=now,
            )
        if rows['predicates']:
            await session.run(
                queries.MERGE_PREDICATES, rows=rows['predicates'], now=now
            )
            await session.run(
                queries.MERGE_PREDICATE_NAMES,
                rows=rows['predicate_names'],
                now=now,
            )
        if rows['entity_name_edges']:
            await session.run(
                queries.MERGE_HAS_ENTITY_NAME,
                pairs=rows['entity_name_edges'],
            )
        if rows['predicate_name_edges']:
            await session.run(
                queries.MERGE_HAS_PREDICATE_NAME,
                pairs=rows['predicate_name_edges'],
            )
        if rows['subject_edges']:
            await session.run(
                queries.MERGE_HAS_SUBJECT, pairs=rows['subject_edges'], now=now
            )
        if rows['object_edges']:
            await session.run(
                queries.MERGE_HAS_OBJECT, pairs=rows['object_edges'], now=now
            )
        if rows['predicate_edges']:
            await session.run(
                queries.MERGE_HAS_PREDICATE,
                pairs=rows['predicate_edges'],
                now=now,
            )


async def persist_name_occurrences(
    components: list[dict],
    kind: str,
    *,
    session_factory: Callable,
) -> None:
    """Materializes lexical occurrence nodes for persisted components."""
    rows = [
        names.name_properties(
            kind,
            component['source'],
            component['uuid'],
            component['name'],
            component['node_position'],
        )
        for component in components
    ]
    if not rows:
        return
    pairs = [
        {'component': component['uuid'], 'name': row['uuid']}
        for component, row in zip(components, rows, strict=True)
    ]
    now = utcnow_iso()
    async with session_factory() as session:
        await session.run(
            queries.MERGE_ENTITY_NAMES
            if kind == 'entity'
            else queries.MERGE_PREDICATE_NAMES,
            rows=rows,
            now=now,
        )
        await session.run(
            queries.MERGE_HAS_ENTITY_NAME
            if kind == 'entity'
            else queries.MERGE_HAS_PREDICATE_NAME,
            pairs=pairs,
        )


async def clear_name_hubs(
    kind: str,
    source: str,
    *,
    session_factory: Callable,
) -> None:
    """Clears only one source's lexical hubs and projections."""
    source_id = source_uuid(source)
    async with session_factory() as session:
        await session.run(
            queries.delete_semantic_name_links_query(kind),
            source_uuid=source_id,
        )
        await session.run(
            queries.delete_name_hubs_query(kind),
            source_uuid=source_id,
        )


async def persist_name_hubs(
    kind: str,
    hubs: list[dict],
    *,
    source: str,
    session_factory: Callable,
) -> None:
    """Persists lexical hubs and deterministic semantic projections."""
    if not hubs:
        return
    rows = [
        names.name_hub_properties(
            kind,
            source,
            hub['canonical_form'],
            hub['aliases'],
            hub_id=hub['uuid'],
        )
        for hub in hubs
    ]
    pairs = [
        {'name': name_id, 'hub': hub['uuid']}
        for hub in hubs
        for name_id in hub['members']
    ]
    source_id = source_uuid(source)
    async with session_factory() as session:
        await session.run(
            queries.merge_name_hubs_query(kind),
            rows=rows,
            now=utcnow_iso(),
        )
        await session.run(
            queries.merge_name_hub_memberships_query(kind),
            pairs=pairs,
        )
        await session.run(
            queries.merge_semantic_name_links_query(kind),
            source_uuid=source_id,
        )


async def clear_global_name_hubs(
    kind: str,
    *,
    session_factory: Callable,
) -> None:
    """Clears the disposable cross-source lexical hub tier."""
    async with session_factory() as session:
        await session.run(queries.delete_meta_name_hubs_query(kind))


async def _validate_global_name_hubs(
    kind: str,
    hubs: list[dict],
    session_factory: Callable,
) -> None:
    records = await queries.all_name_hubs(session_factory, kind)
    source_by_hub = {record['uuid']: record.get('source') for record in records}
    for hub in hubs:
        members = set(hub.get('members', []))
        missing = sorted(members - set(source_by_hub))
        if missing:
            raise ValueError(f'source name hub(s) not found: {missing}')
        sources = {source_by_hub[member] for member in members}
        sources.discard(None)
        declared_sources = set(hub.get('sources', []))
        if declared_sources != sources:
            raise ValueError(
                'meta name hub provenance does not match its members: '
                f'{hub.get("uuid")}'
            )
        if len(sources) < 2:
            raise ValueError(
                'meta name hubs require two distinct source supports: '
                f'{hub.get("uuid")}'
            )


async def persist_global_name_hubs(
    kind: str,
    hubs: list[dict],
    *,
    session_factory: Callable,
) -> None:
    """Persists qualified meta lexical hubs and ALIGNS_TO edges."""
    if not hubs:
        return
    await _validate_global_name_hubs(kind, hubs, session_factory)
    rows = [
        names.global_name_hub_properties(
            kind,
            hub['canonical_form'],
            hub['aliases'],
            hub['sources'],
            hub_id=hub['uuid'],
        )
        for hub in hubs
    ]
    pairs = [
        {'source_hub': member, 'meta_hub': hub['uuid']}
        for hub in hubs
        for member in hub['members']
    ]
    async with session_factory() as session:
        await session.run(
            queries.merge_global_name_hubs_query(kind),
            rows=rows,
            now=utcnow_iso(),
        )
        await session.run(
            queries.merge_name_hub_alignments_query(kind),
            pairs=pairs,
            now=utcnow_iso(),
        )


async def _validate_global_hubs(
    hubs: list[dict],
    session_factory: Callable,
    source_hubs_query: Callable,
) -> None:
    member_ids = sorted(
        {member for hub in hubs for member in hub.get('members', [])}
    )
    records = await source_hubs_query(
        session_factory,
        hub_uuids=member_ids,
    )
    source_by_hub = {record['uuid']: record.get('source') for record in records}
    for hub in hubs:
        members = set(hub.get('members', []))
        missing = sorted(members - set(source_by_hub))
        if missing:
            raise ValueError(f'source hub(s) not found: {missing}')
        sources = {
            source_by_hub[member] for member in members if source_by_hub[member]
        }
        if len(sources) < 2:
            raise ValueError(
                'meta hubs require two distinct source supports: '
                f'{hub.get("uuid")}'
            )


async def persist_entity_hubs(
    hubs: list[dict],
    *,
    session_factory: Callable,
    tier: str,
) -> None:
    await _persist_semantic_hubs(
        hubs,
        session_factory=session_factory,
        graph_module=local_entity_hubs,
        source_hubs_query=queries.all_entity_source_hubs,
        tier=tier,
    )


async def persist_event_hubs(
    hubs: list[dict],
    *,
    session_factory: Callable,
    tier: str,
) -> None:
    await _persist_semantic_hubs(
        hubs,
        session_factory=session_factory,
        graph_module=local_event_hubs,
        source_hubs_query=queries.all_event_source_hubs,
        tier=tier,
    )


async def persist_predicate_hubs(
    hubs: list[dict],
    *,
    session_factory: Callable,
    tier: str,
) -> None:
    await _persist_semantic_hubs(
        hubs,
        session_factory=session_factory,
        graph_module=local_predicate_hubs,
        source_hubs_query=queries.all_predicate_source_hubs,
        tier=tier,
    )


async def _persist_semantic_hubs(
    hubs: list[dict],
    *,
    session_factory: Callable,
    graph_module,
    source_hubs_query: Callable,
    tier: str,
) -> None:
    if not hubs:
        return
    if tier not in {'source', 'meta'}:
        raise ValueError(f'unknown hub tier: {tier}')
    if tier == 'meta':
        singleton_hubs = [
            hub.get('uuid')
            for hub in hubs
            if len(set(hub.get('members', []))) < 2
        ]
        if singleton_hubs:
            raise ValueError(
                'meta hubs require at least two source-hub members: '
                f'{singleton_hubs}'
            )
        await _validate_global_hubs(hubs, session_factory, source_hubs_query)

    hub_rows = [
        graph_module.hub_properties(
            source=hub.get('source'),
            canonical_name=hub['canonical_name'],
            aliases=hub['aliases'],
            description=hub['description'],
            embedding=hub.get('embedding'),
            tier=tier,
            hub_id=hub.get('uuid'),
        )
        for hub in hubs
    ]
    label = graph_module.hub_label(tier)
    now = utcnow_iso()
    if tier == 'source':
        member_pairs = [
            {'component': member, 'hub': row['uuid']}
            for hub, row in zip(hubs, hub_rows, strict=True)
            for member in hub.get('members', [])
        ]
    else:
        member_pairs = [
            {'source_hub': member, 'meta_hub': row['uuid']}
            for hub, row in zip(hubs, hub_rows, strict=True)
            for member in hub.get('members', [])
        ]

    async with session_factory() as session:
        await session.run(
            queries.merge_hubs_query(label), rows=hub_rows, now=now
        )
        if member_pairs:
            if tier == 'source':
                await session.run(
                    queries.merge_canonical_query(
                        graph_module.COMPONENT_LABEL, label
                    ),
                    pairs=member_pairs,
                    now=now,
                )
            else:
                await session.run(
                    queries.merge_alignment_query(
                        graph_module.hub_label(), label
                    ),
                    pairs=member_pairs,
                    now=now,
                )


async def persist_triplet_hubs(
    groups: list[dict],
    *,
    tier: str,
    session_factory: Callable,
) -> None:
    """Persists deterministic TripletHub or MetaTripletHub groups."""
    if not groups:
        return
    if tier not in {'source', 'meta'}:
        raise ValueError(f'unknown triplet hub tier: {tier}')
    if tier == 'meta':
        invalid = [
            group.get('uuid')
            for group in groups
            if len(set(group.get('sources', []))) < 2
        ]
        if invalid:
            raise ValueError(
                'meta triplet hubs require two distinct source supports: '
                f'{invalid}'
            )
    rows = [
        triplet_hub_properties(
            tier=tier,
            source=group.get('source'),
            canonical_name=group['canonical_name'],
            description=group['description'],
            embedding=group.get('embedding'),
            subject_hub=group['subject_hub'],
            predicate_hub=group['predicate_hub'],
            object_hub=group['object_hub'],
            hub_id=group['uuid'],
        )
        for group in groups
    ]
    role_pairs = [
        {
            'hub': group['uuid'],
            'subject_hub': group['subject_hub'],
            'predicate_hub': group['predicate_hub'],
            'object_hub': group['object_hub'],
        }
        for group in groups
    ]
    evidence_pairs = [
        {'triplet': triplet, 'hub': group['uuid']}
        for group in groups
        for triplet in group.get('triplets', [])
    ]
    support_pairs = [
        {'source_hub': source_hub, 'meta_hub': group['uuid']}
        for group in groups
        for source_hub in group.get('local_hubs', [])
    ]
    now = utcnow_iso()
    async with session_factory() as session:
        await session.run(
            queries.merge_triplet_hubs_query(tier), rows=rows, now=now
        )
        await session.run(
            queries.merge_triplet_hub_edges_query(tier),
            rows=role_pairs,
            now=now,
        )
        if tier == 'source' and evidence_pairs:
            await session.run(
                queries.merge_triplet_hub_evidence_query(),
                pairs=evidence_pairs,
            )
        if tier == 'meta' and support_pairs:
            await session.run(
                queries.merge_meta_triplet_hub_support_query(),
                pairs=support_pairs,
            )


async def clear_triplet_hubs(
    tier: str,
    *,
    session_factory: Callable,
    source: str | None = None,
) -> None:
    """Deletes only the requested disposable triplet-hub tier."""
    async with session_factory() as session:
        await session.run(
            queries.delete_triplet_hubs_query(
                tier,
                source=source_uuid(source) if source else None,
            ),
            source_uuid=source_uuid(source) if source else None,
        )


async def attach_entity_components(
    assignments: list[dict], *, aliases: list[dict], session_factory: Callable
) -> None:
    await _attach_source_components(
        assignments,
        aliases=aliases,
        session_factory=session_factory,
        component_node_label=local_entity_hubs.COMPONENT_LABEL,
        source_hub_label=local_entity_hubs.hub_label(),
    )


async def attach_event_components(
    assignments: list[dict], *, aliases: list[dict], session_factory: Callable
) -> None:
    await _attach_source_components(
        assignments,
        aliases=aliases,
        session_factory=session_factory,
        component_node_label=local_event_hubs.COMPONENT_LABEL,
        source_hub_label=local_event_hubs.hub_label(),
    )


async def attach_predicate_components(
    assignments: list[dict], *, aliases: list[dict], session_factory: Callable
) -> None:
    await _attach_source_components(
        assignments,
        aliases=aliases,
        session_factory=session_factory,
        component_node_label=local_predicate_hubs.COMPONENT_LABEL,
        source_hub_label=local_predicate_hubs.hub_label(),
    )


async def _attach_source_components(
    assignments: list[dict],
    *,
    aliases: list[dict],
    session_factory: Callable,
    component_node_label: str,
    source_hub_label: str,
) -> None:
    if not assignments and not aliases:
        return
    now = utcnow_iso()
    async with session_factory() as session:
        if assignments:
            await session.run(
                queries.merge_canonical_query(
                    component_node_label, source_hub_label
                ),
                pairs=assignments,
                now=now,
            )
        if aliases:
            await session.run(
                queries.update_hub_aliases_query(source_hub_label),
                rows=aliases,
                now=now,
            )


async def _validate_meta_assignments(
    assignments: list[dict],
    session_factory: Callable,
    source_hubs_query: Callable,
    qualified_meta_hubs_query: Callable,
) -> None:
    source_hub_uuids = sorted(
        {assignment['source_hub'] for assignment in assignments}
    )
    records = await source_hubs_query(
        session_factory, hub_uuids=source_hub_uuids
    )
    source_by_hub = {record['uuid']: record.get('source') for record in records}
    missing = sorted(set(source_hub_uuids) - set(source_by_hub))
    if missing:
        raise ValueError(f'source hub(s) not found: {missing}')

    batch_sources: dict[str, set[str]] = {}
    for assignment in assignments:
        source = source_by_hub[assignment['source_hub']]
        if not source:
            raise ValueError(
                f'source hub lacks provenance: {assignment["source_hub"]}'
            )
        batch_sources.setdefault(assignment['meta_hub'], set()).add(source)
    qualified = await qualified_meta_hubs_query(session_factory)
    invalid = sorted(
        global_hub
        for global_hub, sources in batch_sources.items()
        if global_hub not in qualified and len(sources) < 2
    )
    if invalid:
        raise ValueError(
            'meta alignment requires two distinct sources for new hubs: '
            f'{invalid}'
        )


async def attach_entity_global_hubs(
    assignments: list[dict],
    *,
    aliases: list[dict],
    session_factory: Callable,
) -> None:
    await _attach_meta_hubs(
        assignments,
        aliases=aliases,
        session_factory=session_factory,
        source_hubs_query=queries.all_entity_source_hubs,
        qualified_meta_hubs_query=queries.qualified_entity_meta_hub_uuids,
        source_hub_label=local_entity_hubs.hub_label(),
        global_hub_label=local_entity_hubs.hub_label('meta'),
    )


async def attach_event_global_hubs(
    assignments: list[dict],
    *,
    aliases: list[dict],
    session_factory: Callable,
) -> None:
    await _attach_meta_hubs(
        assignments,
        aliases=aliases,
        session_factory=session_factory,
        source_hubs_query=queries.all_event_source_hubs,
        qualified_meta_hubs_query=queries.qualified_event_meta_hub_uuids,
        source_hub_label=local_event_hubs.hub_label(),
        global_hub_label=local_event_hubs.hub_label('meta'),
    )


async def attach_predicate_global_hubs(
    assignments: list[dict],
    *,
    aliases: list[dict],
    session_factory: Callable,
) -> None:
    await _attach_meta_hubs(
        assignments,
        aliases=aliases,
        session_factory=session_factory,
        source_hubs_query=queries.all_predicate_source_hubs,
        qualified_meta_hubs_query=queries.qualified_predicate_meta_hub_uuids,
        source_hub_label=local_predicate_hubs.hub_label(),
        global_hub_label=local_predicate_hubs.hub_label('meta'),
    )


async def _attach_meta_hubs(
    assignments: list[dict],
    *,
    aliases: list[dict],
    session_factory: Callable,
    source_hubs_query: Callable,
    qualified_meta_hubs_query: Callable,
    source_hub_label: str,
    global_hub_label: str,
) -> None:
    if not assignments and not aliases:
        return
    if assignments:
        await _validate_meta_assignments(
            assignments,
            session_factory,
            source_hubs_query,
            qualified_meta_hubs_query,
        )
    now = utcnow_iso()
    async with session_factory() as session:
        source_hubs = sorted(
            {assignment['source_hub'] for assignment in assignments}
        )
        if source_hubs:
            await session.run(
                queries.delete_hub_alignments_query(source_hub_label),
                source_hubs=source_hubs,
            )
            await session.run(
                queries.merge_alignment_query(
                    source_hub_label, global_hub_label
                ),
                pairs=assignments,
                now=now,
            )
        if aliases:
            await session.run(
                queries.update_hub_aliases_query(global_hub_label),
                rows=aliases,
                now=now,
            )


async def clear_entity_hubs(source: str, *, session_factory: Callable) -> None:
    await _clear_source_hubs(
        source,
        session_factory=session_factory,
        label=local_entity_hubs.hub_label(),
    )


async def clear_event_hubs(source: str, *, session_factory: Callable) -> None:
    await _clear_source_hubs(
        source,
        session_factory=session_factory,
        label=local_event_hubs.hub_label(),
    )


async def clear_predicate_hubs(
    source: str, *, session_factory: Callable
) -> None:
    await _clear_source_hubs(
        source,
        session_factory=session_factory,
        label=local_predicate_hubs.hub_label(),
    )


async def _clear_source_hubs(
    source: str,
    *,
    session_factory: Callable,
    label: str,
) -> None:
    """Deletes only source-local hubs belonging to one source.

    Durable components and hubs from every other source are preserved.
    Detached alignment and canonical relationships are derived and are
    removed along with the rebuilt source hubs.
    """
    async with session_factory() as session:
        await session.run(
            queries.delete_source_hubs_query(label),
            source=source,
        )


async def clear_invalid_entity_global_hubs(
    *, session_factory: Callable
) -> None:
    await _clear_invalid_meta_hubs(
        session_factory=session_factory,
        label=local_entity_hubs.hub_label('meta'),
    )


async def clear_invalid_event_global_hubs(*, session_factory: Callable) -> None:
    await _clear_invalid_meta_hubs(
        session_factory=session_factory,
        label=local_event_hubs.hub_label('meta'),
    )


async def clear_invalid_predicate_global_hubs(
    *, session_factory: Callable
) -> None:
    await _clear_invalid_meta_hubs(
        session_factory=session_factory,
        label=local_predicate_hubs.hub_label('meta'),
    )


async def _clear_invalid_meta_hubs(
    *,
    session_factory: Callable,
    label: str,
) -> None:
    """Deletes meta hubs without support from two distinct sources."""
    async with session_factory() as session:
        await session.run(queries.delete_invalid_meta_hubs_query(label))


async def clear_entity_global_hubs(*, session_factory: Callable) -> None:
    await _clear_global_hubs(
        session_factory=session_factory,
        label=local_entity_hubs.hub_label('meta'),
    )


async def clear_event_global_hubs(*, session_factory: Callable) -> None:
    await _clear_global_hubs(
        session_factory=session_factory,
        label=local_event_hubs.hub_label('meta'),
    )


async def clear_predicate_global_hubs(*, session_factory: Callable) -> None:
    await _clear_global_hubs(
        session_factory=session_factory,
        label=local_predicate_hubs.hub_label('meta'),
    )


async def _clear_global_hubs(
    *,
    session_factory: Callable,
    label: str,
) -> None:
    """Deletes the disposable meta tier for one kind.

    Source hubs, components, and durable learning data are not touched.
    """
    async with session_factory() as session:
        await session.run(queries.delete_hubs_query(label))


async def persist_cards(
    cards: list[models.Card],
    *,
    session_factory: Callable,
) -> None:
    """Persists learning Card nodes."""
    if not cards:
        return
    rows = [learning.card_properties(card) for card in cards]
    now = utcnow_iso()
    async with session_factory() as session:
        await session.run(queries.merge_cards_query(), rows=rows, now=now)


async def persist_card_target_edges(
    cards: list[models.Card],
    *,
    session_factory: Callable,
) -> None:
    """Persists source-local component→HAS_CARD edges."""
    if not cards:
        return
    pairs = [{'target': card.target_uuid, 'card': card.uuid} for card in cards]
    async with session_factory() as session:
        await session.run(queries.merge_card_target_edges_query(), pairs=pairs)


async def persist_reviews(
    reviews: list[models.Review],
    *,
    session_factory: Callable,
) -> None:
    """Persists Review nodes."""
    if not reviews:
        return
    rows = [learning.review_properties(review) for review in reviews]
    now = utcnow_iso()
    async with session_factory() as session:
        await session.run(queries.merge_reviews_query(), rows=rows, now=now)


async def persist_card_review_edges(
    reviews: list[models.Review],
    *,
    session_factory: Callable,
) -> None:
    """Persists CARD->HAS_REVIEW edges."""
    if not reviews:
        return
    pairs = [
        {'card': review.card_uuid, 'review': review.uuid} for review in reviews
    ]
    now = utcnow_iso()
    async with session_factory() as session:
        await session.run(
            queries.merge_card_review_edges_query(), pairs=pairs, now=now
        )
