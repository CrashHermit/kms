"""Idempotent persistence of extraction results into Neo4j."""

from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from kms.core import models
from kms.graph import assertions, names, queries
from kms.graph.hubs import (
    component_label,
    hub_label,
    hub_properties,
    triplet_hub_properties,
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
    """Returns the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat(timespec='seconds')


def node_batches(
    nodes: list[models.ASTNode], source: str
) -> dict[str | None, list[dict]]:
    """Groups node rows by their Neo4j label for batched merges.

    Args:
        nodes: The AST nodes to batch.
        source: The source key.

    Returns:
        A mapping from node label to the rows for that label.
    """
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
    """Persists the source vertex and all AST nodes.

    Args:
        nodes: The AST nodes to persist.
        source: The source key.
        session_factory: Async callable returning a Neo4j session.
        metadata: Optional extra source metadata.
    """
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
    """Returns the uuid of every id-ordered node for the NEXT chain."""
    return [node_uuid(source, node.id) for node in nodes if node.id is not None]


def _chain_pairs(chain: list[str]) -> list[dict]:
    """Builds consecutive from/to pairs for the NEXT edges."""
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
    source: str,
    *,
    session_factory: Callable,
) -> None:
    """Persists statement vertices and their MEMBER_OF edges.

    Args:
        statements: The statements to persist.
        source: The source key.
        session_factory: Async callable returning a Neo4j session.
    """
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
    """Persists instruction vertices and their MEMBER_OF edges.

    Args:
        instructions: The instructions to persist.
        source: The source key.
        session_factory: Async callable returning a Neo4j session.
    """
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
    """Persists procedures, their steps, and FIRST/THEN edges.

    Args:
        procedures: The procedures to persist.
        source: The source key.
        session_factory: Async callable returning a Neo4j session.
    """
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
    *,
    session_factory: Callable,
    entity_descriptions: dict[int, dict[str, str | None]] | None = None,
    predicate_descriptions: dict[int, dict[str, str | None]] | None = None,
    entity_embeddings: dict[int, dict[str, list[float]]] | None = None,
    predicate_embeddings: dict[int, dict[str, list[float]]] | None = None,
) -> None:
    """Persists the assertion layer: triplets, entities, and predicates.

    Writes the :Triplet anchors, the node-local :Entity and per-triplet
    :Predicate components, and the edges that group them, using the
    per-node descriptions and embeddings when available.
    """
    if not triplets:
        return
    rows = assertions.assertion_rows(
        triplets,
        source,
        entity_descriptions or {},
        predicate_descriptions or {},
        entity_embeddings,
        predicate_embeddings,
    )
    pairs = evidence_pairs(triplets, source)
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
        if rows['entities']:
            await session.run(
                queries.MERGE_ENTITIES, rows=rows['entities'], now=now
            )
        if rows['predicates']:
            await session.run(
                queries.MERGE_PREDICATES, rows=rows['predicates'], now=now
            )
        if rows['entity_names']:
            await session.run(
                queries.MERGE_ENTITY_NAMES,
                rows=rows['entity_names'],
                now=now,
            )
        if rows['predicate_names']:
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
            component['node_id'],
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


async def clear_meta_name_hubs(
    kind: str,
    *,
    session_factory: Callable,
) -> None:
    """Clears the disposable cross-source lexical hub tier."""
    async with session_factory() as session:
        await session.run(queries.delete_meta_name_hubs_query(kind))


async def _validate_meta_name_hubs(
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


async def persist_meta_name_hubs(
    kind: str,
    hubs: list[dict],
    *,
    session_factory: Callable,
) -> None:
    """Persists qualified meta lexical hubs and ALIGNS_TO edges."""
    if not hubs:
        return
    await _validate_meta_name_hubs(kind, hubs, session_factory)
    rows = [
        names.meta_name_hub_properties(
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
            queries.merge_meta_name_hubs_query(kind),
            rows=rows,
            now=utcnow_iso(),
        )
        await session.run(
            queries.merge_name_hub_alignments_query(kind),
            pairs=pairs,
            now=utcnow_iso(),
        )


async def _validate_meta_hubs(
    kind: str,
    hubs: list[dict],
    session_factory: Callable,
) -> None:
    member_ids = sorted(
        {member for hub in hubs for member in hub.get('members', [])}
    )
    records = await queries.all_source_hubs(
        session_factory,
        kind,
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


async def persist_hubs(
    kind: str,
    hubs: list[dict],
    *,
    session_factory: Callable,
    subsumption_edges: list[dict] | None = None,
    tier: str,
) -> None:
    """Persists source hubs or meta hubs and their derived relationships.

    Source hubs use ``CANONICAL`` membership from Entity/Predicate nodes.
    Meta hubs use ``ALIGNS_TO`` membership from source-local hubs. Meta hub
    rows must provide an explicit stable ``uuid`` and may omit ``source``.

    Args:
        kind: The hub kind, 'entity' or 'predicate'.
        hubs: Hub dicts with canonical_name, aliases, description,
            members, and optional embedding. Meta hubs additionally require
            ``uuid``.
        session_factory: Async callable returning a Neo4j session.
        subsumption_edges: Optional general→specific hub pairs.
        tier: ``source`` for EntityHub/PredicateHub or ``meta`` for the
            disposable cross-source layer.
    """
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
        await _validate_meta_hubs(kind, hubs, session_factory)

    hub_rows = [
        hub_properties(
            kind=kind,
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
    label = hub_label(kind, tier=tier)
    now = utcnow_iso()

    if tier == 'source':
        component_node_label = component_label(kind)
        member_pairs: list[dict] = []
        for hub, row in zip(hubs, hub_rows, strict=True):
            for component_uuid in hub.get('members', []):
                member_pairs.append(
                    {'component': component_uuid, 'hub': row['uuid']}
                )
    else:
        source_hub_label = hub_label(kind, tier='source')
        member_pairs = []
        for hub, row in zip(hubs, hub_rows, strict=True):
            for source_hub_uuid in hub.get('members', []):
                member_pairs.append(
                    {
                        'source_hub': source_hub_uuid,
                        'meta_hub': row['uuid'],
                    }
                )

    async with session_factory() as session:
        await session.run(
            queries.merge_hubs_query(label), rows=hub_rows, now=now
        )
        if member_pairs:
            if tier == 'source':
                await session.run(
                    queries.merge_canonical_query(component_node_label, label),
                    pairs=member_pairs,
                    now=now,
                )
            else:
                await session.run(
                    queries.merge_alignment_query(
                        source_hub_label,
                        label,
                    ),
                    pairs=member_pairs,
                    now=now,
                )
        if subsumption_edges:
            await session.run(
                queries.merge_subsumes_query(label),
                pairs=subsumption_edges,
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


async def attach_source_components(
    kind: str,
    assignments: list[dict],
    *,
    aliases: list[dict],
    session_factory: Callable,
) -> None:
    if not assignments and not aliases:
        return
    component_node_label = component_label(kind)
    source_hub_label = hub_label(kind, tier='source')
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
    kind: str,
    assignments: list[dict],
    session_factory: Callable,
) -> None:
    source_hub_uuids = sorted(
        {assignment['source_hub'] for assignment in assignments}
    )
    records = await queries.all_source_hubs(
        session_factory,
        kind,
        hub_uuids=source_hub_uuids,
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
    qualified = await queries.qualified_meta_hub_uuids(
        session_factory,
        kind,
    )
    invalid = sorted(
        meta_hub
        for meta_hub, sources in batch_sources.items()
        if meta_hub not in qualified and len(sources) < 2
    )
    if invalid:
        raise ValueError(
            'meta alignment requires two distinct sources for new hubs: '
            f'{invalid}'
        )


async def attach_meta_hubs(
    kind: str,
    assignments: list[dict],
    *,
    aliases: list[dict],
    subsumption_edges: list[dict],
    session_factory: Callable,
) -> None:
    if not assignments and not aliases and not subsumption_edges:
        return
    if assignments:
        await _validate_meta_assignments(kind, assignments, session_factory)
    source_hub_label = hub_label(kind, tier='source')
    meta_hub_label = hub_label(kind, tier='meta')
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
                queries.merge_alignment_query(source_hub_label, meta_hub_label),
                pairs=assignments,
                now=now,
            )
        if aliases:
            await session.run(
                queries.update_hub_aliases_query(meta_hub_label),
                rows=aliases,
                now=now,
            )
        if subsumption_edges:
            await session.run(
                queries.merge_subsumes_query(meta_hub_label),
                pairs=subsumption_edges,
                now=now,
            )


async def clear_source_hubs(
    kind: str,
    source: str,
    *,
    session_factory: Callable,
) -> None:
    """Deletes only source-local hubs belonging to one source.

    Durable components and hubs from every other source are preserved.
    Detached alignment and canonical relationships are derived and are
    removed along with the rebuilt source hubs.
    """
    label = hub_label(kind, tier='source')
    async with session_factory() as session:
        await session.run(
            queries.delete_source_hubs_query(label),
            source=source,
        )


async def clear_invalid_meta_hubs(
    kind: str,
    *,
    session_factory: Callable,
) -> None:
    """Deletes meta hubs without support from two distinct sources."""
    label = hub_label(kind, tier='meta')
    async with session_factory() as session:
        await session.run(queries.delete_invalid_meta_hubs_query(label))


async def clear_meta_hubs(
    kind: str,
    *,
    session_factory: Callable,
) -> None:
    """Deletes the disposable meta tier for one kind.

    Source hubs, components, and durable learning data are not touched.
    """
    label = hub_label(kind, tier='meta')
    async with session_factory() as session:
        await session.run(queries.delete_hubs_query(label))
