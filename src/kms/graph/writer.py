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

The Cypher lives in ``queries`` — this module composes rows and edge pairs
and hands them to the named queries; no query string is embedded here.
"""

from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from kms.core import models
from kms.graph import queries
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
from kms.graph.triplets import (
    has_object_pairs,
    has_subject_pairs,
    triplet_rows,
    yields_pairs,
)
from kms.graph.community import (
    community_rows,
    evidence_pairs as community_evidence_pairs,
    member_pairs as community_member_pairs,
)
from kms.graph.triplet_hubs import (
    canonical_object_pairs,
    canonical_predicate_pairs as th_canonical_predicate_pairs,
    canonical_subject_pairs,
    supported_by_pairs,
)
from kms.graph.fact_hubs import (
    fact_hub_rows,
    has_fact_pairs,
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
        await session.run(queries.MERGE_INSTRUCTIONS, rows=rows, now=now)
        if pairs:
            await session.run(queries.MERGE_GOVERNS, pairs=pairs, now=now)


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
    """Upsert the ``:Fact`` nodes and their ``:EVIDENCE_FOR`` edges.

    One ``:Fact`` per atomic fact, carrying its text. Each provenance node
    the fact draws on points at it via ``:EVIDENCE_FOR``, the same
    raw-material → construct anchor the statement and procedure tiers use.

    Args:
        facts: The atomic facts, in document order.
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
    """Upsert the ``:Entity`` vertices.

    One ``:Entity`` per (node, triplet, role) — each triplet's subject and
    object reified as their own vertices, carrying the enricher's description
    of that surface form at that node. Entities are reached through their
    triplets (the triplet hub's ``:HAS_SUBJECT``/``:HAS_OBJECT`` edges point
    at them); there is no direct ``(:Node)-[:HAS_ENTITY]->(:Entity)`` edge.

    Args:
        triplets: The triplets, in document order.
        facts: The atomic facts, in document order.
        node_entity_descriptions: The enricher's per-node mapping: node id to
            a list of ``{name, description}`` dicts.
        source: The stable book identity.
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
    """
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
    """Upsert the ``:Triplet`` hubs and their edges.

    One ``:Triplet`` hub per (node, triplet) — an empty connector. The
    triplet's content lives on its edges: ``(:Triplet)-[:HAS_SUBJECT]->(:Entity)``,
    ``(:Triplet)-[:HAS_PREDICATE]->(:Predicate)``, and
    ``(:Triplet)-[:HAS_OBJECT]->(:Entity)``. Each also points back at its
    source ``:Fact`` via ``(:Fact)-[:YIELDS]->(:Triplet)``.

    Args:
        triplets: The triplets, in document order.
        facts: The atomic facts, in document order.
        source: The stable book identity.
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
    """
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
    """Upsert the ``:Predicate`` component vertices and their edges.

    One ``:Predicate`` per (node, triplet) — the DESCRIBED predicate
    component of the triplet hub — carrying the predicate text and, when
    written, its description. Each triplet hub points at the component via
    ``(:Triplet)-[:HAS_PREDICATE]->(:Predicate)``.

    Args:
        triplets: The triplets, in document order.
        facts: The atomic facts, in document order.
        source: The stable book identity.
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
        node_predicate_descriptions: The enricher's per-node predicate
            descriptions.
    """
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
    """Upsert ``:EntityHub`` vertices, ``:Definition`` vertices, and their
    edges.

    One ``:EntityHub`` per cluster (empty connector). Each hub points at a
    ``:Definition`` carrying the synthesised canonical text and its
    embedding. Every spoke in the cluster points at the hub via
    ``(:Entity)-[:CANONICAL]->(:EntityHub)``.

    Args:
        entity_clusters: One list of spoke dicts per cluster. Each spoke
            dict carries at least ``uuid``.
        hub_definitions: One dict per hub:
            ``{hub_uuid, definition_text, definition_embedding}``.
        source: The stable book identity.
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
        definitions: Optional per-cluster dicts with ``display_name``
            for the hub node.
    """
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
    """Upsert ``:PredicateHub`` vertices, ``:Definition`` vertices, and
    their edges.

    One ``:PredicateHub`` per cluster (empty connector). Each hub points
    at a ``:Definition`` carrying the synthesised canonical text and its
    embedding. Every spoke in the cluster points at the hub via
    ``(:Predicate)-[:CANONICAL]->(:PredicateHub)``.

    Args:
        predicate_clusters: One list of spoke dicts per cluster.
        hub_definitions: One dict per hub (same shape as for entities).
        source: The stable book identity.
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
        definitions: Optional per-cluster dicts with ``display_name``
            for the hub node.
    """
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


async def persist_canonical_merge(
    spoke_uuids: list[str],
    hub_uuid: str,
    hub_type: str,
    definition_text: str,
    definition_embedding: list[float] | None,
    *,
    session_factory: Callable,
) -> None:
    """Merge a cluster of new spokes into an EXISTING hub.

    Writes ``:CANONICAL`` edges from each spoke to the existing hub,
    updates the ``:Definition`` text and embedding, and refreshes the
    ``:HAS_DEFINITION`` edge's ``modified_at``.

    Args:
        spoke_uuids: The uuids of the new spokes to attach.
        hub_uuid: The existing hub's uuid.
        hub_type: ``'entity'`` or ``'predicate'`` — picks which
            ``:CANONICAL`` edge type to write.
        definition_text: The regenerated canonical definition.
        definition_embedding: Its embedding, or None.
        session_factory: The injected session factory.
    """
    from kms.graph.definitions import (
        definition_properties,
        definition_uuid,
    )
    from kms.graph.entities import ENTITY_LABEL
    from kms.graph.entity_hubs import ENTITY_HUB_LABEL
    from kms.graph.predicate_hubs import PREDICATE_HUB_LABEL
    from kms.graph.predicates import PREDICATE_LABEL

    spoke_label = (
        ENTITY_LABEL if hub_type == 'entity' else PREDICATE_LABEL
    )
    hub_label = (
        ENTITY_HUB_LABEL
        if hub_type == 'entity'
        else PREDICATE_HUB_LABEL
    )

    now = utcnow_iso()
    def_uuid = definition_uuid(hub_uuid)
    def_props = definition_properties(
        hub_uuid, definition_text, definition_embedding
    )

    async with session_factory() as session:
        # 1. Add :CANONICAL edges from each new spoke to the existing hub
        canonical_cypher = (
            f'UNWIND $spoke_uuids AS spoke_uuid '
            f'MATCH (s:{spoke_label} {{uuid: spoke_uuid}}), '
            f'(h:{hub_label} {{uuid: $hub_uuid}}) '
            f'MERGE (s)-[r:CANONICAL]->(h) '
            f'ON CREATE SET r.created_at = $now '
            f'SET r.modified_at = $now'
        )
        await session.run(
            canonical_cypher,
            spoke_uuids=spoke_uuids,
            hub_uuid=hub_uuid,
            now=now,
        )

        # 2. Update the :Definition node's text and embedding
        await session.run(
            queries.MERGE_DEFINITIONS,
            rows=[def_props],
            now=now,
        )

        # 3. Bump the :HAS_DEFINITION edge
        await session.run(
            queries.MERGE_HAS_DEFINITION,
            pairs=[
                {'hub': hub_uuid, 'definition': def_uuid}
            ],
            now=now,
        )


async def persist_communities(
    communities: list[dict],
    source: str,
    *,
    session_factory: Callable,
) -> None:
    """Upsert ``:Community`` nodes and their edges.

    One ``:Community`` per detected community, carrying a summary text
    and its embedding.  Member hubs are linked via ``:HAS_MEMBER``;
    evidence triplets via ``:COMMUNITY_EVIDENCE``.

    Args:
        communities: One dict per community:
            ``{community_uuid, member_hub_uuids, triplet_uuids,
              summary_text, summary_embedding}``.
        source: The stable book identity.
        session_factory: The injected session factory.
    """
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
    """Upsert ``:TripletHub`` + ``:FactHub`` vertices and their edges.

    Args:
        groups: One dict per canonical assertion:
            ``{triplet_hub_uuid, subj_hub, pred_hub, obj_hub,
              triplet_uuids, fact_text, fact_embedding}``.
        source: The stable book identity.
        session_factory: The injected session factory.
    """
    if not groups:
        return

    # Hub rows (uuid, source only)
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
