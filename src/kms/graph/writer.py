"""
Persist the structural node stream and the entity overlay into Neo4j — the I/O half of the graph.

``persist_nodes`` writes the ``:Node`` layer: one ``:Source`` vertex for the book, one vertex per
``models.ASTNode`` (base ``:Node`` label + its per-type label), all MERGEd on their deterministic uuids so
re-running a book is idempotent, then wires them up — ``(:Source)-[:HEAD]->`` the first node and
``:NEXT`` edges threading the rest in document order so the stream hangs off the source and is
walkable in Cypher. ``persist_entities`` writes the ``:Entity`` overlay on top: one vertex per
Definition / Theorem / Problem (base ``:Entity`` label + its per-type label), rooted under the book
via ``:HAS_ENTITY`` and linked back to the structural chunks it was built from via ``:DERIVED_FROM``.
``persist_procedures`` writes the procedural layer: one ``:Procedure`` per derivation hung off its
entity via ``:HAS_PROCEDURE``, and one ``:Event`` per step threaded ``:FIRST``/``:THEN`` (see
``graph.procedures``). ``persist_concepts`` writes the concept layer: a global ``:Concept`` per distinct
concept and an ``:INSTANCE_OF`` edge from every entity and every event that instantiates it (see
``graph.concepts``). ``persist_dependencies`` adds the concept-level prerequisite edges,
``(:Concept)-[:DEPENDS_ON]->(:Concept)`` (see ``graph.dependencies``).
``persist_references`` writes the cross-entity layer: a global ``:Entity:Canonical``
per distinct reference target, and a ``(:Entity)-[:REFERENCES {relation}]->(:Canonical)`` edge per
reference (see ``graph.references`` for why references route through canonicals). ``persist_uses`` adds
the step-level ``(:Event)-[:USES {relation}]->(:Canonical)`` edges on top (see ``graph.uses``).
``persist_realizes`` closes the loop: a ``(:Entity:Mention)-[:REALIZES]->(:Entity:Canonical)`` edge
ties each cited concept's canonical back to the in-corpus mention that defines/states it, so citations
resolve through the hub to real knowledge (see ``graph.realizes``).

Writes are batched. Cypher can't parameterize a label, so a per-type label would have to be
interpolated — which is only safe for a CLOSED vocabulary. The structural ``:Node`` layer still has
one (``models.NodeType``) and is grouped by label; every semantic layer above it now carries its type
as a *property* (``docs/GENERALIZATION.md``), which is what lets those layers write in a single
batched MERGE each with no interpolation at all. The pure planning (grouping, edge pairs, head) is
factored out and unit-tested; only the ``session.run`` calls need a live database.
"""

from collections import defaultdict
from typing import Any

from kms.core import models
from kms.graph.concepts import (
    CONCEPT_LABEL,
    concept_rows,
    entity_instance_rows,
    event_instance_rows,
)
from kms.graph.db import database, driver
from kms.graph.dependencies import dependency_rows
from kms.graph.entities import (
    CANONICAL_LABEL,
    ENTITY_LABEL,
    MENTION_LABEL,
    entity_properties,
    entity_uuid,
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
    EVENT_LABEL,
    PROCEDURE_LABEL,
    event_rows,
    first_pairs,
    has_procedure_pairs,
    procedure_rows,
    then_pairs,
)
from kms.graph.realizes import realizes_rows
from kms.graph.references import canonical_rows, reference_rows
from kms.graph.uses import uses_rows


def node_batches(
    nodes: list[models.ASTNode], source: str
) -> dict[str | None, list[dict]]:
    """Group the nodes' property maps by their per-type label, so each label is one batched
    MERGE. The ``None`` bucket holds any typeless node (base ``:Node`` label only)."""
    batches: dict[str | None, list[dict]] = defaultdict(list)
    for node in nodes:
        batches[node_label(node)].append(node_properties(node, source))
    return dict(batches)


def next_pairs(nodes: list[models.ASTNode], source: str) -> list[dict]:
    """The ``{from, to}`` uuid pairs for the ``:NEXT`` chain: consecutive nodes in the
    document-ordered stream. Empty for a stream of fewer than two nodes."""
    return [
        {'from': node_uuid(source, a.id), 'to': node_uuid(source, b.id)}
        for a, b in zip(
            nodes, nodes[1:], strict=False
        )  # deliberately uneven: consecutive pairs
    ]


def head_uuid(nodes: list[models.ASTNode], source: str) -> str | None:
    """The uuid of the stream's first node — the ``:HEAD`` the source hangs off — or None if the
    stream is empty."""
    return node_uuid(source, nodes[0].id) if nodes else None


async def persist_nodes(
    nodes: list[models.ASTNode],
    source: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Upsert the book's ``:Source`` root, its structural node stream, the ``:HEAD`` link, and the
    ``:NEXT`` chain. Idempotent: every MERGE keys on a deterministic uuid, so re-persisting the same
    ``source`` updates in place. A no-op for an empty stream. ``source`` is the book identity and
    ``metadata`` its optional attributes; every node's id must be assigned (post-flatten)."""
    if not nodes:
        return
    src = source_properties(source, metadata)
    batches = node_batches(nodes, source)
    pairs = next_pairs(nodes, source)
    head = head_uuid(nodes, source)

    async with driver().session(database=database()) as session:
        await session.run(
            f'MERGE (s:{SOURCE_LABEL} {{uuid: $uuid}}) SET s += $props',
            uuid=src['uuid'],
            props=src,
        )
        for label, rows in batches.items():
            query = f'UNWIND $rows AS row MERGE (n:{NODE_LABEL} {{uuid: row.uuid}}) SET n += row'
            if label:
                query += f' SET n:{label}'
            await session.run(query, rows=rows)

        await session.run(
            f'MATCH (s:{SOURCE_LABEL} {{uuid: $src}}), (n:{NODE_LABEL} {{uuid: $head}}) '
            f'MERGE (s)-[:HEAD]->(n)',
            src=src['uuid'],
            head=head,
        )
        if pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (a:{NODE_LABEL} {{uuid: pair.from}}), (b:{NODE_LABEL} {{uuid: pair.to}}) '
                f'MERGE (a)-[:NEXT]->(b)',
                pairs=pairs,
            )


def entity_rows(entities: list[models.Entity], source: str) -> list[dict]:
    """Every entity's property map, one flat list. Unlike the structural nodes there is no grouping
    to do: an entity's ``type`` is an open property rather than a label (see ``graph.entities``), so
    the whole overlay writes in a single batched MERGE."""
    return [entity_properties(entity, source) for entity in entities]


def member_pairs(entities: list[models.Entity], source: str) -> list[dict]:
    """The ``{entity, node}`` uuid pairs for the ``:DERIVED_FROM`` edges: one per (entity, member) so
    an entity links to every source chunk it was built from. A member id resolves to the ``:Node``
    layer's own deterministic ``node_uuid``, so the edge lands on the real provenance chunk."""
    return [
        {
            'entity': entity_uuid(source, entity.id),
            'node': node_uuid(source, member),
        }
        for entity in entities
        for member in entity.members
    ]


async def persist_entities(entities: list[models.Entity], source: str) -> None:
    """Upsert the book's ``:Entity`` overlay: one vertex per entity (base ``:Entity`` label + the
    ``:Mention`` role label, its induced ``type`` a property), rooted under the already-persisted
    ``:Source`` via ``:HAS_ENTITY``, and linked to its structural chunks via ``:DERIVED_FROM``.
    Idempotent — every MERGE keys on a deterministic uuid, so re-persisting the same ``source``
    updates in place. A no-op for an empty overlay. The ``:Source`` and ``:Node`` vertices are
    expected to already exist (the node persister runs first); the MATCHes here attach to them rather
    than creating them. Every entity's id must be assigned (post-flatten)."""
    if not entities:
        return
    rows = entity_rows(entities, source)
    pairs = member_pairs(entities, source)
    src = source_uuid(source)
    uuids = [entity_uuid(source, entity.id) for entity in entities]

    async with driver().session(database=database()) as session:
        await session.run(
            f'UNWIND $rows AS row MERGE (e:{ENTITY_LABEL} {{uuid: row.uuid}}) '
            f'SET e += row SET e:{MENTION_LABEL}',
            rows=rows,
        )
        await session.run(
            f'MATCH (s:{SOURCE_LABEL} {{uuid: $src}}) '
            f'UNWIND $uuids AS uuid '
            f'MATCH (e:{ENTITY_LABEL} {{uuid: uuid}}) '
            f'MERGE (s)-[:HAS_ENTITY]->(e)',
            src=src,
            uuids=uuids,
        )
        if pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (e:{ENTITY_LABEL} {{uuid: pair.entity}}), (n:{NODE_LABEL} {{uuid: pair.node}}) '
                f'MERGE (e)-[:DERIVED_FROM]->(n)',
                pairs=pairs,
            )


async def persist_procedures(
    entities: list[models.Entity], source: str
) -> None:
    """Upsert the procedural layer: one ``:Procedure`` per derivation (base ``:Procedure`` label, its
    ``type`` a property), hung off its entity via ``:HAS_PROCEDURE``; one ``:Event`` per step,
    threaded ``:FIRST`` from the procedure and ``:THEN`` along the steps. Idempotent — every MERGE keys
    on a deterministic uuid. A no-op when no entity carries a derivation. The citing ``:Entity`` vertices
    are expected to already exist (the entity persister writes them first); the ``:HAS_PROCEDURE`` MATCH
    attaches to them. Every entity's id must be assigned (post-flatten)."""
    procedures = procedure_rows(entities, source)
    if not procedures:
        return
    events = event_rows(entities, source)
    haspairs = has_procedure_pairs(entities, source)
    firsts = first_pairs(entities, source)
    thens = then_pairs(entities, source)

    async with driver().session(database=database()) as session:
        await session.run(
            f'UNWIND $rows AS row MERGE (p:{PROCEDURE_LABEL} {{uuid: row.uuid}}) SET p += row',
            rows=procedures,
        )
        if events:
            await session.run(
                f'UNWIND $rows AS row MERGE (e:{EVENT_LABEL} {{uuid: row.uuid}}) SET e += row',
                rows=events,
            )
        await session.run(
            f'UNWIND $pairs AS pair '
            f'MATCH (e:{ENTITY_LABEL} {{uuid: pair.entity}}), '
            f'(p:{PROCEDURE_LABEL} {{uuid: pair.procedure}}) '
            f'MERGE (e)-[:HAS_PROCEDURE]->(p)',
            pairs=haspairs,
        )
        if firsts:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (p:{PROCEDURE_LABEL} {{uuid: pair.procedure}}), '
                f'(e:{EVENT_LABEL} {{uuid: pair.event}}) '
                f'MERGE (p)-[:FIRST]->(e)',
                pairs=firsts,
            )
        if thens:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (a:{EVENT_LABEL} {{uuid: pair.from}}), (b:{EVENT_LABEL} {{uuid: pair.to}}) '
                f'MERGE (a)-[:THEN]->(b)',
                pairs=thens,
            )


async def persist_concepts(entities: list[models.Entity], source: str) -> None:
    """Upsert the concept layer: mint a global ``:Concept`` per distinct concept, then draw an
    ``:INSTANCE_OF`` edge from every entity and every procedure step that instantiates it. Idempotent
    — concepts MERGE on their deterministic global uuid and edges on the (instance, concept) pair. A
    no-op when nothing instantiates a concept. The ``:Entity`` and ``:Event`` vertices are expected to
    already exist (the entity and procedure persisters write them first)."""
    concepts = concept_rows(entities, source)
    if not concepts:
        return
    entity_instances = entity_instance_rows(entities, source)
    event_instances = event_instance_rows(entities, source)

    async with driver().session(database=database()) as session:
        await session.run(
            f'UNWIND $rows AS row MERGE (c:{CONCEPT_LABEL} {{uuid: row.uuid}}) SET c += row',
            rows=concepts,
        )
        if entity_instances:
            await session.run(
                f'UNWIND $rows AS row '
                f'MATCH (e:{ENTITY_LABEL} {{uuid: row.entity}}), '
                f'(c:{CONCEPT_LABEL} {{uuid: row.concept}}) '
                f'MERGE (e)-[:INSTANCE_OF]->(c)',
                rows=entity_instances,
            )
        if event_instances:
            await session.run(
                f'UNWIND $rows AS row '
                f'MATCH (v:{EVENT_LABEL} {{uuid: row.event}}), '
                f'(c:{CONCEPT_LABEL} {{uuid: row.concept}}) '
                f'MERGE (v)-[:INSTANCE_OF]->(c)',
                rows=event_instances,
            )


async def persist_dependencies(
    dependencies: list[models.Dependency],
) -> None:
    """Upsert the concept prerequisite layer: a ``(:Concept)-[:DEPENDS_ON {support}]->(:Concept)``
    edge per judged dependency (see ``graph.dependencies``). Idempotent — edges MERGE on the
    (dependent, prerequisite) pair, with ``support`` set on the relationship so a re-run updates it in
    place. A no-op when the dependency finder judged nothing. Both endpoints are MATCHed, never
    minted: a dependency naming a concept no entity instantiates draws no edge, so this is safe to run
    over the whole judged set."""
    rows = dependency_rows(dependencies)
    if not rows:
        return
    async with driver().session(database=database()) as session:
        await session.run(
            f'UNWIND $rows AS row '
            f'MATCH (a:{CONCEPT_LABEL} {{uuid: row.dependent}}), '
            f'(b:{CONCEPT_LABEL} {{uuid: row.prerequisite}}) '
            f'MERGE (a)-[d:DEPENDS_ON]->(b) SET d.support = row.support',
            rows=rows,
        )


async def persist_references(
    entities: list[models.Entity], source: str
) -> None:
    """Upsert the cross-entity reference layer: mint a global ``:Entity:Canonical`` per distinct
    reference target (base ``:Entity`` label + the ``:Canonical`` role label, its ``type`` a
    property), then draw a ``(:Entity)-[:REFERENCES {relation}]->(:Canonical)`` edge for each
    reference. Idempotent — canonicals MERGE on their deterministic global uuid and edges MERGE on the
    (entity, canonical) pair (the relation is set on the relationship, so a re-run updates it in
    place). A no-op when no entity carries references. The citing mention ``:Entity`` vertices are
    expected to already exist (the entity persister writes them first); the MATCH attaches to them."""
    rows = reference_rows(entities, source)
    if not rows:
        return
    canonicals = canonical_rows(entities)

    async with driver().session(database=database()) as session:
        await session.run(
            f'UNWIND $rows AS row MERGE (c:{ENTITY_LABEL} {{uuid: row.uuid}}) '
            f'SET c += row SET c:{CANONICAL_LABEL}',
            rows=canonicals,
        )
        await session.run(
            f'UNWIND $rows AS row '
            f'MATCH (e:{ENTITY_LABEL} {{uuid: row.entity}}), '
            f'(c:{CANONICAL_LABEL} {{uuid: row.canonical}}) '
            f'MERGE (e)-[ref:REFERENCES]->(c) SET ref.relation = row.relation',
            rows=rows,
        )


async def persist_uses(entities: list[models.Entity], source: str) -> None:
    """Upsert the step-level ``:USES`` layer: for each procedure step that mentions a reference target,
    draw a ``(:Event)-[:USES {relation}]->(:Entity:Canonical)`` edge (the finer complement of the
    entity-level ``:REFERENCES`` rollup; see ``graph.uses``). Idempotent — edges MERGE on the (event,
    canonical) pair, relation set on the relationship. A no-op when nothing matches. The ``:Event`` and
    ``:Canonical`` vertices are expected to already exist (the procedure and reference persisters run
    first)."""
    rows = uses_rows(entities, source)
    if not rows:
        return
    async with driver().session(database=database()) as session:
        await session.run(
            f'UNWIND $rows AS row '
            f'MATCH (v:{EVENT_LABEL} {{uuid: row.event}}), '
            f'(c:{CANONICAL_LABEL} {{uuid: row.canonical}}) '
            f'MERGE (v)-[u:USES]->(c) SET u.relation = row.relation',
            rows=rows,
        )


async def persist_realizes(entities: list[models.Entity], source: str) -> None:
    """Upsert the ``:REALIZES`` identity edges: for each typed, titled mention whose
    ``(type, title)`` names an already-persisted canonical, draw
    ``(:Entity:Mention)-[:REALIZES]->(:Entity:Canonical)`` (see ``graph.realizes``). Idempotent — edges
    MERGE on the (mention, canonical) pair. The ``MATCH`` on the canonical does the filtering: a mention
    whose title was never cited finds no canonical and gets no edge, so this is safe to run over the
    whole overlay. A no-op when no typed, titled mention exists. The mention ``:Entity`` and the
    ``:Canonical`` vertices are expected to already exist (the entity and reference persisters run
    first)."""
    rows = realizes_rows(entities, source)
    if not rows:
        return
    async with driver().session(database=database()) as session:
        await session.run(
            f'UNWIND $rows AS row '
            f'MATCH (m:{ENTITY_LABEL} {{uuid: row.mention}}), '
            f'(c:{CANONICAL_LABEL} {{uuid: row.canonical}}) '
            f'MERGE (m)-[:REALIZES]->(c)',
            rows=rows,
        )
