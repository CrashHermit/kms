"""
Persist the structural node stream and the entity overlay into Neo4j — the I/O half of the graph.

``persist_nodes`` writes the ``:Node`` layer: one ``:Source`` vertex for the book, one vertex per
``ASTNode`` (base ``:Node`` label + its per-type label), all MERGEd on their deterministic uuids so
re-running a book is idempotent, then wires them up — ``(:Source)-[:HEAD]->`` the first node and
``:NEXT`` edges threading the rest in document order so the stream hangs off the source and is
walkable in Cypher. ``persist_entities`` writes the ``:Entity`` overlay on top: one vertex per
Definition / Theorem / Problem (base ``:Entity`` label + its per-type label), rooted under the book
via ``:HAS_ENTITY`` and linked back to the structural chunks it was built from via ``:DERIVED_FROM``.
``persist_procedures`` writes the procedural layer: one ``:Procedure`` per proof/solution hung off its
entity via ``:HAS_PROCEDURE``, and one ``:Event`` per proof step threaded ``:FIRST``/``:THEN`` (see
``graph.procedures``). ``persist_concepts`` writes the concept layer: a global ``:Concept`` per distinct
concept and a ``(:Entity)-[:INSTANCE_OF]->(:Concept)`` edge per entity concept (see ``graph.concepts``).
``persist_references`` writes the cross-entity layer: a global ``:Entity:Canonical``
per distinct reference target, and a ``(:Entity)-[:REFERENCES {tactic}]->(:Canonical)`` edge per reference
(see ``graph.references`` for why references route through canonicals). ``persist_uses`` adds the
step-level ``(:Event)-[:USES {tactic}]->(:Canonical)`` edges on top (see ``graph.uses``).

Writes are batched: Cypher can't parameterize a label, but the label comes from a closed enum
(``NodeType`` / ``EntityType``), so grouping by label and interpolating it is safe and turns the
whole stream into one MERGE per label plus a couple for the source/edges — no per-vertex
round-trips. The pure planning (grouping, edge pairs, head) is factored out and unit-tested; only the
``session.run`` calls need a live database.
"""

from collections import defaultdict
from typing import Any

from kms.core.models import ASTNode, Entity
from kms.graph.concepts import CONCEPT_LABEL, concept_batches, instance_rows
from kms.graph.db import database, driver
from kms.graph.entities import (
    CANONICAL_LABEL,
    ENTITY_LABEL,
    MENTION_LABEL,
    entity_label,
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
    procedure_batches,
    then_pairs,
)
from kms.graph.references import canonical_batches, reference_rows
from kms.graph.uses import uses_rows


def node_batches(nodes: list[ASTNode], source: str) -> dict[str | None, list[dict]]:
    """Group the nodes' property maps by their per-type label, so each label is one batched
    MERGE. The ``None`` bucket holds any typeless node (base ``:Node`` label only)."""
    batches: dict[str | None, list[dict]] = defaultdict(list)
    for node in nodes:
        batches[node_label(node)].append(node_properties(node, source))
    return dict(batches)


def next_pairs(nodes: list[ASTNode], source: str) -> list[dict]:
    """The ``{from, to}`` uuid pairs for the ``:NEXT`` chain: consecutive nodes in the
    document-ordered stream. Empty for a stream of fewer than two nodes."""
    return [
        {"from": node_uuid(source, a.id), "to": node_uuid(source, b.id)}
        for a, b in zip(nodes, nodes[1:], strict=False)  # deliberately uneven: consecutive pairs
    ]


def head_uuid(nodes: list[ASTNode], source: str) -> str | None:
    """The uuid of the stream's first node — the ``:HEAD`` the source hangs off — or None if the
    stream is empty."""
    return node_uuid(source, nodes[0].id) if nodes else None


async def persist_nodes(
    nodes: list[ASTNode], source: str, metadata: dict[str, Any] | None = None
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
            f"MERGE (s:{SOURCE_LABEL} {{uuid: $uuid}}) SET s += $props",
            uuid=src["uuid"],
            props=src,
        )
        for label, rows in batches.items():
            query = f"UNWIND $rows AS row MERGE (n:{NODE_LABEL} {{uuid: row.uuid}}) SET n += row"
            if label:
                query += f" SET n:{label}"
            await session.run(query, rows=rows)

        await session.run(
            f"MATCH (s:{SOURCE_LABEL} {{uuid: $src}}), (n:{NODE_LABEL} {{uuid: $head}}) "
            f"MERGE (s)-[:HEAD]->(n)",
            src=src["uuid"],
            head=head,
        )
        if pairs:
            await session.run(
                f"UNWIND $pairs AS pair "
                f"MATCH (a:{NODE_LABEL} {{uuid: pair.from}}), (b:{NODE_LABEL} {{uuid: pair.to}}) "
                f"MERGE (a)-[:NEXT]->(b)",
                pairs=pairs,
            )


def entity_batches(entities: list[Entity], source: str) -> dict[str, list[dict]]:
    """Group the entities' property maps by their per-type label, so each label is one batched
    MERGE. Every entity is typed, so there is no ``None`` bucket (unlike ``node_batches``)."""
    batches: dict[str, list[dict]] = defaultdict(list)
    for entity in entities:
        batches[entity_label(entity)].append(entity_properties(entity, source))
    return dict(batches)


def member_pairs(entities: list[Entity], source: str) -> list[dict]:
    """The ``{entity, node}`` uuid pairs for the ``:DERIVED_FROM`` edges: one per (entity, member) so
    an entity links to every source chunk it was built from. A member id resolves to the ``:Node``
    layer's own deterministic ``node_uuid``, so the edge lands on the real provenance chunk."""
    return [
        {"entity": entity_uuid(source, entity.id), "node": node_uuid(source, member)}
        for entity in entities
        for member in entity.members
    ]


async def persist_entities(entities: list[Entity], source: str) -> None:
    """Upsert the book's ``:Entity`` overlay: one vertex per entity (base ``:Entity`` label + its
    per-type label), rooted under the already-persisted ``:Source`` via ``:HAS_ENTITY``, and linked
    to its structural chunks via ``:DERIVED_FROM``. Idempotent — every MERGE keys on a deterministic
    uuid, so re-persisting the same ``source`` updates in place. A no-op for an empty overlay. The
    ``:Source`` and ``:Node`` vertices are expected to already exist (the node persister runs first);
    the MATCHes here attach to them rather than creating them. Every entity's id must be assigned
    (post-flatten)."""
    if not entities:
        return
    batches = entity_batches(entities, source)
    pairs = member_pairs(entities, source)
    src = source_uuid(source)
    uuids = [entity_uuid(source, entity.id) for entity in entities]

    async with driver().session(database=database()) as session:
        for label, rows in batches.items():
            await session.run(
                f"UNWIND $rows AS row MERGE (e:{ENTITY_LABEL} {{uuid: row.uuid}}) "
                f"SET e += row SET e:{label} SET e:{MENTION_LABEL}",
                rows=rows,
            )
        await session.run(
            f"MATCH (s:{SOURCE_LABEL} {{uuid: $src}}) "
            f"UNWIND $uuids AS uuid "
            f"MATCH (e:{ENTITY_LABEL} {{uuid: uuid}}) "
            f"MERGE (s)-[:HAS_ENTITY]->(e)",
            src=src,
            uuids=uuids,
        )
        if pairs:
            await session.run(
                f"UNWIND $pairs AS pair "
                f"MATCH (e:{ENTITY_LABEL} {{uuid: pair.entity}}), (n:{NODE_LABEL} {{uuid: pair.node}}) "
                f"MERGE (e)-[:DERIVED_FROM]->(n)",
                pairs=pairs,
            )


async def persist_procedures(entities: list[Entity], source: str) -> None:
    """Upsert the procedural layer: one ``:Procedure`` per proof/solution (base ``:Procedure`` label +
    its per-kind label), hung off its entity via ``:HAS_PROCEDURE``; one ``:Event`` per proof step,
    threaded ``:FIRST`` from the procedure and ``:THEN`` along the steps. Idempotent — every MERGE keys
    on a deterministic uuid. A no-op when no entity carries a derivation. The citing ``:Entity`` vertices
    are expected to already exist (the entity persister writes them first); the ``:HAS_PROCEDURE`` MATCH
    attaches to them. Every entity's id must be assigned (post-flatten)."""
    proc_batches = procedure_batches(entities, source)
    if not proc_batches:
        return
    events = event_rows(entities, source)
    haspairs = has_procedure_pairs(entities, source)
    firsts = first_pairs(entities, source)
    thens = then_pairs(entities, source)

    async with driver().session(database=database()) as session:
        for label, rows in proc_batches.items():
            await session.run(
                f"UNWIND $rows AS row MERGE (p:{PROCEDURE_LABEL} {{uuid: row.uuid}}) "
                f"SET p += row SET p:{label}",
                rows=rows,
            )
        if events:
            await session.run(
                f"UNWIND $rows AS row MERGE (e:{EVENT_LABEL} {{uuid: row.uuid}}) SET e += row",
                rows=events,
            )
        await session.run(
            f"UNWIND $pairs AS pair "
            f"MATCH (e:{ENTITY_LABEL} {{uuid: pair.entity}}), "
            f"(p:{PROCEDURE_LABEL} {{uuid: pair.procedure}}) "
            f"MERGE (e)-[:HAS_PROCEDURE]->(p)",
            pairs=haspairs,
        )
        if firsts:
            await session.run(
                f"UNWIND $pairs AS pair "
                f"MATCH (p:{PROCEDURE_LABEL} {{uuid: pair.procedure}}), "
                f"(e:{EVENT_LABEL} {{uuid: pair.event}}) "
                f"MERGE (p)-[:FIRST]->(e)",
                pairs=firsts,
            )
        if thens:
            await session.run(
                f"UNWIND $pairs AS pair "
                f"MATCH (a:{EVENT_LABEL} {{uuid: pair.from}}), (b:{EVENT_LABEL} {{uuid: pair.to}}) "
                f"MERGE (a)-[:THEN]->(b)",
                pairs=thens,
            )


async def persist_concepts(entities: list[Entity], source: str) -> None:
    """Upsert the concept layer: mint a global ``:Concept`` per distinct concept (base ``:Concept``
    label + its per-type label), then draw a ``(:Entity)-[:INSTANCE_OF]->(:Concept)`` edge per entity
    concept. Idempotent — concepts MERGE on their deterministic global uuid and edges on the (entity,
    concept) pair. A no-op when no entity instantiates a concept. The citing mention ``:Entity``
    vertices are expected to already exist (the entity persister writes them first)."""
    rows = instance_rows(entities, source)
    if not rows:
        return
    batches = concept_batches(entities)

    async with driver().session(database=database()) as session:
        for label, concepts in batches.items():
            await session.run(
                f"UNWIND $rows AS row MERGE (c:{CONCEPT_LABEL} {{uuid: row.uuid}}) "
                f"SET c += row SET c:{label}",
                rows=concepts,
            )
        await session.run(
            f"UNWIND $rows AS row "
            f"MATCH (e:{ENTITY_LABEL} {{uuid: row.entity}}), "
            f"(c:{CONCEPT_LABEL} {{uuid: row.concept}}) "
            f"MERGE (e)-[:INSTANCE_OF]->(c)",
            rows=rows,
        )


async def persist_references(entities: list[Entity], source: str) -> None:
    """Upsert the cross-entity reference layer: mint a global ``:Entity:Canonical`` per distinct
    reference target (base ``:Entity`` label + the ``:Canonical`` role label + its per-type label),
    then draw a ``(:Entity)-[:REFERENCES {tactic}]->(:Canonical)`` edge for each reference. Idempotent —
    canonicals MERGE on their deterministic global uuid and edges MERGE on the (entity, canonical) pair
    (the tactic is set on the relationship, so a re-run updates it in place). A no-op when no entity
    carries references. The citing mention ``:Entity`` vertices are expected to already exist (the entity
    persister writes them first); the MATCH attaches to them."""
    rows = reference_rows(entities, source)
    if not rows:
        return
    batches = canonical_batches(entities)

    async with driver().session(database=database()) as session:
        for label, canonicals in batches.items():
            await session.run(
                f"UNWIND $rows AS row MERGE (c:{ENTITY_LABEL} {{uuid: row.uuid}}) "
                f"SET c += row SET c:{CANONICAL_LABEL} SET c:{label}",
                rows=canonicals,
            )
        await session.run(
            f"UNWIND $rows AS row "
            f"MATCH (e:{ENTITY_LABEL} {{uuid: row.entity}}), "
            f"(c:{CANONICAL_LABEL} {{uuid: row.canonical}}) "
            f"MERGE (e)-[ref:REFERENCES]->(c) SET ref.tactic = row.tactic",
            rows=rows,
        )


async def persist_uses(entities: list[Entity], source: str) -> None:
    """Upsert the step-level ``:USES`` layer: for each proof step that mentions a reference target, draw
    a ``(:Event)-[:USES {tactic}]->(:Entity:Canonical)`` edge (the finer complement of the entity-level
    ``:REFERENCES`` rollup; see ``graph.uses``). Idempotent — edges MERGE on the (event, canonical) pair,
    tactic set on the relationship. A no-op when nothing matches. The ``:Event`` and ``:Canonical``
    vertices are expected to already exist (the procedure and reference persisters run first)."""
    rows = uses_rows(entities, source)
    if not rows:
        return
    async with driver().session(database=database()) as session:
        await session.run(
            f"UNWIND $rows AS row "
            f"MATCH (v:{EVENT_LABEL} {{uuid: row.event}}), "
            f"(c:{CANONICAL_LABEL} {{uuid: row.canonical}}) "
            f"MERGE (v)-[u:USES]->(c) SET u.tactic = row.tactic",
            rows=rows,
        )
