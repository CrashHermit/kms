"""
Graph representation of the triplet layer — ``:Triplet`` hubs as empty
connectors, written separately per node occurrence.

The triplet extractor decomposes each atomic fact into (subject, predicate,
object) triplets. This module maps one ``models.Triplet`` onto its Neo4j
form as a ``:Triplet`` hub: pure mapping, free of the neo4j driver (the
driver lives in ``graph.db``, the writes in ``graph.writer``).

Representation: the ``:Triplet`` hub is empty — it carries no verbatim
strings. The triplet's content is expressed through its three edges:
``(:Triplet)-[:HAS_SUBJECT]->(:Entity)``,
``(:Triplet)-[:HAS_PREDICATE]->(:Predicate)``, and
``(:Triplet)-[:HAS_OBJECT]->(:Entity)``. The verbatim subject/object strings
live on the ``:Entity`` vertices; the verbatim predicate lives on the
``:Predicate`` vertex. Provenance: the source ``:Fact`` points at each
triplet it produced via ``(:Fact)-[:YIELDS]->(:Triplet)``.

The hub is written SEPARATELY per node occurrence — one vertex per (node,
triplet) — because each node's enricher descriptions must stay with its own
predicate and entity endpoints.

Identity: deterministic uuid5 over ``(source, node_id, fact_uuid, subject,
predicate, object)`` — the anchor node plus the source fact's identity plus
the assertion's content — disjoint from every other uuid. Including the node
keeps the per-node occurrences separate; including the fact means a fact
changing identity gives its triplets fresh uuids instead of gathering
``:YIELDS`` edges from old and new facts onto one node.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes
from kms.graph.entities import entity_uuid
from kms.graph.facts import fact_uuid

TRIPLET_LABEL = 'Triplet'


def triplet_uuid(
    source: str,
    node_id: int,
    fact_uuid_val: str,
    subject: str,
    predicate: str,
    object: str,
) -> str:
    """Stable, deterministic vertex key for one (node, triplet) hub.

    Args:
        source: The stable book identity.
        node_id: The anchor node's id in the flattened stream.
        fact_uuid_val: The source fact's uuid (see ``facts.fact_uuid``).
        subject: The triplet's subject text.
        predicate: The triplet's predicate text.
        object: The triplet's object text.

    Returns:
        The triplet's hex uuid, disjoint from every other uuid.
    """
    return uuid5(
        NAMESPACE_URL,
        f'{source}#triplet#{node_id}#{fact_uuid_val}'
        f'#{subject}#{predicate}#{object}',
    ).hex


def triplet_properties(
    triplet: models.Triplet,
    source: str,
    node_id: int,
    fact_uuid_val: str,
) -> dict:
    """The Neo4j property map for one triplet occurrence.

    The triplet hub is empty — it carries no verbatim strings. The
    subject, predicate, and object live on the :Entity and :Predicate
    vertices reached through the hub's edges.

    Args:
        triplet: The triplet to map.
        source: The stable book identity.
        node_id: The anchor node this occurrence was written for.
        fact_uuid_val: The source fact's uuid, anchoring the triplet's
            identity to exactly one fact.

    Returns:
        The property map, with None values omitted.
    """
    properties = {
        'uuid': triplet_uuid(
            source,
            node_id,
            fact_uuid_val,
            triplet.subject,
            triplet.predicate,
            triplet.object,
        ),
        'source': nodes.source_uuid(source),
        'node_id': node_id,
        'fact_index': triplet.fact_index,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def _triplet_nodes(
    triplet: models.Triplet, facts: list[models.AtomicFact]
) -> list[int]:
    """Every provenance node the triplet's source fact touches.

    Args:
        triplet: The triplet.
        facts: The atomic facts, in document order, aligned with
            ``triplets`` by ``fact_index``.

    Returns:
        The fact's ``node_ids``, or an empty list when none.
    """
    fact = facts[triplet.fact_index]
    return list(fact.node_ids)


def triplet_rows(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
) -> list[dict]:
    """Every triplet's property map, one flat list.

    Each triplet is written SEPARATELY per node occurrence: one row per
    (node, triplet), iterating every node its source fact touches.

    Args:
        triplets: The triplets, in document order (grouped by source fact).
        facts: The atomic facts, in document order, aligned with
            ``triplets`` by ``fact_index``.
        source: The stable book identity.

    Returns:
        One property map per (node, triplet), in document order.
    """
    rows: list[dict] = []
    for triplet in triplets:
        fi = triplet.fact_index
        fact_uuid_val = fact_uuid(source, facts[fi].node_ids, fi)
        for node_id in _triplet_nodes(triplet, facts):
            rows.append(
                triplet_properties(triplet, source, node_id, fact_uuid_val)
            )
    return rows


def yields_pairs(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
) -> list[dict]:
    """The ``{fact, triplet}`` uuid pairs for ``(:Fact)-[:YIELDS]->(:Triplet)``
    edges.

    Each source fact points at every triplet it produced via ``:YIELDS`` —
    the same provenance → construct direction as
    ``(:Node)-[:EVIDENCE_FOR]->(:Fact)``. One pair per (node, triplet).

    Args:
        triplets: The triplets, in document order.
        facts: The atomic facts, in document order.
        source: The stable book identity.

    Returns:
        One ``{fact, triplet}`` per (node, triplet).
    """
    pairs: list[dict] = []
    for triplet in triplets:
        fi = triplet.fact_index
        fact_uuid_val = fact_uuid(source, facts[fi].node_ids, fi)
        for node_id in _triplet_nodes(triplet, facts):
            pairs.append(
                {
                    'fact': fact_uuid_val,
                    'triplet': triplet_uuid(
                        source,
                        node_id,
                        fact_uuid_val,
                        triplet.subject,
                        triplet.predicate,
                        triplet.object,
                    ),
                }
            )
    return pairs


def _role_pairs(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
    *,
    role: str,
) -> list[dict]:
    """The ``{triplet, entity}`` pairs for one role edge.

    One endpoint per (node, triplet) — every triplet always has both a
    subject and an object entity, regardless of whether the enricher
    described them.

    Args:
        triplets: The triplets, in document order.
        facts: The atomic facts, in document order.
        source: The stable book identity.
        role: ``'subject'`` or ``'object'``.

    Returns:
        One ``{triplet, entity}`` per (node, triplet).
    """
    pairs: list[dict] = []
    for triplet in triplets:
        fi = triplet.fact_index
        fact_uuid_val = fact_uuid(source, facts[fi].node_ids, fi)
        for node_id in _triplet_nodes(triplet, facts):
            triplet_uuid_val = triplet_uuid(
                source,
                node_id,
                fact_uuid_val,
                triplet.subject,
                triplet.predicate,
                triplet.object,
            )
            pairs.append(
                {
                    'triplet': triplet_uuid_val,
                    'entity': entity_uuid(
                        source, node_id, triplet_uuid_val, role
                    ),
                }
            )
    return pairs


def has_subject_pairs(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
) -> list[dict]:
    """The ``{triplet, entity}`` uuid pairs for the ``:HAS_SUBJECT`` edges.

    Args:
        triplets: The triplets, in document order.
        facts: The atomic facts, in document order.
        source: The stable book identity.

    Returns:
        One ``{triplet, entity}`` per (node, triplet).
    """
    return _role_pairs(
        triplets,
        facts,
        source,
        role='subject',
    )


def has_object_pairs(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
) -> list[dict]:
    """The ``{triplet, entity}`` uuid pairs for the ``:HAS_OBJECT`` edges.

    Args:
        triplets: The triplets, in document order.
        facts: The atomic facts, in document order.
        source: The stable book identity.

    Returns:
        One ``{triplet, entity}`` per (node, triplet).
    """
    return _role_pairs(
        triplets,
        facts,
        source,
        role='object',
    )
