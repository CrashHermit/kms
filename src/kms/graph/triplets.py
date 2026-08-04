"""
Graph representation of the triplet layer — ``:Triplet`` nodes with
``:YIELDS`` edges from their source ``:Fact``.

The triplet extractor decomposes each atomic fact into (subject, predicate,
object) triplets. This module maps one ``models.Triplet`` onto its Neo4j
form: pure mapping, free of the neo4j driver (the driver lives in
``graph.db``, the writes in ``graph.writer``).

Representation: every triplet carries the ``:Triplet`` label with its
subject, predicate, and object text. Provenance: the source ``:Fact``
points at each triplet it produced via ``(:Fact)-[:YIELDS]->(:Triplet)``.

Identity: deterministic uuid5 over ``(source, fact_index, sub_index)`` —
the source fact's document-order position plus the triplet's position
within that fact — disjoint from every other uuid.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import facts, nodes

TRIPLET_LABEL = 'Triplet'


def triplet_uuid(source: str, fact_index: int, sub_index: int) -> str:
    """Stable, deterministic vertex key for a triplet.

    Args:
        source: The stable book identity.
        fact_index: The source fact's position in the document-order list.
        sub_index: The triplet's 0-based position within that fact.

    Returns:
        The triplet's hex uuid, disjoint from every other uuid.
    """
    return uuid5(
        NAMESPACE_URL, f'{source}#triplet#{fact_index}#{sub_index}'
    ).hex


def triplet_properties(
    triplet: models.Triplet, source: str, sub_index: int
) -> dict:
    """The Neo4j property map for one triplet.

    Args:
        triplet: The triplet to map.
        source: The stable book identity.
        sub_index: The triplet's 0-based position within its source fact.

    Returns:
        The property map, with None values omitted.
    """
    props = {
        'uuid': triplet_uuid(source, triplet.fact_index, sub_index),
        'source': nodes.source_uuid(source),
        'subject': triplet.subject,
        'predicate': triplet.predicate,
        'object': triplet.object,
        'fact_index': triplet.fact_index,
    }
    return {key: value for key, value in props.items() if value is not None}


def triplet_rows(
    triplets: list[models.Triplet], source: str
) -> list[dict]:
    """Every triplet's property map, one flat list.

    Each triplet is assigned a ``sub_index``: its 0-based position
    within its source fact (derived by counting triplets per fact_index
    in document order).

    Args:
        triplets: The triplets, in document order (grouped by source fact).
        source: The stable book identity.

    Returns:
        One property map per triplet.
    """
    rows: list[dict] = []
    sub_by_fact: dict[int, int] = {}
    for triplet in triplets:
        fi = triplet.fact_index
        sub = sub_by_fact.get(fi, 0)
        rows.append(triplet_properties(triplet, source, sub))
        sub_by_fact[fi] = sub + 1
    return rows


def yields_pairs(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
) -> list[dict]:
    """The ``{fact, triplet}`` uuid pairs for ``(:Fact)-[:YIELDS]->(:Triplet)``
    edges.

    Each source fact points at every triplet it produced via
    ``:YIELDS`` — the same provenance → construct direction as
    ``(:Node)-[:EVIDENCE_FOR]->(:Fact)``.

    Args:
        triplets: The triplets, in document order.
        facts: The atomic facts, in document order.
        source: The stable book identity.

    Returns:
        One ``{fact, triplet}`` per triplet.
    """
    sub_by_fact: dict[int, int] = {}
    pairs: list[dict] = []
    for triplet in triplets:
        fi = triplet.fact_index
        sub = sub_by_fact.get(fi, 0)
        fact_uuid = facts.fact_uuid(source, facts[fi].node_ids, fi)
        pairs.append(
            {
                'fact': fact_uuid,
                'triplet': triplet_uuid(source, fi, sub),
            }
        )
        sub_by_fact[fi] = sub + 1
    return pairs
