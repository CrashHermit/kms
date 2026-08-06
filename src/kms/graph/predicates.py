"""
Graph representation of the predicate layer — ``:Predicate`` vertices carrying
the description of one triplet's predicate.

The entity enricher walks the provenance nodes and writes one description per
distinct predicate per node. This module maps each predicate occurrence onto
its Neo4j form as a ``:Predicate`` vertex: pure mapping, free of the neo4j
driver (the driver lives in ``graph.db``, the writes in ``graph.writer``).

Representation: every predicate carries the ``:Predicate`` label with the
predicate text and — when written — its ``description``. The triplet hub whose
predicate this describes points at it via
``(:Triplet)-[:HAS_PREDICATE]->(:Predicate)``. The predicate text also lives
on the triplet (the verbatim record); the ``:Predicate`` vertex is the
DESCRIBED component.

Predicates are per-triplet: a ``:Predicate`` vertex is always written for
every triplet occurrence (regardless of whether the enricher described it).
The same predicate text appearing in several triplets at a node gets one
vertex per triplet — the description, when present, is deliberately
duplicated rather than shared, until a future canonicalization pass merges
them.

Identity: deterministic uuid5 over the source fact's uuid plus the predicate
text and the anchor node — disjoint from every other uuid (see
``graph.triplets.triplet_uuid`` for the hub the predicate hangs off).
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes
from kms.graph.facts import fact_uuid

PREDICATE_LABEL = 'Predicate'


def predicate_uuid(
    source: str,
    node_id: int,
    fact_uuid_val: str,
    subject: str,
    predicate: str,
    object: str,
) -> str:
    """Stable, deterministic vertex key for one per-triplet predicate.

    The subject and object are part of the key so each triplet gets its OWN
    predicate vertex — the same predicate text in two triplets of one fact
    is deliberately duplicated, never shared, until a future pass merges
    them.

    Args:
        source: The stable book identity.
        node_id: The anchor node's id in the flattened stream.
        fact_uuid_val: The source fact's uuid (see ``facts.fact_uuid``).
        subject: The triplet's subject text.
        predicate: The predicate text.
        object: The triplet's object text.

    Returns:
        The predicate's hex uuid, disjoint from every other uuid.
    """
    return uuid5(
        NAMESPACE_URL,
        f'{source}#predicate#{node_id}#{fact_uuid_val}'
        f'#{subject}#{predicate}#{object}',
    ).hex


def predicate_properties(
    source: str,
    node_id: int,
    fact_uuid_val: str,
    subject: str,
    predicate: str,
    object: str,
    description: str | None = None,
    embedding: list[float] | None = None,
) -> dict:
    """The Neo4j property map for one predicate occurrence.

    Args:
        source: The stable book identity.
        node_id: The anchor node this occurrence was written for.
        fact_uuid_val: The source fact's uuid, anchoring the predicate's
            identity to exactly one fact.
        subject: The triplet's subject text.
        predicate: The predicate text.
        object: The triplet's object text.
        description: This node's enricher description of the predicate, if
            any.
        embedding: The embedding vector for ``predicate + description``,
            if computed.

    Returns:
        The property map, with None values omitted.
    """
    properties = {
        'uuid': predicate_uuid(
            source, node_id, fact_uuid_val, subject, predicate, object
        ),
        'source': nodes.source_uuid(source),
        'node_id': node_id,
        'predicate': predicate,
        'description': description,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def _description_for(
    node_id: int,
    predicate: str,
    node_predicate_descriptions: dict[int, list[dict]],
) -> str | None:
    """The description the enricher wrote for *predicate* at *node_id*.

    Args:
        node_id: The anchor node.
        predicate: The predicate text.
        node_predicate_descriptions: The enricher's per-node mapping: node id
            to a list of ``{predicate, description}`` dicts.

    Returns:
        The matching description, or None.
    """
    for entry in node_predicate_descriptions.get(node_id, []):
        if entry['predicate'] == predicate:
            return entry.get('description')
    return None


def _embedding_for(
    node_id: int,
    predicate: str,
    node_predicate_descriptions: dict[int, list[dict]],
) -> list[float] | None:
    """The embedding computed for *predicate* at *node_id*.

    Args:
        node_id: The anchor node.
        predicate: The predicate text.
        node_predicate_descriptions: The per-node mapping, where each entry
            may carry an ``embedding`` field.

    Returns:
        The embedding vector, or None.
    """
    for entry in node_predicate_descriptions.get(node_id, []):
        if entry['predicate'] == predicate:
            return entry.get('embedding')
    return None


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


def predicate_rows(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
    node_predicate_descriptions: dict[int, list[dict]],
) -> list[dict]:
    """Every predicate's property map, one flat list.

    One row per (node, triplet), iterating every node its source fact
    touches. Each row carries that node's own description from the enricher's
    per-node predicate descriptions.

    Args:
        triplets: The triplets, in document order (grouped by source fact).
        facts: The atomic facts, in document order, aligned with
            ``triplets`` by ``fact_index``.
        source: The stable book identity.
        node_predicate_descriptions: The enricher's per-node predicate
            descriptions.

    Returns:
        One property map per (node, triplet), in document order.
    """
    rows: list[dict] = []
    for triplet in triplets:
        fi = triplet.fact_index
        fact_uuid_val = fact_uuid(source, facts[fi].node_ids, fi)
        for node_id in _triplet_nodes(triplet, facts):
            description = _description_for(
                node_id, triplet.predicate, node_predicate_descriptions
            )
            embedding = _embedding_for(
                node_id, triplet.predicate, node_predicate_descriptions
            )
            rows.append(
                predicate_properties(
                    source,
                    node_id,
                    fact_uuid_val,
                    triplet.subject,
                    triplet.predicate,
                    triplet.object,
                    description,
                    embedding,
                )
            )
    return rows


def has_predicate_pairs(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
) -> list[dict]:
    """The ``{triplet, predicate}`` uuid pairs for the ``:HAS_PREDICATE``
    edges.

    Each triplet hub points at the described predicate component it carries.
    One pair per (node, triplet).

    Args:
        triplets: The triplets, in document order.
        facts: The atomic facts, in document order.
        source: The stable book identity.

    Returns:
        One ``{triplet, predicate}`` per (node, triplet).
    """
    from kms.graph.triplets import triplet_uuid

    pairs: list[dict] = []
    for triplet in triplets:
        fi = triplet.fact_index
        fact_uuid_val = fact_uuid(source, facts[fi].node_ids, fi)
        for node_id in _triplet_nodes(triplet, facts):
            pairs.append(
                {
                    'triplet': triplet_uuid(
                        source,
                        node_id,
                        fact_uuid_val,
                        triplet.subject,
                        triplet.predicate,
                        triplet.object,
                    ),
                    'predicate': predicate_uuid(
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
