"""
Graph representation of the fact layer — ``:Fact`` nodes with evidence edges.

The atomic fact pass decomposes the node stream into atomic facts. This
module maps one
``models.AtomicFact`` onto its Neo4j form: pure mapping, free of the neo4j
driver (the driver lives in ``graph.db``, the writes in ``graph.writer``).

Representation: every fact carries the ``:Fact`` label with its text.
Provenance: each provenance node a fact draws on points at it via
``(:Node)-[:EVIDENCE_FOR]->(:Fact)``, the same raw-material → construct
anchor the statement/procedure tiers use; a fact spanning several nodes
carries one edge per node.

Identity: deterministic uuid5 over ``(source, node_ids, index)`` — the
ordered node ids plus the fact's position in the flat document-order list —
disjoint from every other uuid via the ``fact#`` segment.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes

FACT_LABEL = 'Fact'


def fact_uuid(source: str, node_ids: list[int], index: int) -> str:
    """Stable, deterministic vertex key for an atomic fact.

    Args:
        source: The stable book identity.
        node_ids: The provenance nodes the fact was drawn from, in order.
        index: The fact's position in the flat document-order list.

    Returns:
        The fact's hex uuid, disjoint from every other uuid.
    """
    return uuid5(
        NAMESPACE_URL, f'{source}#fact#{nodes.block_key(node_ids)}#{index}'
    ).hex


def fact_properties(fact: models.AtomicFact, source: str, index: int) -> dict:
    """The Neo4j property map for one atomic fact.

    Args:
        fact: The fact to map.
        source: The stable book identity.
        index: The fact's position in the flat document-order list.

    Returns:
        The property map, with None values omitted.
    """
    properties = {
        'uuid': fact_uuid(source, fact.node_ids, index),
        'source': nodes.source_uuid(source),
        'text': fact.text,
        'index': index,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def fact_rows(facts: list[models.AtomicFact], source: str) -> list[dict]:
    """Every fact's property map, one flat list.

    Args:
        facts: The atomic facts, in document order.
        source: The stable book identity.

    Returns:
        One property map per fact.
    """
    return [
        fact_properties(fact, source, index) for index, fact in enumerate(facts)
    ]


def evidence_pairs(facts: list[models.AtomicFact], source: str) -> list[dict]:
    """The ``{node, fact}`` uuid pairs for the ``:EVIDENCE_FOR`` edges.

    Every provenance node a fact draws on evidences it — one edge per node
    id, the same raw-material → construct anchor the statement and procedure
    tiers use.

    Args:
        facts: The atomic facts, in document order.
        source: The stable book identity.

    Returns:
        One ``{node, fact}`` per provenance node, in document order.
    """
    return [
        {
            'node': nodes.node_uuid(source, node_id),
            'fact': fact_uuid(source, fact.node_ids, index),
        }
        for index, fact in enumerate(facts)
        for node_id in fact.node_ids
    ]
