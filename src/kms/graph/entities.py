"""
Graph representation of the entity layer — ``:Entity`` vertices carrying the
descriptions of a triplet's subject/object components.

The entity enricher walks the provenance node stream node-by-node and writes
one description per distinct subject/object surface form it finds at that node
(``node_entity_descriptions``). The graph tier reifies each triplet's subject
and object as their OWN ``:Entity`` vertices: pure mapping, free of the neo4j
driver (the driver lives in ``graph.db``, the writes in ``graph.writer``).

Representation: every entity carries the ``:Entity`` label with its ``name``
(verbatim from the source), its ``description`` (when the enricher provided
one), and the anchor ``node_id``. The owning ``:Triplet`` hub points at it
via ``(:Triplet)-[:HAS_SUBJECT]->(:Entity)`` or
``(:Triplet)-[:HAS_OBJECT]->(:Entity)``. Every triplet's subject and object
always get their own vertices — the entity exists regardless of whether the
enricher described it. There is deliberately NO direct
``(:Node)-[:HAS_ENTITY]->(:Entity)`` edge — the node anchors the entity only
through the ``node_id`` property, and the triplet is what uses it.

Identity is PER-TRIPLET: ``uuid5`` over ``(source, node_id, triplet_uuid,
role)`` where ``role`` is ``'subject'`` or ``'object'``. Every triplet gets its
own subject vertex and its own object vertex — zero sharing, even when the
same surface form appears in two triplets at the same node. This mirrors the
per-triplet ``:Predicate`` decision and means no entity is ever shared between
two assertions.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes
from kms.graph.facts import fact_uuid

ENTITY_LABEL = 'Entity'


def entity_uuid(
    source: str, node_id: int, triplet_uuid_val: str, role: str
) -> str:
    """Stable, deterministic vertex key for one per-triplet entity.

    Args:
        source: The stable book identity.
        node_id: The anchor node's id in the flattened stream.
        triplet_uuid_val: The owning triplet hub's uuid (see
            ``graph.triplets.triplet_uuid``).
        role: ``'subject'`` or ``'object'`` — which component this entity is.

    Returns:
        The entity's hex uuid, disjoint from every other uuid.
    """
    return uuid5(
        NAMESPACE_URL,
        f'{source}#entity#{node_id}#{triplet_uuid_val}#{role}',
    ).hex


def entity_properties(
    source: str,
    node_id: int,
    triplet_uuid_val: str,
    role: str,
    name: str,
    description: str | None = None,
    embedding: list[float] | None = None,
) -> dict:
    """The Neo4j property map for one entity occurrence.

    Args:
        source: The stable book identity.
        node_id: The anchor node's id.
        triplet_uuid_val: The owning triplet hub's uuid.
        role: ``'subject'`` or ``'object'``.
        name: The entity's surface form, verbatim from the source.
        description: The enricher's description of the entity at this node.
        embedding: The embedding vector for ``name + description``, if
            computed.

    Returns:
        The property map, with None values omitted.
    """
    properties = {
        'uuid': entity_uuid(source, node_id, triplet_uuid_val, role),
        'source': nodes.source_uuid(source),
        'node_id': node_id,
        'role': role,
        'name': name,
        'description': description,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def _description_at(
    node_id: int,
    name: str,
    node_entity_descriptions: dict[int, list[dict]],
) -> str | None:
    """The description the enricher wrote for *name* at *node_id*.

    Args:
        node_id: The anchor node.
        name: The entity's surface form.
        node_entity_descriptions: The enricher's per-node mapping: node id to
            a list of ``{name, description}`` dicts.

    Returns:
        The matching description, or None.
    """
    for entry in node_entity_descriptions.get(node_id, []):
        if entry['name'] == name:
            return entry.get('description')
    return None


def _embedding_at(
    node_id: int,
    name: str,
    node_entity_descriptions: dict[int, list[dict]],
) -> list[float] | None:
    """The embedding computed for *name* at *node_id*.

    Args:
        node_id: The anchor node.
        name: The entity's surface form.
        node_entity_descriptions: The per-node mapping, where each entry
            may carry an ``embedding`` field.

    Returns:
        The embedding vector, or None.
    """
    for entry in node_entity_descriptions.get(node_id, []):
        if entry['name'] == name:
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


def entity_rows(
    triplets: list[models.Triplet],
    facts: list[models.AtomicFact],
    source: str,
    node_entity_descriptions: dict[int, list[dict]],
) -> list[dict]:
    """Every entity's property map, one flat list.

    One row per (node, triplet, role): each triplet's subject and object are
    always reified as their own vertices, written separately per node
    occurrence. The enricher's description is attached when available,
    omitted otherwise.

    Args:
        triplets: The triplets, in document order (grouped by source fact).
        facts: The atomic facts, in document order, aligned with
            ``triplets`` by ``fact_index``.
        source: The stable book identity.
        node_entity_descriptions: The enricher's per-node entity descriptions.

    Returns:
        One property map per (node, triplet, role).
    """
    from kms.graph.triplets import triplet_uuid

    rows: list[dict] = []
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
            for role, name in (
                ('subject', triplet.subject),
                ('object', triplet.object),
            ):
                description = _description_at(
                    node_id, name, node_entity_descriptions
                )
                embedding = _embedding_at(
                    node_id, name, node_entity_descriptions
                )
                rows.append(
                    entity_properties(
                        source,
                        node_id,
                        triplet_uuid_val,
                        role,
                        name,
                        description,
                        embedding,
                    )
                )
    return rows
