"""
Graph representation of the canonical triplet layer — ``:TripletHub``
vertices acting as assertion connectors at the hub level.

After canonicalization, every ``:Triplet`` resolves to the same three
canonical hubs (subject EntityHub, PredicateHub, object EntityHub).
Triplets that resolve to the same ``(subj_hub, pred_hub, obj_hub)``
converge on one ``:TripletHub`` — an empty connector, same hub pattern.

Three hubs converge on the TripletHub through role-specific edges:

* ``(:EntityHub)-[:CANONICAL_SUBJECT]->(:TripletHub)``
* ``(:PredicateHub)-[:CANONICAL_PREDICATE]->(:TripletHub)``
* ``(:EntityHub)-[:CANONICAL_OBJECT]->(:TripletHub)``

The canonical assertion text and its embedding live on a ``:FactHub``
reached via ``(:TripletHub)-[:HAS_FACT]->(:FactHub)``.

Every original ``:Triplet`` that backs this canonical assertion is
linked via ``(:TripletHub)-[:SUPPORTED_BY]->(:Triplet)``.

Identity: deterministic uuid5 over
``(source, subj_hub_uuid, pred_hub_uuid, obj_hub_uuid)`` —
idempotent across re-runs; new triplets add ``:SUPPORTED_BY`` edges
without changing the hub uuid.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

TRIPLET_HUB_LABEL = 'TripletHub'


def triplet_hub_uuid(
    source: str,
    subj_hub_uuid: str,
    pred_hub_uuid: str,
    obj_hub_uuid: str,
) -> str:
    """Stable, deterministic vertex key for one canonical assertion.

    Args:
        source: The stable book identity.
        subj_hub_uuid: The subject EntityHub uuid.
        pred_hub_uuid: The PredicateHub uuid.
        obj_hub_uuid: The object EntityHub uuid.

    Returns:
        The hub's hex uuid.
    """
    return uuid5(
        NAMESPACE_URL,
        f'{source}#triplet_hub#'
        f'{subj_hub_uuid}#{pred_hub_uuid}#{obj_hub_uuid}',
    ).hex


def triplet_hub_properties(
    source: str,
    subj_hub_uuid: str,
    pred_hub_uuid: str,
    obj_hub_uuid: str,
) -> dict:
    """The Neo4j property map for one TripletHub.

    The hub is empty — content lives on ``:FactHub``.

    Args:
        source: The stable book identity.
        subj_hub_uuid: Subject EntityHub uuid.
        pred_hub_uuid: PredicateHub uuid.
        obj_hub_uuid: Object EntityHub uuid.

    Returns:
        The property map.
    """
    return {
        'uuid': triplet_hub_uuid(
            source, subj_hub_uuid, pred_hub_uuid, obj_hub_uuid
        ),
        'source': nodes.source_uuid(source),
    }


def canonical_subject_pairs(
    groups: list[dict],
) -> list[dict]:
    """``{entity_hub, triplet_hub}`` pairs for ``:CANONICAL_SUBJECT``
    edges."""
    return [
        {'hub': g['subj_hub'], 'triplet_hub': g['triplet_hub_uuid']}
        for g in groups
    ]


def canonical_predicate_pairs(
    groups: list[dict],
) -> list[dict]:
    """``{predicate_hub, triplet_hub}`` pairs for
    ``:CANONICAL_PREDICATE`` edges."""
    return [
        {'hub': g['pred_hub'], 'triplet_hub': g['triplet_hub_uuid']}
        for g in groups
    ]


def canonical_object_pairs(
    groups: list[dict],
) -> list[dict]:
    """``{entity_hub, triplet_hub}`` pairs for ``:CANONICAL_OBJECT``
    edges."""
    return [
        {'hub': g['obj_hub'], 'triplet_hub': g['triplet_hub_uuid']}
        for g in groups
    ]


def supported_by_pairs(
    groups: list[dict],
) -> list[dict]:
    """``{triplet_hub, triplet}`` pairs for ``:SUPPORTED_BY``
    edges — one per original ``:Triplet`` backing the assertion."""
    pairs: list[dict] = []
    for g in groups:
        for triplet_uuid_val in g['triplet_uuids']:
            pairs.append({
                'triplet_hub': g['triplet_hub_uuid'],
                'triplet': triplet_uuid_val,
            })
    return pairs
