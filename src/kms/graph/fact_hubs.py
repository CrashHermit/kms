"""
Graph representation of the canonical fact layer — ``:FactHub`` vertices
carrying the assembled canonical assertion text.

One ``:FactHub`` per ``:TripletHub``, reached via ``:HAS_FACT``.  Carries
the canonical assertion as a single sentence (assembled from the three
hub display names) and its embedding for vector search.

Distinct from ``:Fact`` (atomic fact intermediates) — ``:FactHub`` is the
canonical-tier assertion that a ``:TripletHub`` represents.

Identity: deterministic uuid5 over the owning ``:TripletHub``'s uuid —
one fact per hub, idempotent across re-runs.
"""

from uuid import NAMESPACE_URL, uuid5

FACT_HUB_LABEL = 'FactHub'


def fact_hub_uuid(triplet_hub_uuid: str) -> str:
    """Stable vertex key for one TripletHub's canonical assertion."""
    return uuid5(NAMESPACE_URL, f'{triplet_hub_uuid}#fact_hub').hex


def fact_hub_properties(
    triplet_hub_uuid: str,
    text: str,
    embedding: list[float] | None = None,
) -> dict:
    """The Neo4j property map for one FactHub.

    Args:
        triplet_hub_uuid: The owning TripletHub's uuid.
        text: The assembled canonical assertion sentence.
        embedding: The embedding vector of *text*, if computed.

    Returns:
        The property map, with None values omitted.
    """
    properties = {
        'uuid': fact_hub_uuid(triplet_hub_uuid),
        'text': text,
        'embedding': embedding,
    }
    return {
        key: value
        for key, value in properties.items()
        if value is not None
    }


def fact_hub_rows(groups: list[dict]) -> list[dict]:
    """Every FactHub's property map, one flat list.

    Args:
        groups: One dict per canonical assertion group:
            ``{triplet_hub_uuid, fact_text, fact_embedding}``.

    Returns:
        One property map per group.
    """
    return [
        fact_hub_properties(
            triplet_hub_uuid=g['triplet_hub_uuid'],
            text=g['fact_text'],
            embedding=g.get('fact_embedding'),
        )
        for g in groups
    ]


def has_fact_pairs(groups: list[dict]) -> list[dict]:
    """``{triplet_hub, fact_hub}`` pairs for ``:HAS_FACT`` edges."""
    return [
        {
            'triplet_hub': g['triplet_hub_uuid'],
            'fact_hub': fact_hub_uuid(g['triplet_hub_uuid']),
        }
        for g in groups
    ]
