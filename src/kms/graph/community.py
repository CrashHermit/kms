"""
Graph representation of the community layer — ``:Community`` nodes that
summarise a cluster of related canonical hubs.

Community detection runs on the canonical hub graph (``:EntityHub`` and
``:PredicateHub`` vertices connected through ``:Triplet`` hubs).  Densely
connected groups of hubs form a community; an LLM synthesises a summary
paragraph from the member hub definitions and the canonical triplets
within the community.

Communities are derived structure, not durable identity — they are
regenerated when the graph has changed enough to warrant it.  Each
community carries a deterministic uuid so re-building on the same
graph is idempotent.

Representation: ``:Community`` carries ``summary_text`` and
``summary_embedding``.  Member hubs are reached via
``(:Community)-[:HAS_MEMBER]->(:EntityHub|:PredicateHub)``.  Evidence
triplets are reached via
``(:Community)-[:COMMUNITY_EVIDENCE]->(:Triplet)``.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

COMMUNITY_LABEL = 'Community'


def community_uuid(source: str, member_hub_uuids: list[str]) -> str:
    """Stable, deterministic vertex key for one community.

    Args:
        source: The stable book identity.
        member_hub_uuids: The uuids of every EntityHub and PredicateHub
            in the community, sorted for determinism.

    Returns:
        The community's hex uuid.
    """
    return uuid5(
        NAMESPACE_URL,
        f'{source}#community#{nodes.block_key(sorted(member_hub_uuids))}',
    ).hex


def community_properties(
    source: str,
    member_hub_uuids: list[str],
    summary_text: str,
    summary_embedding: list[float] | None = None,
) -> dict:
    """The Neo4j property map for one community.

    Args:
        source: The stable book identity.
        member_hub_uuids: The member hub uuids (sorted), used for
            identity.
        summary_text: The LLM-generated summary paragraph.
        summary_embedding: The embedding of *summary_text*.

    Returns:
        The property map, with None values omitted.
    """
    properties = {
        'uuid': community_uuid(source, member_hub_uuids),
        'source': nodes.source_uuid(source),
        'summary_text': summary_text,
        'summary_embedding': summary_embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def community_rows(
    communities: list[dict], source: str
) -> list[dict]:
    """Every community's property map, one flat list.

    Args:
        communities: One dict per community:
            ``{member_hub_uuids, summary_text, summary_embedding}``.
        source: The stable book identity.

    Returns:
        One property map per community.
    """
    return [
        community_properties(
            source,
            c['member_hub_uuids'],
            c['summary_text'],
            c.get('summary_embedding'),
        )
        for c in communities
    ]


def member_pairs(communities: list[dict]) -> list[dict]:
    """The ``{community, hub}`` uuid pairs for ``:HAS_MEMBER`` edges.

    Args:
        communities: One dict per community with ``community_uuid`` and
            ``member_hub_uuids``.

    Returns:
        One ``{community, hub}`` per member hub.
    """
    pairs: list[dict] = []
    for c in communities:
        comm_uuid = c['community_uuid']
        for hub_uuid in c['member_hub_uuids']:
            pairs.append({'community': comm_uuid, 'hub': hub_uuid})
    return pairs


def evidence_pairs(communities: list[dict]) -> list[dict]:
    """The ``{community, triplet}`` uuid pairs for
    ``:COMMUNITY_EVIDENCE`` edges.

    Args:
        communities: One dict per community with ``community_uuid`` and
            ``triplet_uuids``.

    Returns:
        One ``{community, triplet}`` per evidence triplet.
    """
    pairs: list[dict] = []
    for c in communities:
        comm_uuid = c['community_uuid']
        for triplet_uuid in c.get('triplet_uuids', []):
            pairs.append(
                {'community': comm_uuid, 'triplet': triplet_uuid}
            )
    return pairs
