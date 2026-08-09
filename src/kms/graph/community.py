from uuid import NAMESPACE_URL, uuid5

from kms.graph import nodes

COMMUNITY_LABEL = 'Community'


def community_uuid(source: str, member_hub_uuids: list[str]) -> str:
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
    pairs: list[dict] = []
    for c in communities:
        comm_uuid = c['community_uuid']
        for hub_uuid in c['member_hub_uuids']:
            pairs.append({'community': comm_uuid, 'hub': hub_uuid})
    return pairs


def evidence_pairs(communities: list[dict]) -> list[dict]:
    pairs: list[dict] = []
    for c in communities:
        comm_uuid = c['community_uuid']
        for triplet_uuid in c.get('triplet_uuids', []):
            pairs.append(
                {'community': comm_uuid, 'triplet': triplet_uuid}
            )
    return pairs

