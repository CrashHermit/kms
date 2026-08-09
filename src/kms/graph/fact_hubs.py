from uuid import NAMESPACE_URL, uuid5

FACT_HUB_LABEL = 'FactHub'


def fact_hub_uuid(triplet_hub_uuid: str) -> str:
    return uuid5(NAMESPACE_URL, f'{triplet_hub_uuid}#fact_hub').hex


def fact_hub_properties(
    triplet_hub_uuid: str,
    text: str,
    embedding: list[float] | None = None,
) -> dict:
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
    return [
        fact_hub_properties(
            triplet_hub_uuid=g['triplet_hub_uuid'],
            text=g['fact_text'],
            embedding=g.get('fact_embedding'),
        )
        for g in groups
    ]


def has_fact_pairs(groups: list[dict]) -> list[dict]:
    return [
        {
            'triplet_hub': g['triplet_hub_uuid'],
            'fact_hub': fact_hub_uuid(g['triplet_hub_uuid']),
        }
        for g in groups
    ]

