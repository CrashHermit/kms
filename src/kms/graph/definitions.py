from uuid import NAMESPACE_URL, uuid5

DEFINITION_LABEL = 'Definition'


def definition_uuid(hub_uuid: str) -> str:
    return uuid5(NAMESPACE_URL, f'{hub_uuid}#definition').hex


def definition_properties(
    hub_uuid: str,
    text: str,
    embedding: list[float] | None = None,
) -> dict:
    properties = {
        'uuid': definition_uuid(hub_uuid),
        'text': text,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def definition_rows(definitions: list[dict]) -> list[dict]:
    return [
        definition_properties(
            hub_uuid=entry['hub_uuid'],
            text=entry['definition_text'],
            embedding=entry.get('definition_embedding'),
        )
        for entry in definitions
    ]


def has_definition_pairs(definitions: list[dict]) -> list[dict]:
    return [
        {
            'hub': entry['hub_uuid'],
            'definition': definition_uuid(entry['hub_uuid']),
        }
        for entry in definitions
    ]

