"""
Graph representation of the canonical definition layer — ``:Definition``
vertices carrying the synthesised canonical text for a hub.

The canonicalization pass clusters entity or predicate spokes and writes
one ``:Definition`` per hub. This module maps one definition onto its
Neo4j form: pure mapping, free of the neo4j driver (the driver lives in
``graph.db``, the writes in ``graph.writer``).

Representation: ``:Definition`` carries ``text`` (the LLM-synthesised
canonical description) and ``embedding`` (the vector of that text, for
semantic search over canonical concepts). It is reached from its hub via
``(:EntityHub)-[:HAS_DEFINITION]->(:Definition)`` or the predicate
equivalent. The hub itself is empty — the content lives here.

Identity: deterministic uuid5 over the hub's uuid — one definition per
hub, idempotent across re-runs.
"""

from uuid import NAMESPACE_URL, uuid5


DEFINITION_LABEL = 'Definition'


def definition_uuid(hub_uuid: str) -> str:
    """Stable, deterministic vertex key for one hub's definition.

    Args:
        hub_uuid: The owning hub's uuid (entity or predicate).

    Returns:
        The definition's hex uuid, disjoint from every other uuid.
    """
    return uuid5(NAMESPACE_URL, f'{hub_uuid}#definition').hex


def definition_properties(
    hub_uuid: str,
    text: str,
    embedding: list[float] | None = None,
) -> dict:
    """The Neo4j property map for one definition.

    Args:
        hub_uuid: The owning hub's uuid.
        text: The canonical definition text.
        embedding: The embedding vector of ``text``, if computed.

    Returns:
        The property map, with None values omitted.
    """
    properties = {
        'uuid': definition_uuid(hub_uuid),
        'text': text,
        'embedding': embedding,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def definition_rows(definitions: list[dict]) -> list[dict]:
    """Every definition's property map, one flat list.

    Args:
        definitions: One dict per hub: ``{hub_uuid, definition_text,
            definition_embedding}``.

    Returns:
        One property map per definition.
    """
    return [
        definition_properties(
            hub_uuid=entry['hub_uuid'],
            text=entry['definition_text'],
            embedding=entry.get('definition_embedding'),
        )
        for entry in definitions
    ]


def has_definition_pairs(definitions: list[dict]) -> list[dict]:
    """The ``{hub, definition}`` uuid pairs for ``:HAS_DEFINITION`` edges.

    Works for both ``:EntityHub`` and ``:PredicateHub`` — the edge is
    label-agnostic.

    Args:
        definitions: One dict per hub (same shape as ``definition_rows``).

    Returns:
        One ``{hub, definition}`` per hub.
    """
    return [
        {
            'hub': entry['hub_uuid'],
            'definition': definition_uuid(entry['hub_uuid']),
        }
        for entry in definitions
    ]
