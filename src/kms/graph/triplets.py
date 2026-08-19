"""Row and edge builders for the raw triplet layer."""

from kms.core import identity, models
from kms.graph import nodes

TRIPLET_LABEL = 'Triplet'


def triplet_uuid(
    source: str,
    node_id: int,
    subject: str,
    predicate: str,
    object: str,
) -> str:
    """Returns the deterministic uuid for one triplet occurrence.

    Identity covers the source, the node the fact was extracted from,
    and the verbatim subject/predicate/object strings.
    """
    return identity.triplet_uuid(
        source, node_id, subject, predicate, object
    )


def triplet_properties(
    triplet: models.Triplet,
    source: str,
    node_id: int,
) -> dict:
    """Builds the empty :Triplet connector's properties.

    The verbatim subject/predicate/object strings are not persisted on
    the :Triplet; they live on its :Entity and :Predicate components.
    Only the uuid (derived from those strings so identity stays
    deterministic), source, and node_id are stored.
    """
    occurrence_id = _occurrence_id(triplet, source, node_id)
    return {
        'uuid': occurrence_id,
        'source': nodes.source_uuid(source),
        'node_id': node_id,
    }


def _occurrence_id(
    triplet: models.Triplet, source: str, node_id: int
) -> str:
    """Returns and verifies one assigned triplet occurrence UUID."""
    expected = triplet_uuid(
        source,
        node_id,
        triplet.subject,
        triplet.predicate,
        triplet.object,
    )
    assigned = triplet.occurrence_uuids.get(node_id)
    if assigned is None:
        raise ValueError(
            f'triplet occurrence for node {node_id} is missing its uuid'
        )
    if assigned != expected:
        raise ValueError(
            f'triplet occurrence uuid {assigned!r} does not match expected '
            f'{expected!r}'
        )
    return assigned


def triplet_rows(
    triplets: list[models.Triplet],
    source: str,
) -> list[dict]:
    """Builds one row per triplet occurrence (per evidence node)."""
    rows: list[dict] = []
    for triplet in triplets:
        for node_id in triplet.node_ids:
            rows.append(triplet_properties(triplet, source, node_id))
    return rows


def evidence_pairs(
    triplets: list[models.Triplet],
    source: str,
) -> list[dict]:
    """Builds evidence node→triplet edge pairs."""
    pairs: list[dict] = []
    for triplet in triplets:
        for node_id in triplet.node_ids:
            pairs.append(
                {
                    'node': nodes.node_uuid(source, node_id),
                    'triplet': _occurrence_id(triplet, source, node_id),
                }
            )
    return pairs
