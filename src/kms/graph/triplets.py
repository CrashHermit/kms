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
    return identity.triplet_uuid(source, node_id, subject, predicate, object)


def triplet_properties(
    triplet: models.Triplet,
    source: str,
    node_position: int,
) -> dict:
    """Builds the empty :Triplet connector's properties.

    The verbatim subject/predicate/object strings are not persisted on
    the :Triplet; they live on its :Entity and :Predicate components.
    Only the uuid (derived from those strings so identity stays
    deterministic), source, and node_position are stored.
    """
    occurrence_id = _occurrence_id(triplet, source, node_position)
    return {
        'uuid': occurrence_id,
        'source': nodes.source_uuid(source),
        'node_position': node_position,
    }


def _occurrence_id(
    triplet: models.Triplet, source: str, node_position: int
) -> str:
    """Returns and verifies one assigned triplet occurrence UUID."""
    expected = triplet_uuid(
        source,
        node_position,
        triplet.subject,
        triplet.predicate,
        triplet.object,
    )
    assigned = triplet.occurrence_uuids.get(node_position)
    if assigned is None:
        raise ValueError(
            f'triplet occurrence for position {node_position} is missing its uuid'
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
        for node_position in triplet.evidence_positions:
            rows.append(triplet_properties(triplet, source, node_position))
    return rows


def evidence_pairs(
    triplets: list[models.Triplet],
    source: str,
    doc_nodes: list[models.Node],
) -> list[dict]:
    """Builds evidence node→triplet edge pairs."""
    pairs: list[dict] = []
    for triplet in triplets:
        for node_position in triplet.evidence_positions:
            if not 0 <= node_position < len(doc_nodes):
                raise ValueError(
                    f'triplet evidence position {node_position} is outside '
                    f'the node stream'
                )
            node = doc_nodes[node_position]
            pairs.append(
                {
                    'node': nodes.node_uuid(source, node),
                    'triplet': _occurrence_id(triplet, source, node_position),
                }
            )
    return pairs
