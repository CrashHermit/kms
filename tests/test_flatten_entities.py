"""Flattening the finder overlays into one document-ordered, globally-id'd entity list — pure
logic, no database. This is what the collector runs and every stage after it reads."""

from kms.core import models


def _nodes():
    return [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH,
            content=str(i),
            id=i,
            segment_index=0,
        )
        for i in range(6)
    ]


def test_flatten_concatenates_orders_by_document_position_and_assigns_ids():
    nodes = _nodes()
    flat = models.flatten_entities(
        [
            [models.Entity(type='problem', members=[3, 4])],
            [models.Entity(type='definition', members=[0])],
            [models.Entity(type='theorem', members=[1, 2])],
        ],
        nodes,
    )
    # Ordered by first member's document position (def@0, thm@1, prob@3), ids 0..2.
    assert [(e.id, e.type, e.members) for e in flat] == [
        (0, 'definition', [0]),
        (1, 'theorem', [1, 2]),
        (2, 'problem', [3, 4]),
    ]


def test_flatten_sorts_memberless_entities_to_the_end():
    nodes = _nodes()
    flat = models.flatten_entities(
        [
            [models.Entity(type='problem', members=[])],
            [models.Entity(type='definition', members=[2])],
            [],
        ],
        nodes,
    )
    assert [(e.id, e.type) for e in flat] == [
        (0, 'definition'),
        (1, 'problem'),
    ]


def test_flatten_takes_any_number_of_overlays_including_one():
    # the block-finder layer contributes a single overlay; the per-type layer three. Same call.
    nodes = _nodes()
    flat = models.flatten_entities(
        [[models.Entity(type='law', members=[1]), models.Entity(members=[0])]],
        nodes,
    )
    assert [(e.id, e.type) for e in flat] == [(0, None), (1, 'law')]
