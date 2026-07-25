"""Ordering the block overlay into one document-ordered, globally-id'd entity list — pure logic,
no database. This is what the entity persister feeds to the graph writer."""

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


def test_flatten_orders_by_document_position_and_assigns_ids():
    nodes = _nodes()
    flat = models.flatten_entities(
        [
            models.Entity(type='exercise', members=[3, 4]),
            models.Entity(type='definition', members=[0]),
            models.Entity(type='theorem', members=[1, 2]),
        ],
        nodes,
    )
    # Ordered by first member's document position (def@0, thm@1, exercise@3), ids 0..2.
    assert [(e.id, e.type, e.members) for e in flat] == [
        (0, 'definition', [0]),
        (1, 'theorem', [1, 2]),
        (2, 'exercise', [3, 4]),
    ]


def test_flatten_sorts_memberless_entities_to_the_end():
    nodes = _nodes()
    flat = models.flatten_entities(
        [
            models.Entity(type='exercise', members=[]),
            models.Entity(type='definition', members=[2]),
        ],
        nodes,
    )
    assert [(e.id, e.type) for e in flat] == [
        (0, 'definition'),
        (1, 'exercise'),
    ]


def test_flatten_leaves_procedures_attached_to_their_block():
    nodes = _nodes()
    theorem = models.Entity(
        type='theorem',
        members=[1],
        procedures=[models.Procedure(members=[2], steps=['Clear.'])],
    )
    flat = models.flatten_entities([theorem], nodes)
    assert flat[0].procedures[0].steps == ['Clear.']
