"""Flattening the three per-type overlays into one document-ordered, globally-id'd entity list —
pure logic, no database. This is what the entity persister feeds to the graph writer."""

from kms.core.models import ASTNode, Entity, EntityType, NodeType, flatten_entities


def _nodes():
    return [ASTNode(type=NodeType.PARAGRAPH, content=str(i), id=i, seg_index=0) for i in range(6)]


def test_flatten_concatenates_orders_by_document_position_and_assigns_ids():
    nodes = _nodes()
    flat = flatten_entities(
        problem=[Entity(type=EntityType.PROBLEM, members=[3, 4])],
        definition=[Entity(type=EntityType.DEFINITION, members=[0])],
        theorem=[Entity(type=EntityType.THEOREM, members=[1, 2])],
        nodes=nodes,
    )
    # Ordered by first member's document position (def@0, thm@1, prob@3), ids 0..2.
    assert [(e.id, e.type.value, e.members) for e in flat] == [
        (0, "definition", [0]),
        (1, "theorem", [1, 2]),
        (2, "problem", [3, 4]),
    ]


def test_flatten_sorts_memberless_entities_to_the_end():
    nodes = _nodes()
    flat = flatten_entities(
        problem=[Entity(type=EntityType.PROBLEM, members=[])],
        definition=[Entity(type=EntityType.DEFINITION, members=[2])],
        theorem=[],
        nodes=nodes,
    )
    assert [(e.id, e.type.value) for e in flat] == [(0, "definition"), (1, "problem")]
