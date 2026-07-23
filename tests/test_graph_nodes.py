"""Structural-node graph mapping — pure, no database. Verifies identity is stable/deterministic
and that a core.ASTNode maps onto the expected Neo4j property shape."""

from kms.core.models import ASTNode, NodeType
from kms.graph.nodes import node_label, node_properties, node_uuid


def test_node_uuid_is_deterministic_for_same_source_and_index():
    assert node_uuid("hefferon.pdf", 7) == node_uuid("hefferon.pdf", 7)


def test_node_uuid_distinguishes_index_and_source():
    assert node_uuid("hefferon.pdf", 7) != node_uuid("hefferon.pdf", 8)  # different position
    assert node_uuid("hefferon.pdf", 7) != node_uuid("lebl.pdf", 7)  # different book


def test_node_properties_maps_type_content_and_provenance():
    node = ASTNode(type=NodeType.MATH, content="$x^2$", id=3, seg_index=2)
    props = node_properties(node, "book.pdf")
    assert props["type"] == "math"
    assert props["content"] == "$x^2$"
    assert props["index"] == 3 and props["seg_index"] == 2
    assert props["uuid"] == node_uuid("book.pdf", 3)


def test_node_properties_omits_unset_role_but_keeps_index_zero():
    node = ASTNode(type=NodeType.PARAGRAPH, content="text", id=0, seg_index=0)
    props = node_properties(node, "book.pdf")
    assert "role" not in props  # None is dropped, like in nodes.json
    assert props["index"] == 0  # a falsy-but-valid value is kept


def test_node_properties_keeps_role_when_set():
    node = ASTNode(type=NodeType.LIST, content="1. do it", id=5, seg_index=1, role="instruction")
    assert node_properties(node, "book.pdf")["role"] == "instruction"


def test_node_label_is_the_capitalized_type():
    assert node_label(ASTNode(type=NodeType.MATH)) == "Math"
    assert node_label(ASTNode(type=NodeType.PARAGRAPH)) == "Paragraph"


def test_node_label_is_none_without_a_type():
    assert node_label(ASTNode()) is None
