"""Structural-node graph mapping — pure, no database. Verifies identity is stable/deterministic
and that a core.ASTNode maps onto the expected Neo4j property shape."""

from kms.core.models import ASTNode, NodeType
from kms.graph.nodes import (
    node_label,
    node_properties,
    node_uuid,
    source_properties,
    source_uuid,
)


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
    assert "role" not in props  # None-valued fields are dropped from the property map
    assert props["index"] == 0  # a falsy-but-valid value is kept


def test_node_properties_keeps_role_when_set():
    node = ASTNode(type=NodeType.LIST, content="1. do it", id=5, seg_index=1, role="instruction")
    assert node_properties(node, "book.pdf")["role"] == "instruction"


def test_node_label_is_the_capitalized_type():
    assert node_label(ASTNode(type=NodeType.MATH)) == "Math"
    assert node_label(ASTNode(type=NodeType.PARAGRAPH)) == "Paragraph"


def test_node_label_is_none_without_a_type():
    assert node_label(ASTNode()) is None


def test_node_properties_link_back_to_the_source_node():
    node = ASTNode(type=NodeType.MATH, content="$x$", id=3, seg_index=2)
    assert node_properties(node, "book.pdf")["source"] == source_uuid("book.pdf")


def test_source_uuid_is_deterministic_and_distinct_per_book():
    assert source_uuid("book.pdf") == source_uuid("book.pdf")
    assert source_uuid("book.pdf") != source_uuid("other.pdf")


def test_source_properties_carry_key_and_uuid_and_merge_metadata():
    props = source_properties("book.pdf", {"title": "Linear Algebra", "author": "Hefferon"})
    assert props["key"] == "book.pdf"
    assert props["uuid"] == source_uuid("book.pdf")
    assert props["title"] == "Linear Algebra" and props["author"] == "Hefferon"


def test_source_metadata_cannot_clobber_key_or_uuid():
    props = source_properties("book.pdf", {"uuid": "hacked", "key": "hacked"})
    assert props["uuid"] == source_uuid("book.pdf") and props["key"] == "book.pdf"


def test_source_properties_drop_none_metadata():
    assert "title" not in source_properties("book.pdf", {"title": None})
