"""Writer planning — the pure grouping/edge logic, no database. The two session.run calls in
persist_nodes are covered by the opt-in integration test."""

from kms.core.models import ASTNode, NodeType
from kms.graph.nodes import node_uuid
from kms.graph.writer import next_pairs, node_batches

_STREAM = [
    ASTNode(type=NodeType.HEADER, content="§1", id=0, seg_index=0),
    ASTNode(type=NodeType.PARAGRAPH, content="a", id=1, seg_index=0),
    ASTNode(type=NodeType.MATH, content="$x$", id=2, seg_index=0),
    ASTNode(type=NodeType.PARAGRAPH, content="b", id=3, seg_index=0),
]


def test_batches_group_by_per_type_label():
    batches = node_batches(_STREAM, "book.pdf")
    assert set(batches) == {"Header", "Paragraph", "Math"}
    assert len(batches["Paragraph"]) == 2  # two paragraphs share the label bucket
    assert len(batches["Math"]) == 1


def test_batch_rows_carry_the_uuid_for_the_merge_key():
    rows = node_batches(_STREAM, "book.pdf")["Math"]
    assert rows[0]["uuid"] == node_uuid("book.pdf", 2)


def test_typeless_node_falls_into_the_none_bucket():
    batches = node_batches([ASTNode(content="?", id=0, seg_index=0)], "book.pdf")
    assert set(batches) == {None}


def test_next_pairs_thread_consecutive_nodes_in_order():
    pairs = next_pairs(_STREAM, "book.pdf")
    assert len(pairs) == 3  # n-1 edges for n nodes
    assert pairs[0] == {"from": node_uuid("book.pdf", 0), "to": node_uuid("book.pdf", 1)}
    assert pairs[-1] == {"from": node_uuid("book.pdf", 2), "to": node_uuid("book.pdf", 3)}


def test_next_pairs_empty_for_single_or_no_nodes():
    assert next_pairs(_STREAM[:1], "book.pdf") == []
    assert next_pairs([], "book.pdf") == []
