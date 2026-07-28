"""Structural-node graph mapping — pure, no database."""

from kms.core import models
from kms.graph import nodes


def test_node_uuid_is_deterministic():
    assert nodes.node_uuid('hefferon.pdf', 7) == nodes.node_uuid(
        'hefferon.pdf', 7
    )


def test_node_uuid_distinguishes_index_and_source():
    assert nodes.node_uuid('hefferon.pdf', 7) != nodes.node_uuid(
        'hefferon.pdf', 8
    )
    assert nodes.node_uuid('hefferon.pdf', 7) != nodes.node_uuid(
        'lebl.pdf', 7
    )


def test_node_properties_maps_kind_content_and_provenance():
    node = models.MathNode(content='$x^2$', id=3, segment_index=2)
    props = nodes.node_properties(node, 'book.pdf')
    assert props['type'] == 'math'
    assert props['content'] == '$x^2$'
    assert props['index'] == 3 and props['segment_index'] == 2


def test_node_properties_keep_index_zero():
    node = models.ParagraphNode(content='text', id=0, segment_index=0)
    props = nodes.node_properties(node, 'book.pdf')
    assert props['index'] == 0


def test_node_properties_omits_role_field():
    node = models.ListNode(content='1. do it', id=5, segment_index=1)
    assert 'role' not in nodes.node_properties(node, 'book.pdf')


def test_node_label_derives_from_class_name():
    assert nodes.node_label(models.MathNode()) == 'Math'
    assert nodes.node_label(models.ParagraphNode()) == 'Paragraph'
    assert nodes.node_label(models.InstructionNode()) == 'Instruction'


def test_node_label_for_base_astnode():
    assert nodes.node_label(models.ASTNode()) == 'AST'


def test_node_properties_link_back_to_source():
    node = models.MathNode(content='$x$', id=3, segment_index=2)
    assert (
        nodes.node_properties(node, 'book.pdf')['source']
        == nodes.source_uuid('book.pdf')
    )


def test_source_uuid_is_deterministic():
    assert nodes.source_uuid('book.pdf') == nodes.source_uuid('book.pdf')
    assert nodes.source_uuid('book.pdf') != nodes.source_uuid('other.pdf')


def test_source_properties_carry_key_and_uuid():
    props = nodes.source_properties('book.pdf', {'title': 'Linear Algebra'})
    assert props['key'] == 'book.pdf'


def test_source_metadata_cannot_clobber_key():
    props = nodes.source_properties(
        'book.pdf', {'uuid': 'hacked', 'key': 'hacked'}
    )
    assert props['uuid'] == nodes.source_uuid('book.pdf')


def test_source_properties_drop_none_metadata():
    assert 'title' not in nodes.source_properties(
        'book.pdf', {'title': None}
    )
