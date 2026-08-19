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
    assert nodes.node_uuid('hefferon.pdf', 7) != nodes.node_uuid('lebl.pdf', 7)


def test_node_properties_maps_kind_content_and_provenance():
    node = models.Node(type='math', content='$x^2$', id=3, document_index=2)
    props = nodes.node_properties(node, 'book.pdf')
    assert props['type'] == 'math'
    assert props['content'] == '$x^2$'
    assert props['index'] == 3 and props['document_index'] == 2


def test_node_properties_keep_index_zero():
    node = models.Node(
        type='paragraph', content='text', id=0, document_index=0
    )
    props = nodes.node_properties(node, 'book.pdf')
    assert props['index'] == 0


def test_node_properties_omits_role_field():
    node = models.Node(
        type='list', content='1. do it', id=5, document_index=1
    )
    assert 'role' not in nodes.node_properties(node, 'book.pdf')


def test_node_label_derives_from_class_name():
    assert (
        nodes.node_label(
            models.Node(
                type='math',
            )
        )
        == 'Math'
    )
    assert (
        nodes.node_label(
            models.Node(
                type='paragraph',
            )
        )
        == 'Paragraph'
    )
    assert (
        nodes.node_label(
            models.Node(
                type='instruction',
            )
        )
        == 'Instruction'
    )


def test_node_label_for_typeless_node():
    assert nodes.node_label(models.Node()) is None


def test_node_properties_link_back_to_source():
    node = models.Node(type='math', content='$x$', id=3, document_index=2)
    assert nodes.node_properties(node, 'book.pdf')[
        'source'
    ] == nodes.source_uuid('book.pdf')


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
    assert 'title' not in nodes.source_properties('book.pdf', {'title': None})
