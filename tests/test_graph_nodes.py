import asyncio

from kms.core import models
from kms.graph import nodes, queries, writer


def test_node_uuid_is_deterministic():
    node = models.SourceNode(uuid='node-7', document_index=0)
    assert nodes.node_uuid('hefferon.pdf', node) == nodes.node_uuid(
        'hefferon.pdf', node
    )


def test_node_uuid_distinguishes_index_and_source():
    # Nodes without explicit UUIDs use their source and position.
    node_a = models.SourceNode(document_index=0, index=7)
    node_b = models.SourceNode(document_index=0, index=8)
    assert nodes.node_uuid('hefferon.pdf', node_a) != nodes.node_uuid(
        'hefferon.pdf', node_b
    )
    node_c = models.SourceNode(document_index=0, index=7)
    assert nodes.node_uuid('hefferon.pdf', node_a) != nodes.node_uuid(
        'lebl.pdf', node_c
    )


def test_node_properties_maps_kind_content_and_provenance():
    node = models.SourceNode(
        uuid='node-3', type='math', content='$x^2$', document_index=2
    )
    props = nodes.node_properties(node, 'book.pdf')
    assert props['type'] == 'math'
    assert props['content'] == '$x^2$'
    assert props['uuid'] == 'node-3' and props['document_index'] == 2


def test_node_properties_keep_index_zero():
    node = models.SourceNode(
        uuid='node-0', type='paragraph', content='text', document_index=0
    )
    props = nodes.node_properties(node, 'book.pdf')
    assert props['uuid'] == 'node-0'


def test_node_properties_omits_role_field():
    node = models.SourceNode(
        uuid='node-5', type='list', content='1. do it', document_index=1
    )
    assert 'role' not in nodes.node_properties(node, 'book.pdf')


def test_visual_assets_are_graph_rows_and_node_edges():
    node = models.SourceNode(
        uuid='node-asset',
        assets=[
            models.VisualAsset(path='page-1.png'),
            models.VisualAsset(path='page-2.png'),
        ],
    )
    rows = nodes.visual_asset_rows([node], 'book.pdf')
    pairs = nodes.visual_asset_pairs([node], 'book.pdf')
    assert [row['path'] for row in rows] == ['page-1.png', 'page-2.png']
    assert [row['index'] for row in rows] == [0, 1]
    assert [pair['node'] for pair in pairs] == ['node-asset', 'node-asset']
    assert [pair['asset'] for pair in pairs] == [row['uuid'] for row in rows]


def test_persist_nodes_merges_visual_assets_and_attachment_edges():
    calls = []

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def run(self, query, **kwargs):
            calls.append((query, kwargs))

    node = models.SourceNode(
        uuid='node-asset',
        assets=[models.VisualAsset(path='page-1.png')],
    )

    asyncio.run(
        writer.persist_nodes(
            [node],
            'book.pdf',
            session_factory=Session,
        )
    )

    asset_rows = next(
        kwargs['rows']
        for query, kwargs in calls
        if query == queries.MERGE_VISUAL_ASSETS
    )
    attachment_pairs = next(
        kwargs['pairs']
        for query, kwargs in calls
        if query == queries.MERGE_NODE_ASSETS
    )
    assert asset_rows[0]['path'] == 'page-1.png'
    assert attachment_pairs == [
        {'node': 'node-asset', 'asset': asset_rows[0]['uuid']}
    ]


def test_node_properties_do_not_duplicate_visual_asset_paths():
    node = models.SourceNode(
        uuid='node-asset',
        assets=[models.VisualAsset(path='page-1.png')],
    )
    assert 'image_paths' not in nodes.node_properties(node, 'book.pdf')


def test_node_label_derives_from_class_name():
    assert (
        nodes.node_label(
            models.SourceNode(
                type='math',
            )
        )
        == 'Math'
    )
    assert (
        nodes.node_label(
            models.SourceNode(
                type='paragraph',
            )
        )
        == 'Paragraph'
    )
    assert (
        nodes.node_label(
            models.SourceNode(
                type='instruction',
            )
        )
        == 'Instruction'
    )


def test_node_label_for_typeless_node():
    assert nodes.node_label(models.SourceNode()) is None


def test_node_properties_link_back_to_source():
    node = models.SourceNode(
        uuid='node-3', type='math', content='$x$', document_index=2
    )
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
