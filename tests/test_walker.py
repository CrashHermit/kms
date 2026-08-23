import pytest

from kms.core import models, walker


def _nodes():
    return [
        models.Node(uuid=f'node-{index}', content=f'node {index}') for index in range(5)
    ]


def test_marked_window_keeps_targets_and_adds_static_context_per_side():
    window = walker.marked_window(
        _nodes(),
        [2, 3],
        backward_budget=100,
        forward_budget=100,
        marker='statement',
    )

    assert [node.position for node in window] == [0, 1, 2, 3, 4]
    assert [node.marker for node in window] == [
        None,
        None,
        'statement',
        'statement',
        None,
    ]


def test_marked_window_zero_budget_is_directional():
    nodes = _nodes()
    backward_only = walker.marked_window(
        nodes, [2], backward_budget=100, forward_budget=0
    )
    forward_only = walker.marked_window(
        nodes, [2], backward_budget=0, forward_budget=100
    )

    # WindowNode.position is local position within the window
    assert [node.position for node in backward_only] == [0, 1, 2]
    assert [node.position for node in forward_only] == [0, 1, 2]


def test_marked_window_rejects_empty_targets():
    with pytest.raises(ValueError, match='at least one target'):
        walker.marked_window(_nodes(), [])


def test_nodes_before_and_after_retain_image_nodes(tmp_path):
    image_path = tmp_path / 'figure.png'
    image_path.write_bytes(b'image')
    nodes = [
        models.Node(content='before'),
        models.Node(type='image', content='', image_path=str(image_path)),
        models.Node(content='target'),
        models.Node(type='image', content='', image_path=str(image_path)),
        models.Node(content='after'),
    ]

    before = walker.nodes_before(nodes, cursor=2, budget=100)
    after = walker.nodes_after(nodes, cursor=2, budget=100)

    assert before == nodes[:2]
    assert after == nodes[3:]


def test_fixed_windows_keep_interleaved_images_in_document_order(tmp_path):
    image_path = tmp_path / 'figure.png'
    image_path.write_bytes(b'image')
    nodes = [
        models.Node(type='paragraph', content='left'),
        models.Node(type='image', content='', image_path=str(image_path)),
        models.Node(type='paragraph', content='right'),
    ]

    windows = walker.fixed_windows_with_context(
        nodes, budget=100, backward_budget=0, forward_budget=0
    )

    assert len(windows) == 1
    assert [item.position for item in windows[0].items] == [0, 1, 2]
    assert [item.type for item in windows[0].items] == [
        'paragraph',
        'image',
        'paragraph',
    ]
    assert windows[0].items[1].image_path == str(image_path)


def test_fixed_windows_put_adjacent_images_in_multimodal_context(tmp_path):
    image_path = tmp_path / 'figure.png'
    image_path.write_bytes(b'image')
    nodes = [
        models.Node(type='image', content='', image_path=str(image_path)),
        models.Node(type='paragraph', content='target'),
        models.Node(type='image', content='', image_path=str(image_path)),
    ]

    windows = walker.fixed_windows_with_context(
        nodes, budget=2, backward_budget=100, forward_budget=100
    )

    assert len(windows) == 1
    assert [item.type for item in windows[0].items] == ['paragraph']
    assert [item.type for item in windows[0].before] == ['image']
    assert [item.type for item in windows[0].after] == ['image']
    assert windows[0].before[0].image_path == str(image_path)
    assert windows[0].after[0].image_path == str(image_path)


def test_fixed_windows_budget_is_deterministic_and_preserves_order():
    nodes = [models.Node(content=f'node {index}') for index in range(4)]

    windows = walker.fixed_windows_with_context(
        nodes, budget=4, backward_budget=0, forward_budget=0
    )

    assert [
        [item.content for item in window.items] for window in windows
    ] == [['node 0', 'node 1'], ['node 2', 'node 3']]
    assert [
        [item.position for item in window.items] for window in windows
    ] == [[0, 1], [0, 1]]


def test_text_only_context_helpers_remain_text_only():
    nodes = [
        models.Node(content='before'),
        models.Node(type='image', content='', image_path='figure.png'),
        models.Node(content='target'),
        models.Node(content='after'),
    ]

    assert walker.content_before(nodes, 2, 100) == 'before'
    assert walker.content_after(nodes, 2, 100) == 'after'
