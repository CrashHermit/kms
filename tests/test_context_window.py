import pytest

from kms.core import context_window, models


def _nodes():
    return [
        models.SourceNode(uuid=f'node-{index}', content=f'node {index}')
        for index in range(5)
    ]


def test_select_around_keeps_targets_and_adds_context_per_side():
    window = context_window.select_around(
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


def test_select_around_zero_budget_is_directional():
    nodes = _nodes()
    backward_only = context_window.select_around(
        nodes, [2], backward_budget=100, forward_budget=0
    )
    forward_only = context_window.select_around(
        nodes, [2], backward_budget=0, forward_budget=100
    )

    assert [node.position for node in backward_only] == [0, 1, 2]
    assert [node.position for node in forward_only] == [0, 1, 2]


def test_select_around_rejects_empty_targets():
    with pytest.raises(ValueError, match='at least one target'):
        context_window.select_around(_nodes(), [])


def test_select_around_preserves_image_assets_for_callers(tmp_path):
    image_path = tmp_path / 'figure.png'
    image_path.write_bytes(b'image')
    nodes = [
        models.SourceNode(type='paragraph', content='before'),
        models.SourceNode(
            type='image',
            content=None,
            assets=[models.VisualAsset(path=str(image_path))],
        ),
        models.SourceNode(type='paragraph', content='after'),
    ]

    window = context_window.select_around(
        nodes,
        [1],
        backward_budget=100,
        forward_budget=100,
    )

    assert [node.type for node in window] == ['paragraph', 'image', 'paragraph']
    assert window[1].assets[0].path == str(image_path)
    assert window[1].marker == 'target'


def test_nodes_before_and_after_retain_image_nodes(tmp_path):
    image_path = tmp_path / 'figure.png'
    image_path.write_bytes(b'image')
    nodes = [
        models.SourceNode(content='before'),
        models.SourceNode(
            type='image',
            content=None,
            assets=[models.VisualAsset(path=str(image_path))],
        ),
        models.SourceNode(content='target'),
        models.SourceNode(
            type='image',
            content=None,
            assets=[models.VisualAsset(path=str(image_path))],
        ),
        models.SourceNode(content='after'),
    ]

    before = context_window.nodes_before(nodes, cursor=2, budget=100)
    after = context_window.nodes_after(nodes, cursor=2, budget=100)

    assert before == nodes[:2]
    assert after == nodes[3:]


def test_select_target_context_keeps_target_and_bounds_each_direction():
    nodes = [
        models.SourceNode(content='a' * 7),
        models.SourceNode(content='b' * 7),
        models.SourceNode(type='image', content='target' * 20),
        models.SourceNode(content='c' * 7),
        models.SourceNode(content='d' * 7),
    ]

    before, target, after = context_window.select_target_context(
        nodes, position=2, before_budget=2, after_budget=2
    )

    assert [node.content for node in before] == ['b' * 7]
    assert target.content == 'target' * 20
    assert target.position == 0
    assert [node.content for node in after] == ['c' * 7]
    assert [node.position for node in before] == [0]
    assert [node.position for node in after] == [0]
