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
