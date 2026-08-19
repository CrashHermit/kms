import pytest

from kms.core import models, semantic, walker


def test_position_for_id_resolves_nonpositional_ids():
    nodes = [
        models.Node(id=10, content='first'),
        models.Node(id=25, content='second'),
    ]
    assert walker.position_for_id(nodes, 25) == 1


def test_window_content_uses_stable_node_ids():
    nodes = [
        models.Node(id=10, content='first'),
        models.Node(id=25, content='second'),
    ]
    result = semantic.window_content(nodes, 25, 100, 100)
    assert 'second' in result.render()
    assert 'first' in result.render()


def test_position_for_id_reports_missing_ids():
    with pytest.raises(KeyError, match='node id 99'):
        walker.position_for_id([models.Node(id=10)], 99)
