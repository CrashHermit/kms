"""Tests for pure source-local composition."""

import pytest

from kms.construction.composition import compose_statement
from kms.core import models


def _bundle() -> models.ConstructionBundle:
    source = models.Source(key='book')
    nodes = [
        models.Node(
            id=0,
            type=models.NodeType.PARAGRAPH,
            content='first',
            document_index=2,
        ),
        models.Node(
            id=1,
            type=models.NodeType.IMAGE,
            content='![99]()',
            document_index=2,
            image_path='images/actual.png',
        ),
        models.Node(
            id=2,
            type=models.NodeType.PARAGRAPH,
            content=' second ',
            document_index=2,
        ),
    ]
    return models.ConstructionBundle(source=source, nodes=nodes)


def test_compose_statement_uses_members_and_document_order() -> None:
    bundle = _bundle()
    statement = models.Statement(block=[0, 1, 2], members=[2, 0])

    composed = compose_statement(bundle, statement)

    assert [part.node_id for part in composed.parts] == [0, 2]
    assert composed.text == 'first\n\n second '


def test_compose_statement_preserves_interleaved_image_provenance() -> None:
    bundle = _bundle()
    statement = models.Statement(block=[0, 1, 2], members=[2, 1, 0])

    composed = compose_statement(bundle, statement)

    assert [part.node_id for part in composed.parts] == [0, 1, 2]
    assert composed.parts[1].content == '![99]()'
    assert composed.parts[1].image_path == 'images/actual.png'
    assert composed.parts[1].document_index == 2
    assert composed.pictures == [
        {'index': 0, 'document_index': 2, 'image_path': 'images/actual.png'}
    ]


def test_compose_statement_supports_empty_and_image_only_statements() -> None:
    bundle = _bundle()

    assert compose_statement(bundle, models.Statement(block=[0], members=[])).text == ''
    image_only = compose_statement(
        bundle, models.Statement(block=[1], members=[1])
    )

    assert image_only.text == ''
    assert len(image_only.pictures) == 1


def test_compose_statement_rejects_missing_member_ids() -> None:
    with pytest.raises(ValueError, match='missing node ids'):
        compose_statement(
            _bundle(), models.Statement(block=[0], members=[999])
        )


def test_compose_statement_does_not_mutate_inputs() -> None:
    bundle = _bundle()
    statement = models.Statement(block=[0, 1], members=[1, 0])
    before_nodes = repr(bundle.nodes)
    before_statement = repr(statement)

    compose_statement(bundle, statement)

    assert repr(bundle.nodes) == before_nodes
    assert repr(statement) == before_statement
