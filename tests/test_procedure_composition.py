"""Tests for pure procedure composition."""

import pytest

from kms.construction import procedure_inputs
from kms.construction.composition import compose_procedure
from kms.core import identity, models


def _bundle() -> models.ConstructionBundle:
    source = models.Source(key='book')
    nodes = [
        models.Node(id=10, type=models.NodeType.PARAGRAPH, content='intro'),
        models.Node(id=20, type=models.NodeType.IMAGE, image_path='figure.png'),
        models.Node(id=30, type=models.NodeType.PARAGRAPH, content='finish'),
    ]
    statement = models.Statement(block=[1, 2], members=[10, 20])
    procedure = models.Procedure(block=[1, 2], members=[10, 20])
    identity.assign_statement_procedure_ids([statement], [procedure], 'book')
    return models.ConstructionBundle(
        source=source,
        nodes=nodes,
        statements=[statement],
        procedures=[procedure],
    )


def test_compose_procedure_preserves_ordered_content_and_steps() -> None:
    procedure = models.Procedure(
        block=[1, 2],
        members=[30, 10, 20],
        steps=[models.Step('second', index=2), models.Step('first', index=0)],
    )

    composed = compose_procedure(_bundle(), procedure)

    assert [part.node_id for part in composed.content.parts] == [10, 20, 30]
    assert composed.content.text == 'intro\n\nfinish'
    assert [step.index for step in composed.steps] == [0, 2]
    assert [step.text for step in composed.steps] == ['first', 'second']
    assert composed.statement is not None
    assert composed.statement.text == 'intro'


def test_compose_procedure_supports_empty_content_and_orphan_statement() -> None:
    bundle = models.ConstructionBundle(source=models.Source(key='book'))
    procedure = models.Procedure(block=[9], members=[], steps=[models.Step('do')])

    composed = compose_procedure(bundle, procedure)

    assert composed.content.parts == ()
    assert composed.content.text == ''
    assert composed.statement is None
    assert [step.text for step in composed.steps] == ['do']


def test_compose_procedure_rejects_bad_members_and_steps() -> None:
    with pytest.raises(ValueError, match='duplicate member ids'):
        compose_procedure(
            _bundle(), models.Procedure(block=[1], members=[10, 10])
        )

    with pytest.raises(ValueError, match='missing node ids'):
        compose_procedure(
            _bundle(), models.Procedure(block=[1], members=[999])
        )

    with pytest.raises(ValueError, match='negative'):
        compose_procedure(
            _bundle(), models.Procedure(block=[1], steps=[models.Step('bad', -1)])
        )

    with pytest.raises(ValueError, match='duplicate indexes'):
        compose_procedure(
            _bundle(),
            models.Procedure(
                block=[1],
                steps=[models.Step('a', 0), models.Step('b', 0)],
            ),
        )


def test_compose_procedure_rejects_ambiguous_statement_linkage() -> None:
    bundle = _bundle()
    bundle.statements.append(models.Statement(block=[1, 2], members=[30]))

    with pytest.raises(ValueError, match='ambiguous'):
        compose_procedure(bundle, models.Procedure(block=[1, 2]))


def test_materialization_inputs_resolve_ids_and_attach_by_statement_block(
    monkeypatch,
) -> None:
    monkeypatch.setattr(procedure_inputs.content, 'load_image', lambda path: None)
    bundle = _bundle()
    procedure = models.Procedure(block=[1, 2], members=[30, 10])
    identity.assign_statement_procedure_ids(bundle.statements, [procedure], 'book')
    bundle.procedures = [procedure]

    inputs = procedure_inputs.build_procedure_materialization_inputs(bundle)

    assert len(inputs) == 2
    statement_input, procedure_input = inputs
    assert statement_input.statement_uuid == (
        identity.statement_uuid('book', [1, 2])
    )
    assert procedure_input.procedure_uuid == (
        identity.procedure_uuid(
            'book',
            [1, 2],
            0,
            statement_uuid_value=identity.statement_uuid('book', [1, 2]),
        )
    )
    assert procedure_input.member_count == 2
    assert procedure_input.procedure is not None


def test_compose_procedure_does_not_mutate_inputs() -> None:
    bundle = _bundle()
    procedure = models.Procedure(
        block=[1, 2],
        members=[20, 10],
        steps=[models.Step('step', 3)],
    )
    before_bundle = repr(bundle)
    before_procedure = repr(procedure)

    compose_procedure(bundle, procedure)

    assert repr(bundle) == before_bundle
    assert repr(procedure) == before_procedure
