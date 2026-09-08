"""Tests for pure procedure composition."""

import pytest

from kms.construction.composition import compose_procedure
from kms.core import identity, models


def _bundle() -> models.ConstructionBundle:
    source = models.Source(key='book')
    nodes = [
        models.SourceNode(
            uuid='node-0', type=models.NodeType.PARAGRAPH, content='intro'
        ),
        models.SourceNode(
            uuid='node-1',
            type=models.NodeType.IMAGE,
            assets=[models.VisualAsset(path='figure.png')],
        ),
        models.SourceNode(
            uuid='node-2', type=models.NodeType.PARAGRAPH, content='finish'
        ),
    ]
    statement = models.Statement(block=[0, 1], member_positions=[0, 1])
    procedure = models.Procedure(block=[0, 1], member_positions=[0, 1])
    identity.assign_statement_procedure_ids([statement], [procedure], 'book')
    return models.ConstructionBundle(
        source=source,
        nodes=nodes,
        statements=[statement],
        procedures=[procedure],
    )


def test_compose_procedure_preserves_ordered_content() -> None:
    procedure = models.Procedure(
        block=[0, 1],
        member_positions=[0, 1, 2],
    )

    composed = compose_procedure(_bundle(), procedure)

    assert [part.position for part in composed.content.parts] == [0, 1, 2]
    assert composed.content.text == 'intro\n\nfinish'
    assert composed.statement is not None
    assert composed.statement.text == 'intro'


def test_compose_procedure_supports_empty_content_and_orphan_statement() -> (
    None
):
    bundle = models.ConstructionBundle(source=models.Source(key='book'))
    procedure = models.Procedure(block=[9], member_positions=[])

    composed = compose_procedure(bundle, procedure)

    assert composed.content.parts == ()
    assert composed.content.text == ''
    assert composed.statement is None


def test_compose_procedure_rejects_bad_members() -> None:
    with pytest.raises(ValueError, match='duplicate member positions'):
        compose_procedure(
            _bundle(), models.Procedure(block=[1], member_positions=[0, 0])
        )

    with pytest.raises(ValueError, match='missing node positions'):
        compose_procedure(
            _bundle(), models.Procedure(block=[1], member_positions=[999])
        )


def test_compose_procedure_rejects_ambiguous_statement_linkage() -> None:
    bundle = _bundle()
    # Add a statement with the same block as the procedure to create ambiguity
    bundle.statements.append(
        models.Statement(block=[0, 1], member_positions=[2])
    )
    # Create a procedure without statement_uuid so it falls through to ambiguous check
    procedure = models.Procedure(block=[0, 1], member_positions=[0, 1])

    with pytest.raises(ValueError, match='ambiguous'):
        compose_procedure(bundle, procedure)


def test_procedure_enrichment_inputs_are_statement_centered() -> None:
    from kms.construction import procedure_enrichment

    bundle = _bundle()
    inputs = procedure_enrichment.build_procedure_inputs(bundle)

    assert len(inputs) == 1
    assert inputs[0].statement_uuid == identity.statement_uuid('book', [0, 1])
    assert inputs[0].procedure_uuid == bundle.procedures[0].uuid
    assert inputs[0].procedure is not None


def test_compose_procedure_does_not_mutate_inputs() -> None:
    bundle = _bundle()
    procedure = models.Procedure(
        block=[0, 1],
        member_positions=[2, 0],
    )
    before_bundle = repr(bundle)
    before_procedure = repr(procedure)

    compose_procedure(bundle, procedure)

    assert repr(bundle) == before_bundle
    assert repr(procedure) == before_procedure
