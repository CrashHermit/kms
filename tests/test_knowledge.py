"""Tests for source-local canonical knowledge selection."""

import pytest

from kms.construction.knowledge import (
    knowledge_for_procedure,
    knowledge_for_statement,
)
from kms.core import models


def _assertion(uuid: str, evidence: set[int]) -> models.KnowledgeAssertion:
    concept = models.CanonicalConcept('concept', 'concept', 'a concept')
    predicate = models.CanonicalPredicate('predicate', 'relates', 'a relation')
    return models.KnowledgeAssertion(
        uuid=uuid,
        name=uuid,
        description=f'description {uuid}',
        subject=concept,
        predicate=predicate,
        object=concept,
        evidence_node_ids=frozenset(evidence),
    )


def _bundle() -> models.ConstructionBundle:
    return models.ConstructionBundle(source=models.Source(key='book'))


def test_knowledge_selection_uses_members_not_blocks() -> None:
    index = models.KnowledgeIndex(
        source='book',
        assertions=(
            _assertion('supported', {10}),
            _assertion('block-only', {99}),
        ),
    )
    statement = models.Statement(block=[99], member_positions=[10])

    result = knowledge_for_statement(_bundle(), statement, index)

    assert [assertion.uuid for assertion in result.assertions] == ['supported']


def test_procedure_selection_uses_procedure_members() -> None:
    index = models.KnowledgeIndex(
        source='book', assertions=(_assertion('fact', {20}),)
    )
    procedure = models.Procedure(block=[20], member_positions=[20])

    result = knowledge_for_procedure(_bundle(), procedure, index)

    assert [assertion.uuid for assertion in result.assertions] == ['fact']


def test_empty_members_return_empty_knowledge() -> None:
    index = models.KnowledgeIndex(
        source='book', assertions=(_assertion('fact', {1}),)
    )

    result = knowledge_for_statement(
        _bundle(), models.Statement(block=[1], member_positions=[]), index
    )

    assert result.assertions == ()


def test_cross_source_selection_is_rejected() -> None:
    index = models.KnowledgeIndex(source='other', assertions=())

    with pytest.raises(ValueError, match='does not match'):
        knowledge_for_statement(
            _bundle(), models.Statement(block=[1], member_positions=[1]), index
        )


def test_knowledge_union_deduplicates_and_merges_evidence() -> None:
    left = models.Knowledge((_assertion('same', {1}),))
    right = models.Knowledge(
        (_assertion('same', {2}), _assertion('other', {3}))
    )

    result = left.union(right)

    assert [assertion.uuid for assertion in result.assertions] == [
        'other',
        'same',
    ]
    assert result.assertions[1].evidence_node_ids == frozenset({1, 2})
    assert 'concept --relates--> concept: description same' in result.render()
