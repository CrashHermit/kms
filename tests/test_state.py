"""Tests for projecting workflow state into construction data."""

import pytest

from kms.core import models, state


def _state() -> state.State:
    source = models.Source(
        key='book',
        metadata={'title': 'A book'},
        documents=[models.Document(index=0, image_path='page.png')],
    )
    return {
        'pdf_path': 'book.pdf',
        'output_dir': 'output',
        'source': source,
        'documents': source.documents,
        'nodes': [models.SourceNode(uuid='node-10', document_index=0, content='text')],
        'instructions': [models.Instruction(block=[0], member_positions=[0])],
        'statements': [models.Statement(block=[1], member_positions=[0])],
        'procedures': [models.Procedure(block=[2], member_positions=[0])],
        'triplets': [models.Triplet('subject', 'predicate', 'object', evidence_positions=[0])],
        'entity_descriptions': {0: {'term': None}},
        'predicate_descriptions': {0: {'term': 'relation'}},
        'entity_embeddings': {0: {'model': [1.0, 2.0]}},
        'predicate_embeddings': {0: {'model': [3.0, 4.0]}},
        'spans': [[0, 1]],
    }


def test_to_construction_bundle_projects_durable_data_only() -> None:
    current_state = _state()

    bundle = state.to_construction_bundle(current_state)

    assert bundle.source.key == 'book'
    assert bundle.source.metadata == {'title': 'A book'}
    assert bundle.nodes == current_state['nodes']
    assert bundle.instructions == current_state['instructions']
    assert bundle.statements == current_state['statements']
    assert bundle.procedures == current_state['procedures']
    assert bundle.triplets == current_state['triplets']
    assert bundle.entity_descriptions == {0: {'term': None}}
    assert bundle.predicate_descriptions == {0: {'term': 'relation'}}
    assert bundle.entity_embeddings == {0: {'model': [1.0, 2.0]}}
    assert bundle.predicate_embeddings == {0: {'model': [3.0, 4.0]}}


def test_to_construction_bundle_returns_a_valid_bundle() -> None:
    bundle = state.to_construction_bundle(_state())

    models.validate_bundle(bundle)


def test_to_construction_bundle_uses_empty_defaults() -> None:
    bundle = state.to_construction_bundle(
        {'source': models.Source(key='empty')}
    )

    assert bundle.source.key == 'empty'
    assert bundle.nodes == []
    assert bundle.instructions == []
    assert bundle.statements == []
    assert bundle.procedures == []
    assert bundle.triplets == []
    assert bundle.entity_descriptions == {}
    assert bundle.predicate_descriptions == {}
    assert bundle.entity_embeddings == {}
    assert bundle.predicate_embeddings == {}


def test_to_construction_bundle_requires_source_identity() -> None:
    with pytest.raises(ValueError, match='non-empty source'):
        state.to_construction_bundle({})

    with pytest.raises(ValueError, match='non-empty source'):
        state.to_construction_bundle({'source': models.Source(key='  ')})


def test_to_construction_bundle_does_not_alias_state_collections() -> None:
    current_state = _state()
    bundle = state.to_construction_bundle(current_state)

    bundle.nodes.clear()
    bundle.entity_descriptions[0]['term'] = 'changed'
    bundle.entity_embeddings[0]['model'].append(5.0)

    assert current_state['nodes']
    assert current_state['entity_descriptions'] == {0: {'term': None}}
    assert current_state['entity_embeddings'] == {0: {'model': [1.0, 2.0]}}
