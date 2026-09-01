import asyncio
from types import SimpleNamespace

import pytest

from kms.core import context_window, models, semantic


class _RecordingEnricher:
    def __init__(self, results_by_term):
        self.results_by_term = results_by_term
        self.calls = []

    async def aforward(self, **kwargs):
        self.calls.append(kwargs)
        return self.results_by_term[kwargs['request'].terms[0]]


def test_select_target_context_is_directional_and_text_only():
    nodes = [
        models.SourceNode(type='paragraph', content='before'),
        models.SourceNode(
            type='image',
            content='target diagram',
            assets=[models.VisualAsset(path='target.png')],
        ),
        models.SourceNode(type='paragraph', content='after'),
    ]

    before, target, after = context_window.select_target_context(
        nodes, position=1, before_budget=100, after_budget=100
    )

    assert [node.content for node in before] == ['before']
    assert target.content == 'target diagram'
    assert target.position == 0
    assert [node.content for node in after] == ['after']
    assert target.assets[0].path == 'target.png'

    projected = context_window.node_input(target)
    assert projected.node_type == 'image'
    assert projected.text == 'target diagram'
    assert not hasattr(projected, 'assets')


def test_select_target_context_preserves_empty_image_target():
    nodes = [
        models.SourceNode(type='paragraph', content='before'),
        models.SourceNode(type='image', content=None),
        models.SourceNode(type='paragraph', content='after'),
    ]

    _, target, _ = context_window.select_target_context(
        nodes, position=1, before_budget=100, after_budget=100
    )

    assert target.position == 0
    assert target.content is None


def test_describe_terms_requires_exact_ordered_one_to_one_results():
    valid = [
        SimpleNamespace(term='alpha', description='A'),
        SimpleNamespace(term='beta', description='B'),
    ]
    enricher = _RecordingEnricher({'alpha': [valid[0]], 'beta': [valid[1]]})
    result = asyncio.run(
        semantic.describe_terms(
            [models.SourceNode(type='paragraph', content='passage')],
            {0: {'beta', 'alpha'}},
            enricher,
            before_budget=10,
            after_budget=10,
            max_concurrency=1,
        )
    )
    assert result == {0: {'alpha': 'A', 'beta': 'B'}}
    assert [call['request'].terms for call in enricher.calls] == [
        ['alpha'],
        ['beta'],
    ]
    assert list(enricher.calls[0]) == ['request']

    cases = [
        [],
        valid + [SimpleNamespace(term='gamma', description='C')],
        [
            SimpleNamespace(term='alpha', description='A'),
            SimpleNamespace(term='alpha', description='A again'),
        ],
        [SimpleNamespace(term='beta', description='B')],
    ]
    for results in cases:
        enricher = _RecordingEnricher({'alpha': results})
        with pytest.raises(ValueError, match='position 0'):
            asyncio.run(
                semantic.describe_terms(
                    [models.SourceNode(type='paragraph', content='passage')],
                    {0: {'alpha'}},
                    enricher,
                    before_budget=10,
                    after_budget=10,
                    max_concurrency=1,
                )
            )


def test_context_node_projection_uses_standard_fields():
    node = context_window.ContextNode(
        position=4,
        type='image',
        content='A diagram',
        assets=[models.VisualAsset(path='figure.png')],
    )
    projected = context_window.node_input(node)
    assert projected.model_dump() == {
        'index': 1,
        'node_type': 'image',
        'text': 'A diagram',
    }
    assert not hasattr(projected, 'assets')
    assert not hasattr(projected, 'marker')
