import asyncio
from types import SimpleNamespace

import pytest
from PIL import Image

from kms.core import models, semantic


def _image(path):
    Image.new('RGB', (4, 2), (0, 0, 255)).save(path)


def _blocks(nodes):
    return semantic.window_content(
        nodes,
        position=1,
        before_budget=100,
        after_budget=100,
    ).openai_blocks()


def _describe(results_by_term):
    class _Enricher:
        async def aforward(self, **kwargs):
            return results_by_term[kwargs['terms'][0]]

    return asyncio.run(
        semantic.describe_terms(
            [models.Node(type='paragraph', content='passage')],
            {0: set(results_by_term)},
            _Enricher(),
            before_budget=10,
            after_budget=10,
            max_concurrency=1,
        )
    )


def test_describe_terms_requires_exact_ordered_one_to_one_results():
    valid = [
        SimpleNamespace(term='alpha', description='A'),
        SimpleNamespace(term='beta', description='B'),
    ]
    assert _describe({'alpha': [valid[0]], 'beta': [valid[1]]}) == {
        0: {'alpha': 'A', 'beta': 'B'}
    }

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
        with pytest.raises(ValueError, match='position 0'):
            _describe({'alpha': results})


def test_window_content_keeps_target_image_between_neighbors(tmp_path):
    target_path = tmp_path / 'target.png'
    _image(target_path)
    nodes = [
        models.Node(type='paragraph', content='before'),
        models.Node(type='image', content='', image_path=str(target_path)),
        models.Node(type='paragraph', content='after'),
    ]

    blocks = _blocks(nodes)

    assert [block['type'] for block in blocks] == [
        'text',
        'text',
        'image_url',
        'text',
    ]
    assert [block.get('text') for block in blocks if block['type'] == 'text'] == [
        '[0] (paragraph): before',
        '[1] (image) <target>:',
        '[2] (paragraph): after',
    ]


def test_window_content_keeps_neighbor_images_in_order(tmp_path):
    before_path = tmp_path / 'before.png'
    after_path = tmp_path / 'after.png'
    _image(before_path)
    _image(after_path)
    nodes = [
        models.Node(type='image', image_path=str(before_path)),
        models.Node(type='paragraph', content='target'),
        models.Node(type='image', image_path=str(after_path)),
    ]

    blocks = _blocks(nodes)

    assert [block['type'] for block in blocks] == [
        'text',
        'image_url',
        'text',
        'text',
        'image_url',
    ]
    assert blocks[0]['text'] == '[0] (image):'
    assert blocks[2]['text'] == '[1] (paragraph) <target>: target'
    assert blocks[3]['text'] == '[2] (image):'


def test_window_content_keeps_empty_target_image(tmp_path):
    image_path = tmp_path / 'target.png'
    _image(image_path)
    nodes = [
        models.Node(type='paragraph', content='before'),
        models.Node(type='image', image_path=str(image_path)),
        models.Node(type='paragraph', content='after'),
    ]

    blocks = _blocks(nodes)

    assert blocks[2]['type'] == 'image_url'
    assert blocks[2]['image_url']['url'].startswith('data:image/png;base64,')


def test_window_content_preserves_multiple_neighbor_images(tmp_path):
    paths = [tmp_path / f'image-{index}.png' for index in range(3)]
    for path in paths:
        _image(path)
    nodes = [
        models.Node(type='image', image_path=str(paths[0])),
        models.Node(type='image', image_path=str(paths[1])),
        models.Node(type='paragraph', content='target'),
        models.Node(type='image', image_path=str(paths[2])),
    ]

    blocks = semantic.window_content(
        nodes, position=2, before_budget=100, after_budget=100
    ).openai_blocks()

    assert [block['type'] for block in blocks] == [
        'text',
        'image_url',
        'text',
        'image_url',
        'text',
        'text',
        'image_url',
    ]
    assert [
        block.get('text') for block in blocks if block['type'] == 'text'
    ] == [
        '[0] (image):',
        '[1] (image):',
        '[2] (paragraph) <target>: target',
        '[3] (image):'
    ]
