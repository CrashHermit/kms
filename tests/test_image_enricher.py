import asyncio
from types import SimpleNamespace

import dspy
import pytest
from PIL import Image

from kms.construction import image_enricher
from kms.core import context_window, models

_PNG = b'\x89PNG\r\n\x1a\n'


def _image(path, content=None):
    return models.SourceNode(
        type=models.NodeType.IMAGE,
        content=content,
        assets=[models.VisualAsset(path=str(path))],
    )


def _write_image(path, color='red'):
    Image.new('RGB', (2, 2), color).save(path)


class _FakeEnricher:
    def __init__(self, description='A described figure.'):
        self.description = description
        self.calls = []

    async def aforward(self, **inputs):
        self.calls.append(inputs)
        return self.description


def test_context_text_preserves_text_and_omits_images(tmp_path):
    image_path = tmp_path / 'neighbor.png'
    _write_image(image_path)
    nodes = [
        models.SourceNode(type=models.NodeType.PARAGRAPH, content='before'),
        _image(image_path),
        models.SourceNode(type=models.NodeType.CAPTION, content='Figure 1.'),
    ]
    selected = context_window.select_around(
        nodes, [1], backward_budget=100, forward_budget=100
    )

    target_index = next(
        index for index, node in enumerate(selected) if node.marker == 'target'
    )

    assert image_enricher._context_text(selected[:target_index]) == (
        '[0] (paragraph): before'
    )
    assert image_enricher._context_text(selected[target_index + 1 :]) == (
        '[2] (caption): Figure 1.'
    )


def test_context_text_omits_neighboring_image_bytes(tmp_path):
    image_path = tmp_path / 'neighbor.png'
    _write_image(image_path)
    context = [
        context_window.ContextNode(
            position=0,
            type=models.NodeType.IMAGE,
            assets=[models.VisualAsset(path=str(image_path))],
        )
    ]

    assert (
        image_enricher._context_text(context) == '[0] (image): [IMAGE_OMITTED]'
    )


def test_image_enricher_encode_preserves_native_image_list():
    module = object.__new__(image_enricher.ImageEnricher)
    first = dspy.Image(url='data:image/png;base64,AAAA')
    second = dspy.Image(url='data:image/png;base64,BBBB')

    encoded = module.encode([first, second], 'before', 'after')

    assert encoded == {
        'images': [first, second],
        'context_before': 'before',
        'context_after': 'after',
    }


def test_dspy_formats_image_list_as_native_blocks():
    first = dspy.Image(url='data:image/png;base64,AAAA')
    second = dspy.Image(url='data:image/png;base64,BBBB')
    messages = dspy.ChatAdapter().format(
        image_enricher.ImageEnrichmentSignature,
        demos=[],
        inputs={
            'images': [first, second],
            'context_before': 'before',
            'context_after': 'after',
        },
    )

    blocks = messages[1]['content']
    assert sum(block['type'] == 'image_url' for block in blocks) == 2
    assert any('before' in block.get('text', '') for block in blocks)
    assert any('after' in block.get('text', '') for block in blocks)


def test_image_enricher_node_describes_each_post_seam_node(tmp_path):
    first_path = tmp_path / 'first.png'
    second_path = tmp_path / 'second.png'
    _write_image(first_path, 'red')
    _write_image(second_path, 'blue')
    target = models.SourceNode(
        type=models.NodeType.IMAGE,
        assets=[
            models.VisualAsset(path=str(first_path)),
            models.VisualAsset(path=str(second_path)),
        ],
    )
    nodes = [
        models.SourceNode(type=models.NodeType.PARAGRAPH, content='before'),
        target,
        models.SourceNode(type=models.NodeType.CAPTION, content='Figure 2.'),
    ]
    enricher = _FakeEnricher()
    stage = image_enricher.ImageEnrichmentNode(
        enricher,
        backward_budget=100,
        forward_budget=100,
        max_concurrent_calls=2,
    )

    result = asyncio.run(stage.run({'documents': [], 'nodes': nodes}))

    assert result['nodes'] is nodes
    assert target.content == 'A described figure.'
    assert [asset.path for asset in target.assets] == [
        str(first_path),
        str(second_path),
    ]
    assert len(enricher.calls) == 1
    assert len(enricher.calls[0]['images']) == 2
    assert enricher.calls[0]['context_before'] == '[0] (paragraph): before'
    assert enricher.calls[0]['context_after'] == '[2] (caption): Figure 2.'


def test_image_enricher_skips_non_images_and_assetless_images():
    assetless = models.SourceNode(type=models.NodeType.IMAGE)
    text = models.SourceNode(type=models.NodeType.PARAGRAPH, content='text')
    enricher = _FakeEnricher()
    stage = image_enricher.ImageEnrichmentNode(enricher)

    result = asyncio.run(
        stage.run({'documents': [], 'nodes': [text, assetless]})
    )

    assert result['nodes'] == [text, assetless]
    assert enricher.calls == []


def test_image_enricher_rejects_missing_asset_before_calls(tmp_path):
    missing = _image(tmp_path / 'missing.png')
    enricher = _FakeEnricher()
    stage = image_enricher.ImageEnrichmentNode(enricher)

    with pytest.raises(ValueError, match='image asset does not exist'):
        asyncio.run(stage.run({'documents': [], 'nodes': [missing]}))

    assert enricher.calls == []


def test_decode_rejects_empty_description():
    module = object.__new__(image_enricher.ImageEnricher)

    with pytest.raises(
        ValueError, match='description must be a non-empty string'
    ):
        module.decode(SimpleNamespace(description=''))
