import asyncio
from types import SimpleNamespace

import pytest
from PIL import Image

from kms.construction import image_seam_merger
from kms.core import models


def _segment(index, nodes):
    return models.Document(index=index, image_path='', nodes=nodes)


def _image(path, content=None):
    return models.SourceNode(
        type=models.NodeType.IMAGE,
        content=content,
        assets=[models.VisualAsset(path=str(path))],
    )


class _Judge:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def aforward(self, **inputs):
        self.calls.append(inputs)
        return self.result


async def _merge(top, bottom, judge):
    return await image_seam_merger._merge_pair(judge, top, bottom)


def test_image_edges_select_images_around_text():
    top = _segment(0, [models.SourceNode(content='top'), _image('a.png')])
    bottom = _segment(1, [_image('b.png'), models.SourceNode(content='bottom')])

    pairs = image_seam_merger._pairs([top, bottom], parity=0)

    assert pairs == [(top, bottom)]


def test_image_edges_skip_apparatus_only():
    top = _segment(
        0,
        [
            models.SourceNode(
                type=models.NodeType.BIBLIOGRAPHIC, content='ref'
            ),
            _image('a.png'),
        ],
    )
    bottom = _segment(
        1,
        [
            _image('b.png'),
            models.SourceNode(type=models.NodeType.NOTE, content='note'),
        ],
    )

    assert image_seam_merger._pairs([top, bottom], parity=0) == [(top, bottom)]


def test_text_edges_do_not_get_skipped_to_find_images():
    top = _segment(0, [models.SourceNode(content='tail'), _image('a.png')])
    bottom = _segment(1, [models.SourceNode(content='head')])

    assert image_seam_merger._pairs([top, bottom], parity=0) == []


def test_multiple_assets_and_image_content_are_ineligible():
    multiple = models.SourceNode(
        type=models.NodeType.IMAGE,
        assets=[
            models.VisualAsset(path='a.png'),
            models.VisualAsset(path='b.png'),
        ],
    )
    with_content = _image('c.png', content='caption')
    bottom = _segment(1, [_image('bottom.png')])

    assert (
        image_seam_merger._pairs([_segment(0, [multiple]), bottom], parity=0)
        == []
    )
    assert (
        image_seam_merger._pairs(
            [_segment(0, [with_content]), bottom], parity=0
        )
        == []
    )


def test_false_judge_preserves_both_image_nodes():
    top = _segment(0, [_image('a.png')])
    bottom = _segment(1, [_image('b.png')])
    judge = _Judge(False)

    result = asyncio.run(_merge(top, bottom, judge))

    assert len(judge.calls) == 1
    assert [asset.path for asset in result[0][1][0].assets] == ['a.png']
    assert [asset.path for asset in result[1][1][0].assets] == ['b.png']


def test_true_judge_appends_assets_and_removes_bottom_node():
    top = _segment(0, [_image('a.png')])
    bottom = _segment(1, [_image('b.png')])
    judge = _Judge(True)

    result = asyncio.run(_merge(top, bottom, judge))

    assert [asset.path for asset in result[0][1][0].assets] == [
        'a.png',
        'b.png',
    ]
    assert result[1][1] == []


def test_encode_loads_two_images(tmp_path):
    top_path = tmp_path / 'top.png'
    bottom_path = tmp_path / 'bottom.png'
    Image.new('RGB', (2, 2), 'red').save(top_path)
    Image.new('RGB', (2, 2), 'blue').save(bottom_path)
    merger = object.__new__(image_seam_merger.ImageSeamMerger)

    encoded = merger.encode(_image(top_path), _image(bottom_path))

    assert set(encoded) == {'top_image', 'bottom_image'}
    assert all(
        isinstance(value, image_seam_merger.dspy.Image)
        for value in encoded.values()
    )


def test_encode_rejects_missing_image(tmp_path):
    merger = object.__new__(image_seam_merger.ImageSeamMerger)
    missing = _image(tmp_path / 'missing.png')
    existing_path = tmp_path / 'existing.png'
    Image.new('RGB', (2, 2), 'red').save(existing_path)

    with pytest.raises(ValueError, match='missing.png'):
        merger.encode(missing, _image(existing_path))


def test_decode_requires_boolean_decision():
    merger = object.__new__(image_seam_merger.ImageSeamMerger)

    assert merger.decode(SimpleNamespace(is_continuation=True)) is True
    with pytest.raises(ValueError, match='is_continuation'):
        merger.decode(SimpleNamespace(is_continuation='true'))
