import asyncio
from types import SimpleNamespace

import pytest

from kms.construction import block_corrector
from kms.core import edits, models


class _Corrector:
    async def acorrect(self, region):
        assert region.block.type == 'text'
        return {'corrected_text': 'corrected text', 'edits': []}


def _document():
    return models.Document(
        index=3,
        image_path='page.png',
        nodes=[
            models.SourceNode(
                index=0,
                type=models.NodeType.PARAGRAPH,
                content='original text',
                provenance={
                    'provider_type': 'text',
                    'crop_path': 'crop.png',
                },
            )
        ],
    )


def test_dispatch_only_sends_documents_with_crops():
    node = block_corrector.BlockCorrectorNode(_Corrector())
    sends = node.dispatch({'documents': [_document()]})
    assert len(sends) == 1
    assert sends[0].arg['document'].index == 3


def test_worker_corrects_canonical_node_content():
    node = block_corrector.BlockCorrectorNode(_Corrector())
    result = asyncio.run(node.worker({'document': _document()}))
    nodes = result['block_correction_results'][0][1]
    assert nodes[0].content == 'corrected text'


def test_router_and_editor_encode_structured_one_based_lines(monkeypatch):
    monkeypatch.setattr(
        block_corrector.images, 'load_image', lambda _path: 'image'
    )
    router = block_corrector.BlockCorrectionRouter.__new__(
        block_corrector.BlockCorrectionRouter
    )
    editor = block_corrector.BlockCorrectionEditor.__new__(
        block_corrector.BlockCorrectionEditor
    )
    expected = [models.LineInput(index=1, text='a')]
    assert router.encode('crop.png', 'text', 'a')['lines'] == expected
    encoded = editor.encode('crop.png', 'text', 'a', [1])
    assert encoded['lines'] == expected
    assert encoded['allowed_indexes'] == [1]


def test_editor_accepts_index_one_replacement():
    replacement = edits.LineReplacement(index=1, replacement='A')
    result = block_corrector.BlockCorrectionEditor.decode(
        None,
        SimpleNamespace(edits=[replacement]),
        original_text='a',
        block_type='text',
        allowed_indexes=[1],
    )
    assert result == [replacement]
    assert edits.apply_line_replacements('a', result) == 'A'


def test_editor_accepts_multiline_replacement():
    replacement = edits.LineReplacement(index=2, replacement='B\nC')
    result = block_corrector.BlockCorrectionEditor.decode(
        None,
        SimpleNamespace(edits=[replacement]),
        original_text='a\nb',
        block_type='text',
        allowed_indexes=[2],
    )
    assert edits.apply_line_replacements('a\nb', result) == 'a\nB\nC'


def test_locator_allows_empty_locations_as_conservative_no_op():
    result = block_corrector.BlockCorrectionLocator.decode(
        None,
        SimpleNamespace(locations=[]),
        original_text='faithful',
        block_type='text',
    )
    assert result == []


def test_editor_allows_empty_replacements_as_conservative_no_op():
    result = block_corrector.BlockCorrectionEditor.decode(
        None,
        SimpleNamespace(edits=[]),
        original_text='faithful',
        block_type='text',
        allowed_indexes=[1],
    )
    assert result == []
def test_locator_accepts_structured_one_based_location():
    location = models.LineSelection(index=1)
    result = block_corrector.BlockCorrectionLocator.decode(
        None,
        SimpleNamespace(locations=[location]),
        original_text='one line',
        block_type='text',
    )
    assert result == [location]


@pytest.mark.parametrize('index', [0, 2])
def test_locator_rejects_invalid_one_line_indices(index):
    with pytest.raises(ValueError, match=r'valid range=1\.\.1'):
        block_corrector.BlockCorrectionLocator.decode(
            None,
            SimpleNamespace(locations=[models.LineSelection(index=index)]),
            original_text='one line',
            block_type='text',
        )


def test_editor_rejects_unauthorized_replacement_index():
    replacement = edits.LineReplacement(index=2, replacement='wrong')
    with pytest.raises(ValueError, match='allowed_indexes'):
        block_corrector.BlockCorrectionEditor.decode(
            None,
            SimpleNamespace(edits=[replacement]),
            original_text='a\nb',
            block_type='text',
            allowed_indexes=[1],
        )


def test_staged_corrector_passes_localized_indexes_to_editor():
    class Router:
        async def needs_correction(self, _region):
            return True

    class Locator:
        async def locate(self, _region):
            return [models.LineSelection(index=1)]

    class Editor:
        def __init__(self):
            self.allowed_indexes = None

        async def aforward(self, **inputs):
            self.allowed_indexes = inputs['allowed_indexes']
            return [edits.LineReplacement(index=1, replacement='corrected')]

    corrector = block_corrector.BlockCorrector.__new__(
        block_corrector.BlockCorrector
    )
    corrector.router = Router()
    corrector.locator = Locator()
    corrector.editor = Editor()
    region = SimpleNamespace(
        crop_path='crop.png',
        block=SimpleNamespace(type='text', content='original'),
    )
    result = asyncio.run(corrector.acorrect(region))
    assert corrector.editor.allowed_indexes == [1]
    assert result['corrected_text'] == 'corrected'
    assert len(result['edits']) == 1


def test_staged_corrector_router_false_skips_downstream():
    class Router:
        async def needs_correction(self, _region):
            return False

    class UnexpectedStage:
        async def locate(self, _region):
            raise AssertionError('locator should be skipped')

        async def aforward(self, **_inputs):
            raise AssertionError('editor should be skipped')

    corrector = block_corrector.BlockCorrector.__new__(
        block_corrector.BlockCorrector
    )
    corrector.router = Router()
    corrector.locator = UnexpectedStage()
    corrector.editor = UnexpectedStage()
    region = SimpleNamespace(
        crop_path='crop.png',
        block=SimpleNamespace(type='text', content='original'),
    )
    assert asyncio.run(corrector.acorrect(region)) == {
        'corrected_text': 'original',
        'edits': [],
    }

def test_editor_rejects_duplicate_line_indices():
    replacements = [
        edits.LineReplacement(index=1, replacement='A'),
        edits.LineReplacement(index=1, replacement='B'),
    ]
    with pytest.raises(ValueError, match='duplicate line indices'):
        block_corrector.BlockCorrectionEditor.decode(
            None,
            SimpleNamespace(edits=replacements),
            original_text='a\nb',
            block_type='text',
            allowed_indexes=[1],
        )


@pytest.mark.parametrize('index', [0, 2])
def test_editor_rejects_invalid_one_line_indices(index):
    replacement = edits.LineReplacement(index=index, replacement='wrong')
    with pytest.raises(ValueError, match=r'valid range=1\.\.1'):
        block_corrector.BlockCorrectionEditor.decode(
            None,
            SimpleNamespace(edits=[replacement]),
            original_text='one line',
            block_type='text',
            allowed_indexes=[index],
        )


def test_collect_writes_corrected_nodes_to_documents():
    document = _document()
    node = block_corrector.BlockCorrectorNode(_Corrector())
    result = node.collect(
        {
            'documents': [document],
            'block_correction_results': [
                (3, [models.SourceNode(content='corrected text')])
            ],
        }
    )
    assert result['documents'][0].nodes[0].content == 'corrected text'
