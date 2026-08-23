import asyncio
from types import SimpleNamespace

import pytest

from kms.construction import block_corrector
from kms.core import models


class _Corrector:
    async def acorrect(self, region):
        assert region.block.type == 'text'
        return {'corrected_text': 'corrected text', 'edits': []}


def _document():
    return models.Document(
        index=3,
        image_path='page.png',
        content='original text',
        nodes=[
            models.Node(
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


def test_editor_rejects_duplicate_line_indices():
    edits = [
        block_corrector.LineEdit(index=1, replacement='A'),
        block_corrector.LineEdit(index=1, replacement='B'),
    ]
    with pytest.raises(ValueError, match='duplicate line indices'):
        block_corrector.BlockCorrectionEditor.decode(
            None,
            SimpleNamespace(edits=edits),
            original_text='a\nb',
            block_type='text',
        )


def test_collect_writes_corrected_nodes_to_documents():
    document = _document()
    node = block_corrector.BlockCorrectorNode(_Corrector())
    result = node.collect(
        {
            'documents': [document],
            'block_correction_results': [
                (3, [models.Node(content='corrected text')])
            ],
        }
    )
    assert result['documents'][0].nodes[0].content == 'corrected text'
