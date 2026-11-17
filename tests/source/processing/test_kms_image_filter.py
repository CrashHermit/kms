from __future__ import annotations

import asyncio

from PIL import Image

from kms.config.inference import ContextWindowSettings
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.budget import ImagePayloadBudget
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.processing.image_filter.models import ImageFilterDecision
from kms.source.processing.image_filter.node import ImageFilterNode
from kms.source.processing.state import SourceProcessingState


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _Filter:
    async def acall(self, *, items):
        return [
            ImageFilterDecision(
                position=item.position,
                keep=item.target.text != 'furniture',
            )
            for item in reversed(items)
        ]


def _node() -> ImageFilterNode:
    counter = MaterialTokenEstimator(_Counter())
    return ImageFilterNode(
        _Filter(),
        ContextWindowSettings(backward_budget=20, forward_budget=20),
        budget=TokenBudget(counter=counter, token_limit=100_000),
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        batch_size=8,
        token_counters=(counter,),
    )


def test_filter_results_target_original_image_uuids_after_flat_stream_changes(
    tmp_path,
):
    keep_path = tmp_path / 'figure.png'
    remove_path = tmp_path / 'furniture.png'
    Image.new('RGB', (8, 8), 'red').save(keep_path)
    Image.new('RGB', (8, 8), 'white').save(remove_path)
    text = SourceBlock(
        uuid='text',
        block_type=BlockType.PARAGRAPH,
        content='heading',
    )
    figure = SourceBlock(
        uuid='figure',
        block_type=BlockType.IMAGE,
        content='substantive figure',
        image_path=str(keep_path),
    )
    furniture = SourceBlock(
        uuid='furniture',
        block_type=BlockType.IMAGE,
        content='furniture',
        image_path=str(remove_path),
    )
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source', key='source.pdf'),
        image_seam_blocks=[text, figure, furniture],
    )
    node = _node()

    sends = asyncio.run(node.dispatch(state))
    results = asyncio.run(node.worker(sends[0].arg))['image_filter_results']
    state = state.model_copy(
        update={
            'image_filter_results': results,
            'image_seam_blocks': [figure, furniture],
        }
    )
    blocks = node.collect(state)['image_filtered_blocks']

    assert [(result.block_uuid, result.keep) for result in results] == [
        ('figure', True),
        ('furniture', False),
    ]
    assert [block.uuid for block in blocks] == ['figure']
    assert blocks[0].image_path == str(keep_path)


def test_empty_image_stream_skips_model_dispatch():
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source', key='source.pdf'),
        image_seam_blocks=[
            SourceBlock(uuid='text', block_type='paragraph', content='text')
        ],
    )

    assert asyncio.run(_node().dispatch(state)) == 'image_filter_collect'
