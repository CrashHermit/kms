from __future__ import annotations

import asyncio

from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.processing.correction.models import ContentCorrectionResult
from kms.source.processing.formatting.models import (
    FormattingOutput,
    FormattingResult,
)
from kms.source.processing.formatting.node import FormattingNode
from kms.source.processing.state import SourceProcessingState


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _Formatter:
    async def acall(self, *, blocks):
        return [
            FormattingOutput(
                position=item.position, formatted_content=item.content.upper()
            )
            for item in reversed(blocks)
        ]


def _node() -> FormattingNode:
    counter = MaterialTokenEstimator(_Counter())
    return FormattingNode(
        _Formatter(),
        budget=TokenBudget(counter=counter, token_limit=1000),
        batch_size=8,
    )


def _state() -> SourceProcessingState:
    first = SourceBlock(
        uuid='first',
        block_type=BlockType.PARAGRAPH,
        content='first',
    )
    later = SourceBlock(
        uuid='later',
        block_type=BlockType.PARAGRAPH,
        content='later',
    )
    return SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source', key='source.pdf'),
        corrected_blocks=[first, later],
        correction_results=[
            ContentCorrectionResult(
                source_block=first, keep=True, cleanup_instructions=[]
            ),
            ContentCorrectionResult(
                source_block=later, keep=True, cleanup_instructions=[]
            ),
        ],
    )


def test_formatting_worker_maps_packet_positions_back_to_block_uuids():
    node = _node()
    state = _state()
    sends = asyncio.run(node.dispatch(state))
    results = asyncio.run(node.worker(sends[0].arg))['formatting_results']

    assert [
        (result.block_uuid, result.formatted_content) for result in results
    ] == [
        ('first', 'FIRST'),
        ('later', 'LATER'),
    ]


def test_formatting_collection_keeps_flat_order_after_a_prior_removal():
    state = _state().model_copy(
        update={
            'corrected_blocks': [
                SourceBlock(
                    uuid='later', block_type='paragraph', content='later'
                ),
            ],
            'formatting_results': [
                FormattingResult(
                    block_uuid='later', formatted_content='formatted later'
                ),
                FormattingResult(
                    block_uuid='first', formatted_content='formatted first'
                ),
            ],
        }
    )

    blocks = _node().collect(state)['formatted_blocks']

    assert [(block.uuid, block.content) for block in blocks] == [
        ('later', 'formatted later'),
    ]
