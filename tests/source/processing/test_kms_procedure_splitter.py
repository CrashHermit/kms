from __future__ import annotations

import asyncio

from kms.config.inference import ContextWindowSettings
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.budget import ImagePayloadBudget
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.processing.procedure_splitter.models import (
    ProcedureSplitDecision,
    ProcedureSplitPiece,
    ProcedureSplitRouteDecision,
)
from kms.source.processing.procedure_splitter.node import ProcedureSplitterNode
from kms.source.processing.state import SourceProcessingState


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _Router:
    async def acall(self, *, context_before, targets, context_after):
        return [
            ProcedureSplitRouteDecision(
                position=target.position, requires_split=True
            )
            for target in targets
            if target.source_block.material.text
            == 'proof and unrelated paragraph'
        ]


class _Splitter:
    async def acall(
        self, *, context_before, candidates, reference_blocks, context_after
    ):
        return [
            ProcedureSplitDecision(
                position=candidate.position,
                pieces=[
                    ProcedureSplitPiece(content='proof '),
                    ProcedureSplitPiece(content='and unrelated paragraph'),
                ],
            )
            for candidate in candidates
        ]


def _node() -> ProcedureSplitterNode:
    counter = MaterialTokenEstimator(_Counter())
    budget = TokenBudget(counter=counter, token_limit=100_000)
    return ProcedureSplitterNode(
        _Router(),
        _Splitter(),
        ContextWindowSettings(
            backward_budget=10, forward_budget=10, target_budget=0
        ),
        batch_size=8,
        token_counters=(counter,),
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        router_budget=budget,
        splitter_budget=budget,
    )


def test_procedure_split_conserves_text_and_source_structure():
    original = SourceBlock(
        uuid='mixed',
        block_type=BlockType.CODE,
        content='proof and unrelated paragraph',
        crop_path='original-crop.png',
        crop_bbox=(2, 4, 8, 10),
        embedding=[7.0],
    )
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source', key='source.pdf'),
        exercise_split_blocks=[original],
    )
    node = _node()
    sends = asyncio.run(node.dispatch(state))
    result = asyncio.run(node.worker(sends[0].arg))
    state = state.model_copy(
        update={'procedure_split_results': result['procedure_split_results']}
    )

    blocks = node.collect(state)['procedure_split_blocks']

    assert [block.content for block in blocks] == [
        'proof ',
        'and unrelated paragraph',
    ]
    assert ''.join(block.content or '' for block in blocks) == original.content
    assert [block.block_type for block in blocks] == [
        BlockType.CODE,
        BlockType.CODE,
    ]
    assert all(block.crop_path == original.crop_path for block in blocks)
    assert all(block.crop_bbox == original.crop_bbox for block in blocks)
    assert all(block.embedding is None for block in blocks)
    assert len({block.uuid for block in blocks}) == 2
