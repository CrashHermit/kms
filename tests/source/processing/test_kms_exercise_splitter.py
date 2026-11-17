from __future__ import annotations

import asyncio

from kms.config.inference import ContextWindowSettings
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.budget import ImagePayloadBudget
from kms.media.models import Material
from kms.source.context import (
    PositionedSourceBlock,
    SourceBlockContext,
    SourceBlockWindow,
)
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.processing.exercise_splitter.models import (
    SplitCandidate,
    SplitDecision,
    SplitPiece,
    SplitRequest,
    SplitResult,
    SplitRouteDecision,
)
from kms.source.processing.exercise_splitter.node import ExerciseSplitterNode
from kms.source.processing.state import SourceProcessingState


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _Router:
    async def acall(
        self,
        *,
        context_before: list[SourceBlockContext],
        targets: list[PositionedSourceBlock],
        context_after: list[SourceBlockContext],
    ) -> list[SplitRouteDecision]:
        return [
            SplitRouteDecision(position=target.position, requires_split=True)
            for target in reversed(targets)
            if target.source_block.material.text == 'prefix task answer tail'
        ]


class _Splitter:
    def __init__(self) -> None:
        self.calls: list[list[SplitCandidate]] = []

    async def acall(
        self,
        *,
        context_before: list[SourceBlockContext],
        candidates: list[SplitCandidate],
        context_after: list[SourceBlockContext],
        reference_blocks: list[PositionedSourceBlock],
    ) -> list[SplitDecision]:
        self.calls.append(candidates)
        return [
            SplitDecision(
                position=candidate.position,
                pieces=[
                    SplitPiece(content='prefix '),
                    SplitPiece(content='task '),
                    SplitPiece(content='answer '),
                    SplitPiece(content='tail'),
                ],
            )
            for candidate in candidates
        ]


def _block(
    uuid: str,
    content: str,
    *,
    block_type: BlockType = BlockType.PARAGRAPH,
    embedding: list[float] | None = None,
) -> SourceBlock:
    return SourceBlock(
        uuid=uuid,
        block_type=block_type,
        content=content,
        embedding=embedding,
        crop_path=f'{uuid}.png',
        crop_bbox=(1, 2, 3, 4),
    )


def _node(
    splitter: _Splitter, *, target_budget: int = 0
) -> ExerciseSplitterNode:
    counter = MaterialTokenEstimator(_Counter())
    budget = TokenBudget(counter=counter, token_limit=100_000)
    return ExerciseSplitterNode(
        _Router(),
        splitter,
        ContextWindowSettings(
            backward_budget=100,
            forward_budget=100,
            target_budget=target_budget,
        ),
        batch_size=1 if target_budget else 8,
        token_counters=(counter,),
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        router_budget=budget,
        splitter_budget=budget,
    )


def _state(blocks: list[SourceBlock]) -> SourceProcessingState:
    return SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source-1', key='source.pdf'),
        image_filtered_blocks=blocks,
    )


def test_dispatch_advances_across_flat_target_windows():
    blocks = [_block(str(index), f'block-{index}') for index in range(4)]
    node = _node(_Splitter(), target_budget=8)

    sends = asyncio.run(node.dispatch(_state(blocks)))

    assert [send.arg['split_request'].flat_position for send in sends] == [
        0,
        1,
        2,
        3,
    ]
    assert [
        [block.content for block in send.arg['split_request'].window.target]
        for send in sends
    ] == [[block.content] for block in blocks]


def test_worker_maps_target_local_decisions_to_absolute_stream_positions():
    splitter = _Splitter()
    node = _node(splitter)
    block = _block('target', 'prefix task answer tail')
    context = SourceBlockContext(
        block_type=BlockType.PARAGRAPH,
        material=Material.from_text(block.content or ''),
    )

    result = asyncio.run(
        node.worker(
            {
                'split_request': SplitRequest(
                    flat_position=17,
                    window=SourceBlockWindow(
                        target=[block],
                        context_before=[_block('prior', 'prior')],
                    ),
                )
            }
        )
    )

    assert [
        item.flat_position for item in result['exercise_split_results']
    ] == [17]
    assert [
        piece.content for piece in result['exercise_split_results'][0].pieces
    ] == [
        'prefix ',
        'task ',
        'answer ',
        'tail',
    ]
    assert context.material.text == block.content


def test_collector_conserves_fragments_and_preserves_unselected_blocks():
    untouched = _block(
        'keep', 'unselected', block_type=BlockType.LIST, embedding=[1.0]
    )
    target = _block(
        'target', 'prefix task answer tail', block_type=BlockType.LIST
    )
    state = _state([untouched, target]).model_copy(
        update={
            'exercise_split_results': [
                SplitResult(
                    flat_position=1,
                    pieces=[
                        SplitPiece(content='prefix '),
                        SplitPiece(content='task '),
                        SplitPiece(content='answer '),
                        SplitPiece(content='tail'),
                    ],
                )
            ]
        }
    )

    blocks = ExerciseSplitterNode.collect(_node(_Splitter()), state)[
        'exercise_split_blocks'
    ]

    fragments = blocks[1:]
    assert blocks[0] is untouched
    assert ''.join(block.content or '' for block in fragments) == target.content
    assert [block.block_type for block in fragments] == [BlockType.LIST] * 4
    assert all(block.crop_path == target.crop_path for block in fragments)
    assert all(block.crop_bbox == target.crop_bbox for block in fragments)
    assert all(block.embedding is None for block in fragments)
    assert len({block.uuid for block in fragments}) == 4


def test_no_split_result_passes_original_block_identity_through():
    original = _block('stable', 'no boundary found', block_type=BlockType.TABLE)
    state = _state([original])

    blocks = _node(_Splitter()).collect(state)['exercise_split_blocks']

    assert blocks == [original]
    assert blocks[0] is original
