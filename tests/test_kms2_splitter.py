import asyncio
import uuid

import dspy

from kms2.core.model import (
    BlockType,
    Source,
    SourceBlock,
    SourceBlockContext,
    SourcePage,
    SplitCandidate,
    SplitDecision,
    SplitPiece,
    VisualAsset,
)
from kms2.langgraph.source.state import SourceState
from kms2.module.source import splitter as splitter_module
from kms2.node.source.splitter import SplitterNode


class _Predictor:
    def __init__(self, **values: object) -> None:
        self.values = values
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> dspy.Prediction:
        self.calls.append(kwargs)
        return dspy.Prediction(**self.values)

    async def acall(self, **kwargs: object) -> dspy.Prediction:
        self.calls.append(kwargs)
        return dspy.Prediction(**self.values)


def _context(content: str, block_type: BlockType = BlockType.TEXT):
    return SourceBlockContext(block_type=block_type, content=content)


def _projected_parent(content: str) -> SourceBlockContext:
    return _context(
        content,
        BlockType.LIST,
    ).model_copy(update={'asset_paths': ['packed-figure.png']})


def test_signatures_expose_routed_split_contracts():
    assert set(splitter_module.ExerciseStripRouterSignature.input_fields) == {
        'context_before',
        'target_block',
        'context_after',
    }
    assert set(splitter_module.ExerciseStripRouterSignature.output_fields) == {
        'contains_multiple_exercises'
    }
    assert set(splitter_module.ExerciseSplitterSignature.input_fields) == {
        'context_before',
        'candidates',
        'context_after',
    }
    assert set(splitter_module.ExerciseSplitterSignature.output_fields) == {
        'splits'
    }


def test_router_forwards_context_and_returns_model_boolean():
    router = splitter_module.ExerciseStripRouterModule(
        dspy.LM('openai/dummy', api_key='test')
    )
    predictor = _Predictor(contains_multiple_exercises=True)
    router.predictor = predictor
    before = [_context('before')]
    target = _context('1. first 2. second')
    after = [_context('after')]

    assert (
        router(
            context_before=before,
            target_block=target,
            context_after=after,
        )
        is True
    )
    assert (
        asyncio.run(
            router.aforward(
                context_before=before,
                target_block=target,
                context_after=after,
            )
        )
        is True
    )
    assert predictor.calls == [
        {
            'context_before': before,
            'target_block': target,
            'context_after': after,
        },
        {
            'context_before': before,
            'target_block': target,
            'context_after': after,
        },
    ]


def test_splitter_forwards_candidates_and_returns_model_decisions():
    splitter = splitter_module.ExerciseSplitterModule(
        dspy.LM('openai/dummy', api_key='test')
    )
    decision = SplitDecision(
        position=1,
        pieces=[SplitPiece(content='one'), SplitPiece(content='two')],
    )
    predictor = _Predictor(splits=[decision])
    splitter.predictor = predictor
    before = [_context('before')]
    candidates = [
        SplitCandidate(position=1, source_block=_context('packed')),
    ]
    after = [_context('after')]

    assert splitter(
        context_before=before,
        candidates=candidates,
        context_after=after,
    ) == [decision]
    assert asyncio.run(
        splitter.aforward(
            context_before=before,
            candidates=candidates,
            context_after=after,
        )
    ) == [decision]
    assert predictor.calls == [
        {
            'context_before': before,
            'candidates': candidates,
            'context_after': after,
        },
        {
            'context_before': before,
            'candidates': candidates,
            'context_after': after,
        },
    ]


class _Router:
    def __init__(self, positive_contents: set[str]) -> None:
        self.positive_contents = positive_contents
        self.calls: list[dict[str, object]] = []

    async def aforward(self, **kwargs: object) -> bool:
        self.calls.append(kwargs)
        target_block = kwargs['target_block']
        return target_block.content in self.positive_contents


class _Splitter:
    def __init__(self, decisions: list[SplitDecision]) -> None:
        self.decisions = decisions
        self.calls: list[dict[str, object]] = []

    async def aforward(self, **kwargs: object) -> list[SplitDecision]:
        self.calls.append(kwargs)
        return self.decisions


def _state(pages: list[SourcePage]) -> SourceState:
    return SourceState(
        pdf_path='document.pdf',
        source=Source(uuid='source-1', key='document.pdf'),
        image_seam_pages=pages,
    )


def test_node_routes_flat_context_and_restores_independent_page_blocks():
    before = SourceBlock(
        uuid='before',
        block_type=BlockType.TEXT,
        content='before',
    )
    parent = SourceBlock(
        uuid='packed',
        block_type=BlockType.LIST,
        content='1. first exercise 2. second exercise',
        crop_path='packed.png',
        crop_bbox=(1, 2, 3, 4),
        assets=[VisualAsset(path='packed-figure.png')],
    )
    after = SourceBlock(
        uuid='after',
        block_type=BlockType.TEXT,
        content='after',
    )
    pages = [
        SourcePage(index=0, blocks=[before, parent]),
        SourcePage(index=1, blocks=[after]),
    ]
    router = _Router({parent.content})
    splitter = _Splitter(
        [
            SplitDecision(
                position=0,
                pieces=[
                    SplitPiece(content='1. first exercise'),
                    SplitPiece(content='2. second exercise'),
                ],
            )
        ]
    )
    node = SplitterNode(
        router,
        splitter,
        backward_budget=10,
        target_budget=1,
        forward_budget=10,
    )
    result = asyncio.run(node.run(_state(pages)))
    split_pages = result['split_pages']
    child_blocks = split_pages[0].blocks[1:]

    assert router.calls[1] == {
        'context_before': [_context('before')],
        'target_block': _projected_parent(parent.content),
        'context_after': [_context('after')],
    }
    assert len(splitter.calls) == 1
    assert splitter.calls[0] == {
        'context_before': [_context('before')],
        'candidates': [
            SplitCandidate(
                position=0,
                source_block=_projected_parent(parent.content),
            )
        ],
        'context_after': [_context('after')],
    }
    assert [block.content for block in child_blocks] == [
        '1. first exercise',
        '2. second exercise',
    ]
    assert all(block.block_type is BlockType.LIST for block in child_blocks)
    assert all(block.uuid != parent.uuid for block in child_blocks)
    assert len({block.uuid for block in child_blocks}) == 2
    assert all(uuid.UUID(block.uuid).version == 4 for block in child_blocks)
    assert all(
        block.crop_path is None
        and block.crop_bbox is None
        and block.assets == []
        for block in child_blocks
    )
    assert split_pages[1] == pages[1]
    assert split_pages[1].blocks[0] is after


def test_node_preserves_pages_and_skips_splitter_without_candidates():
    pages = [
        SourcePage(
            index=0,
            blocks=[SourceBlock(block_type=BlockType.TEXT, content='ordinary')],
        )
    ]
    router = _Router(set())
    splitter = _Splitter([])
    node = SplitterNode(
        router,
        splitter,
        backward_budget=10,
        target_budget=1,
        forward_budget=10,
    )

    result = asyncio.run(node.run(_state(pages)))

    assert result['split_pages'] == pages
    assert result['split_pages'][0].blocks[0] is pages[0].blocks[0]
    assert splitter.calls == []
