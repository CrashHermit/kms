import asyncio

from kms2.config import ExerciseFinderSettings
from kms2.core.model import (
    BlockType,
    Instruction,
    Source,
    SourceBlock,
    SourcePage,
)
from kms2.langgraph.source.state import SourceState
from kms2.node.source.exercise_finder import ExerciseFinderNode


class StartRouter:
    def __init__(self, starts: set[str]) -> None:
        self.starts = starts
        self.calls: list[str | None] = []

    async def aforward(self, *, target_block, **_kwargs) -> bool:
        self.calls.append(target_block.content)
        return target_block.content in self.starts


class BoundaryRouter:
    def __init__(self, boundaries: set[str]) -> None:
        self.boundaries = boundaries
        self.calls: list[str] = []

    async def aforward(self, *, candidate_block, **_kwargs) -> bool:
        self.calls.append(candidate_block.content)
        return candidate_block.content in self.boundaries


def _state() -> SourceState:
    blocks = [
        SourceBlock(
            uuid=f'block-{index}',
            block_type=block_type,
            content=content,
        )
        for index, (block_type, content) in enumerate(
            [
                (BlockType.INSTRUCTION, 'shared instruction'),
                (BlockType.PARAGRAPH, 'instruction detail'),
                (BlockType.PARAGRAPH, 'exercise one'),
                (BlockType.IMAGE, 'exercise image'),
                (BlockType.TABLE, 'exercise table'),
                (BlockType.LIST, 'exercise list'),
                (BlockType.PARAGRAPH, 'exercise two'),
                (BlockType.PARAGRAPH, 'exercise two detail'),
            ]
        )
    ]
    return SourceState(
        pdf_path='source.pdf',
        source=Source(key='source-1'),
        split_pages=[SourcePage(index=0, blocks=blocks)],
        instructions=[Instruction(member_block_uuids=['block-0', 'block-1'])],
    )


def test_exercise_finder_reconsiders_exclusive_boundary_candidate():
    start_router = StartRouter({'exercise one', 'exercise two'})
    boundary_router = BoundaryRouter({'exercise two'})
    node = ExerciseFinderNode(
        start_router,
        boundary_router,
        ExerciseFinderSettings(),
    )

    result = asyncio.run(node.run(_state()))

    assert [
        component.member_block_uuids
        for component in result['exercise_components']
    ] == [
        ['block-2', 'block-3', 'block-4', 'block-5'],
        ['block-6', 'block-7'],
    ]
    assert start_router.calls == ['exercise one', 'exercise two']
    assert boundary_router.calls == [
        'exercise image',
        'exercise table',
        'exercise list',
        'exercise two',
        'exercise two detail',
    ]


def test_exercise_finder_emits_suffix_when_no_boundary_is_found():
    start_router = StartRouter({'exercise one'})
    boundary_router = BoundaryRouter(set())
    node = ExerciseFinderNode(
        start_router,
        boundary_router,
        ExerciseFinderSettings(),
    )

    result = asyncio.run(node.run(_state()))

    assert [
        component.member_block_uuids
        for component in result['exercise_components']
    ] == [
        [
            'block-2',
            'block-3',
            'block-4',
            'block-5',
            'block-6',
            'block-7',
        ]
    ]
