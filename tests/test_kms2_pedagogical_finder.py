import asyncio

from kms2.config.source_processing import PedagogicalFinderSettings
from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.instruction import Instruction
from kms2.core.model.source_processing.pedagogical import ExerciseComponent
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.node.source_processing.pedagogical_finder import PedagogicalFinderNode


class StartRouter:
    def __init__(self, starts: set[str]) -> None:
        self.starts = starts
        self.calls = []

    async def aforward(self, *, target_block, **kwargs) -> bool:
        self.calls.append((target_block, kwargs))
        return target_block.content in self.starts


class BoundaryRouter:
    def __init__(self, boundaries: set[str]) -> None:
        self.boundaries = boundaries
        self.calls = []

    async def aforward(self, *, candidate_block, **kwargs) -> bool:
        self.calls.append((candidate_block, kwargs))
        return candidate_block.content in self.boundaries


def _state() -> SourceProcessingState:
    blocks = [
        SourceBlock(
            uuid=f'block-{index}',
            block_type=BlockType.PARAGRAPH,
            content=content,
        )
        for index, content in enumerate(
            [
                'instruction',
                'skip',
                'component one',
                'component detail',
                'component two',
                'tail',
            ]
        )
    ]
    return SourceProcessingState(
        pdf_path='source.pdf',
        source=Source(key='source-1'),
        split_pages=[SourcePage(index=0, blocks=blocks)],
        instructions=[Instruction(member_block_uuids=['block-0'])],
        exercise_components=[ExerciseComponent(member_block_uuids=['block-1'])],
    )


def test_pedagogical_finder_excludes_claimed_members_and_reconsiders_boundary():
    state = _state()
    original_pages = state.split_pages
    start_router = StartRouter({'component one', 'component two'})
    boundary_router = BoundaryRouter({'component two'})

    result = asyncio.run(
        PedagogicalFinderNode(
            start_router,
            boundary_router,
            PedagogicalFinderSettings(),
        ).run(state)
    )

    assert [
        component.member_block_uuids
        for component in result['pedagogical_components']
    ] == [['block-2', 'block-3'], ['block-4', 'block-5']]
    assert [block.content for block, _ in start_router.calls] == [
        'component one',
        'component two',
    ]
    assert [block.content for block, _ in boundary_router.calls] == [
        'component detail',
        'component two',
        'tail',
    ]
    assert state.split_pages is original_pages


def test_pedagogical_finder_emits_suffix_when_no_boundary_is_found():
    state = _state()
    original_pages = state.split_pages
    start_router = StartRouter({'component one'})
    boundary_router = BoundaryRouter(set())

    result = asyncio.run(
        PedagogicalFinderNode(
            start_router,
            boundary_router,
            PedagogicalFinderSettings(),
        ).run(state)
    )

    assert [
        component.member_block_uuids
        for component in result['pedagogical_components']
    ] == [['block-2', 'block-3', 'block-4', 'block-5']]
    assert state.split_pages is original_pages
