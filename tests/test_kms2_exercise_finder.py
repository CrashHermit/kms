import asyncio

from kms2.config.inference import ContextWindowSettings
from kms2.config.source_processing import ExerciseFinderSettings
from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.instruction import Instruction
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.node.source_processing.exercise_finder import ExerciseFinderNode


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


class _Counter:
    def __init__(self, counts: dict[str, int] | None = None) -> None:
        self.counts = counts or {}
        self.calls = 0

    def count_texts(self, texts: list[str]) -> list[int]:
        self.calls += 1
        return [self.counts.get(text, 1) for text in texts]


def _state() -> SourceProcessingState:
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
    return SourceProcessingState(
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
        start_token_counters=(_Counter(),),
        boundary_token_counters=(_Counter(),),
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
        start_token_counters=(_Counter(),),
        boundary_token_counters=(_Counter(),),
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


def test_distinct_profile_costs_select_start_and_boundary_contexts():
    class CapturingRouter:
        def __init__(self, answer: bool) -> None:
            self.answer = answer
            self.calls = []

        async def aforward(self, **kwargs):
            self.calls.append(kwargs)
            return self.answer

    class ProfileCounter:
        def __init__(self, costs: list[int]) -> None:
            self.costs = costs

        def count_texts(self, texts: list[str]) -> list[int]:
            return self.costs

    state = SourceProcessingState(
        pdf_path='source.pdf',
        source=Source(key='source-1'),
        split_pages=[
            SourcePage(
                index=0,
                blocks=[
                    SourceBlock(
                        uuid=f'candidate-{i}',
                        block_type=BlockType.PARAGRAPH,
                        content=text,
                    )
                    for i, text in enumerate(['start', 'boundary', 'tail'])
                ],
            )
        ],
    )
    start_router = CapturingRouter(True)
    boundary_router = CapturingRouter(True)
    node = ExerciseFinderNode(
        start_router,
        boundary_router,
        ExerciseFinderSettings(
            start_router=ExerciseFinderSettings().start_router.model_copy(
                update={'model_server_profile': 'start'}
            ),
            boundary_router=ExerciseFinderSettings().boundary_router.model_copy(
                update={'model_server_profile': 'boundary'}
            ),
            start_context_window=ContextWindowSettings(
                backward_budget=0,
                forward_budget=3,
            ),
            boundary_context_window=ContextWindowSettings(
                backward_budget=1,
                forward_budget=0,
            ),
        ),
        start_token_counters=(ProfileCounter([1, 3, 1]),),
        boundary_token_counters=(ProfileCounter([3, 1, 1]),),
    )

    asyncio.run(node.run(state))

    assert [
        block.content for block in start_router.calls[0]['context_after']
    ] == ['boundary']
    assert boundary_router.calls[0]['candidate_block'].content == 'boundary'
    assert boundary_router.calls[0]['context_before'] == []
