from __future__ import annotations

import asyncio

from kms.config.inference import ContextWindowSettings
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.budget import ImagePayloadBudget
from kms.source.context import SourceBlockContext
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.models.unit import SourceUnitKind
from kms.source.processing.exercise_finder.models import (
    ExerciseBoundaryDecision,
    ExerciseStartDecision,
)
from kms.source.processing.exercise_finder.node import ExerciseFinderNode
from kms.source.processing.exercise_finder.settings import (
    ExerciseFinderSettings,
)
from kms.source.processing.instruction_redistribution.models import (
    ExerciseSupportGroup,
)
from kms.source.processing.state import SourceProcessingState


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _StartRouter:
    def __init__(self, starts: set[str]) -> None:
        self.starts = starts

    async def acall(self, *, items):
        return [
            ExerciseStartDecision(
                position=item.position,
                is_exercise_start=item.target_block.material.text
                in self.starts,
            )
            for item in reversed(items)
        ]


class _BoundaryRouter:
    def __init__(self, boundaries: set[str]) -> None:
        self.boundaries = boundaries

    async def acall(self, *, start_block: SourceBlockContext, items):
        return [
            ExerciseBoundaryDecision(
                position=item.position,
                is_exercise_boundary=item.target_block.material.text
                in self.boundaries,
            )
            for item in reversed(items)
        ]


def _node(starts: set[str], boundaries: set[str]) -> ExerciseFinderNode:
    counter = MaterialTokenEstimator(_Counter())
    budget = TokenBudget(counter=counter, token_limit=100_000)
    return ExerciseFinderNode(
        _StartRouter(starts),
        _BoundaryRouter(boundaries),
        ExerciseFinderSettings(
            start_batch_size=2,
            boundary_batch_size=2,
            start_context_window=ContextWindowSettings(
                backward_budget=20, forward_budget=20
            ),
            boundary_context_window=ContextWindowSettings(
                backward_budget=20, forward_budget=20
            ),
        ),
        start_budget=budget,
        boundary_budget=budget,
        start_token_counters=(counter,),
        boundary_token_counters=(counter,),
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
    )


def _state(*contents: str) -> SourceProcessingState:
    blocks = [
        SourceBlock(
            uuid=f'b{index}',
            block_type=BlockType.PARAGRAPH,
            content=content,
        )
        for index, content in enumerate(contents)
    ]
    return SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source', key='source.pdf'),
        procedure_split_blocks=blocks,
    )


def test_finder_keeps_setup_and_subparts_together_and_stops_before_solution():
    state = _state(
        'section setup',
        'shared setup for the task',
        'Use the displayed graph to answer the following.',
        '(a) find x',
        '(b) find y',
        'worked solution and proof',
        'independent task',
    )
    node = _node({'shared setup for the task'}, {'worked solution and proof'})

    units = asyncio.run(node.run(state))['exercise_units']

    assert [unit.kind for unit in units] == [SourceUnitKind.EXERCISE]
    assert [block.uuid for block in units[0].blocks] == ['b1', 'b2', 'b3', 'b4']
    assert [block.content for block in units[0].blocks[2:]] == [
        '(a) find x',
        '(b) find y',
    ]


def test_unlabeled_following_task_ends_previous_exercise_exclusively():
    state = _state(
        'Exercise 1: calculate',
        'equation support',
        'Describe a method',
        'solution',
    )
    node = _node(
        {'Exercise 1: calculate', 'Describe a method'},
        {'Describe a method', 'solution'},
    )

    units = asyncio.run(node.run(state))['exercise_units']

    assert [[block.uuid for block in unit.blocks] for unit in units] == [
        ['b0', 'b1'],
        ['b2'],
    ]
    assert all(unit.kind is SourceUnitKind.EXERCISE for unit in units)


def test_default_windows_include_shared_instruction_for_each_task():
    instruction = 'Rewrite each statement as a biconditional.'
    state = _state(instruction, '1. A implies B.', '2. C implies D.')
    state.exercise_support_groups = [ExerciseSupportGroup(block_uuids=['b0'])]
    seen_boundaries = []

    class StartRouter:
        async def acall(self, *, items):
            return [
                ExerciseStartDecision(
                    position=item.position,
                    is_exercise_start=(
                        item.target_block.material.text.startswith(('1.', '2.'))
                        and any(
                            block.material.text == instruction
                            for block in item.context_before
                        )
                    ),
                )
                for item in items
            ]

    class BoundaryRouter:
        async def acall(self, *, start_block, items):
            seen_boundaries.extend(items)
            return [
                ExerciseBoundaryDecision(
                    position=item.position,
                    is_exercise_boundary=item.target_block.material.text.startswith(
                        '2.'
                    ),
                )
                for item in items
            ]

    counter = MaterialTokenEstimator(_Counter())
    budget = TokenBudget(counter=counter, token_limit=100_000)
    node = ExerciseFinderNode(
        StartRouter(),
        BoundaryRouter(),
        ExerciseFinderSettings(),
        start_budget=budget,
        boundary_budget=budget,
        start_token_counters=(counter,),
        boundary_token_counters=(counter,),
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
    )

    units = asyncio.run(node.run(state))['exercise_units']

    assert [[block.uuid for block in unit.blocks] for unit in units] == [
        ['b1'],
        ['b2'],
    ]
    assert any(
        block.material.text == instruction
        for block in seen_boundaries[0].context_before
    )


def test_previously_classified_starts_survive_later_boundaries():
    state = _state('Task 1', 'Task 2', 'Task 3', 'Task 4')

    class ChangingStartRouter:
        def __init__(self):
            self.seen = set()

        async def acall(self, *, items):
            decisions = []
            for item in items:
                text = item.target_block.material.text
                decisions.append(
                    ExerciseStartDecision(
                        position=item.position,
                        is_exercise_start=text not in self.seen,
                    )
                )
                self.seen.add(text)
            return decisions

    node = _node(set(), {'Task 2', 'Task 3', 'Task 4'})
    node._settings.start_batch_size = 4
    node._start_router = ChangingStartRouter()

    units = asyncio.run(node.run(state))['exercise_units']

    assert [[block.content for block in unit.blocks] for unit in units] == [
        ['Task 1'],
        ['Task 2'],
        ['Task 3'],
        ['Task 4'],
    ]
