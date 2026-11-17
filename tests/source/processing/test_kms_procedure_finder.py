from __future__ import annotations

import asyncio

from kms.config.inference import ContextWindowSettings
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.budget import ImagePayloadBudget
from kms.source.context import SourceBlockContext
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.processing.procedure_finder.models import (
    ProcedureBoundaryDecision,
    ProcedureStartDecision,
)
from kms.source.processing.procedure_finder.node import ProcedureFinderNode
from kms.source.processing.procedure_finder.settings import (
    ProcedureFinderSettings,
)
from kms.source.processing.state import SourceProcessingState


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _Start:
    def __init__(self, starts: set[str]) -> None:
        self.starts = starts

    async def acall(self, *, items):
        return [
            ProcedureStartDecision(
                position=item.position,
                is_procedure_start=item.target_block.material.text
                in self.starts,
            )
            for item in reversed(items)
        ]


class _Boundary:
    def __init__(self, boundaries: set[str]) -> None:
        self.boundaries = boundaries

    async def acall(self, *, start_block: SourceBlockContext, items):
        return [
            ProcedureBoundaryDecision(
                position=item.position,
                is_procedure_boundary=item.target_block.material.text
                in self.boundaries,
            )
            for item in reversed(items)
        ]


def _node(starts: set[str], boundaries: set[str]) -> ProcedureFinderNode:
    counter = MaterialTokenEstimator(_Counter())
    budget = TokenBudget(counter=counter, token_limit=100_000)
    settings = ProcedureFinderSettings(
        start_batch_size=2,
        boundary_batch_size=2,
        start_context_window=ContextWindowSettings(
            backward_budget=20, forward_budget=20
        ),
        boundary_context_window=ContextWindowSettings(
            backward_budget=20, forward_budget=20
        ),
    )
    return ProcedureFinderNode(
        _Start(starts),
        _Boundary(boundaries),
        settings,
        start_budget=budget,
        boundary_budget=budget,
        start_token_counters=(counter,),
        boundary_token_counters=(counter,),
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
    )


def test_procedure_finder_scans_after_exercises_and_groups_steps_until_boundary():
    blocks = [
        SourceBlock(
            uuid='intro',
            block_type='paragraph',
            content='intro',
        ),
        SourceBlock(
            uuid='task',
            block_type='paragraph',
            content='exercise task',
        ),
        SourceBlock(
            uuid='method',
            block_type='paragraph',
            content='method begins',
        ),
        SourceBlock(
            uuid='step',
            block_type='paragraph',
            content='calculation step',
        ),
        SourceBlock(
            uuid='proof',
            block_type='paragraph',
            content='proof conclusion',
        ),
        SourceBlock(
            uuid='tail',
            block_type='paragraph',
            content='unrelated exposition',
        ),
    ]
    exercise = SourceUnit(kind=SourceUnitKind.EXERCISE, blocks=[blocks[1]])
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source', key='source.pdf'),
        procedure_split_blocks=blocks,
        exercise_units=[exercise],
    )

    units = asyncio.run(
        _node({'method begins'}, {'unrelated exposition'}).run(state)
    )['procedure_units']

    assert len(units) == 1
    assert units[0].kind is SourceUnitKind.PROCEDURE
    assert [block.uuid for block in units[0].blocks] == [
        'method',
        'step',
        'proof',
    ]
    assert all(block.uuid != 'task' for unit in units for block in unit.blocks)
