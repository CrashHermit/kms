from __future__ import annotations

import asyncio

from PIL import Image

from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.budget import ImagePayloadBudget
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.processing.instruction_redistribution.models import (
    ExerciseSupportGroup,
    InstructionApplicabilityDecision,
    InstructionBoundaryDecision,
    InstructionStartDecision,
)
from kms.source.processing.instruction_redistribution.node import (
    InstructionRedistributionNode,
)
from kms.source.processing.instruction_redistribution.settings import (
    InstructionRedistributionSettings,
)
from kms.source.processing.state import SourceProcessingState


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _NoDetection:
    async def acall(self, **kwargs):
        return []


class _StartDetection:
    async def acall(self, *, items):
        return [
            InstructionStartDecision(
                position=item.position,
                is_instruction_start=item.target_block.material.text
                == 'For Exercises 1 and 2, use the diagram.',
            )
            for item in reversed(items)
        ]


class _BoundaryDetection:
    async def acall(self, *, start_block, items):
        return [
            InstructionBoundaryDecision(
                position=item.position,
                is_instruction_boundary=item.target_block.material.text
                == 'Exercise one',
            )
            for item in reversed(items)
        ]


class _Applicability:
    async def acall(self, *, instruction_blocks, items):
        return [
            InstructionApplicabilityDecision(
                position=item.position,
                applies=any(
                    'needing shared diagram' in block.material.text
                    for block in item.exercise_blocks
                ),
            )
            for item in reversed(items)
        ]


def _node(start=None, boundary=None) -> InstructionRedistributionNode:
    counter = MaterialTokenEstimator(_Counter())
    budget = TokenBudget(counter=counter, token_limit=100_000)
    image_budget = ImagePayloadBudget(8, 26_214_400, 40_000_000)
    return InstructionRedistributionNode(
        start or _NoDetection(),
        boundary or _NoDetection(),
        _Applicability(),
        InstructionRedistributionSettings(),
        start_budget=budget,
        boundary_budget=budget,
        applicability_budget=budget,
        start_token_counters=(counter,),
        boundary_token_counters=(counter,),
        applicability_token_counters=(counter,),
        image_budget=image_budget,
        applicability_image_budget=image_budget,
    )


def test_detection_reserves_directive_and_visual_support_as_one_group(tmp_path):
    image_path = tmp_path / 'diagram.png'
    Image.new('RGB', (8, 8), 'blue').save(image_path)
    directive = SourceBlock(
        uuid='directive',
        block_type=BlockType.PARAGRAPH,
        content='For Exercises 1 and 2, use the diagram.',
    )
    diagram = SourceBlock(
        uuid='diagram',
        block_type=BlockType.IMAGE,
        content=None,
        image_path=str(image_path),
    )
    exercise = SourceBlock(
        uuid='exercise',
        block_type=BlockType.PARAGRAPH,
        content='Exercise one',
    )
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source', key='source.pdf'),
        procedure_split_blocks=[directive, diagram, exercise],
    )

    groups = asyncio.run(
        _node(_StartDetection(), _BoundaryDetection()).detect(state)
    )['exercise_support_groups']

    assert [group.block_uuids for group in groups] == [['directive', 'diagram']]


def test_shared_support_is_copied_only_to_applicable_exercise_and_original_removed(
    tmp_path,
):
    diagram = tmp_path / 'shared.png'
    Image.new('RGB', (8, 8), 'blue').save(diagram)
    directive = SourceBlock(
        uuid='directive',
        block_type=BlockType.PARAGRAPH,
        content='For Exercises 1 and 2, use the diagram.',
    )
    image = SourceBlock(
        uuid='diagram',
        block_type=BlockType.IMAGE,
        content=None,
        image_path=str(diagram),
        image_sha256='sha256',
    )
    first_task = SourceBlock(
        uuid='exercise-1',
        block_type=BlockType.PARAGRAPH,
        content='Exercise needing shared diagram',
    )
    second_task = SourceBlock(
        uuid='exercise-2',
        block_type=BlockType.PARAGRAPH,
        content='Exercise two also needing shared diagram',
    )
    support_unit = SourceUnit(kind=SourceUnitKind.CONTENT, blocks=[directive])
    image_unit = SourceUnit(kind=SourceUnitKind.CONTENT, blocks=[image])
    first = SourceUnit(kind=SourceUnitKind.EXERCISE, blocks=[first_task])
    second = SourceUnit(kind=SourceUnitKind.EXERCISE, blocks=[second_task])
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source', key='source.pdf'),
        procedure_split_blocks=[directive, image, first_task, second_task],
        exercise_support_groups=[
            ExerciseSupportGroup(block_uuids=['directive', 'diagram'])
        ],
        units=[support_unit, image_unit, first, second],
    )

    units = asyncio.run(_node().run(state))['units']

    assert [unit.kind for unit in units] == [
        SourceUnitKind.EXERCISE,
        SourceUnitKind.EXERCISE,
    ]
    copied_first_text, copied_first_image = units[0].blocks[:2]
    copied_second_text, copied_second_image = units[1].blocks[:2]
    assert [block.uuid for block in units[0].blocks[2:]] == ['exercise-1']
    assert [block.uuid for block in units[1].blocks[2:]] == ['exercise-2']
    assert copied_first_text.uuid != directive.uuid
    assert copied_second_text.uuid != directive.uuid
    assert copied_first_text.uuid != copied_second_text.uuid
    assert copied_first_text.content == directive.content
    assert copied_first_text.block_type is directive.block_type
    assert copied_first_image.uuid != image.uuid
    assert copied_second_image.uuid != image.uuid
    assert copied_first_image.uuid != copied_second_image.uuid
    assert (
        copied_first_image.image_path
        == copied_second_image.image_path
        == image.image_path
    )
    assert (
        copied_first_image.image_sha256
        == copied_second_image.image_sha256
        == image.image_sha256
    )
    assert copied_first_image.embedding is None
    assert copied_second_image.embedding is None
    assert all(
        block.uuid not in {'directive', 'diagram'}
        for unit in units
        for block in unit.blocks
    )


def test_unassigned_shared_support_remains_a_content_unit():
    directive = SourceBlock(
        uuid='directive',
        block_type=BlockType.PARAGRAPH,
        content='Use the supplied method.',
    )
    task = SourceBlock(
        uuid='task',
        block_type=BlockType.PARAGRAPH,
        content='A separate exercise.',
    )
    content = SourceUnit(kind=SourceUnitKind.CONTENT, blocks=[directive])
    exercise = SourceUnit(kind=SourceUnitKind.EXERCISE, blocks=[task])
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source', key='source.pdf'),
        procedure_split_blocks=[directive, task],
        exercise_support_groups=[
            ExerciseSupportGroup(block_uuids=['directive'])
        ],
        units=[content, exercise],
    )

    units = asyncio.run(_node().run(state))['units']

    assert units == [content, exercise]
