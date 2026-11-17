"""Runtime demonstrations for procedure discovery."""

import dspy

from kms.media.models import Material
from kms.source.context import SourceBlockContext
from kms.source.models.source import BlockType
from kms.source.processing.procedure_finder.models import (
    ProcedureBoundaryDecision,
    ProcedureClassificationInput,
    ProcedureStartDecision,
)


def _item(content: str) -> ProcedureClassificationInput:
    return ProcedureClassificationInput(
        position=0,
        context_before=[],
        target_block=SourceBlockContext(
            block_type=BlockType.PARAGRAPH, material=Material.from_text(content)
        ),
        context_after=[],
    )


def procedure_start_examples() -> list[dspy.Example]:
    return [
        dspy.Example(
            items=[_item('Proof: assume the claim is false.')],
            decisions=[
                ProcedureStartDecision(position=0, is_procedure_start=True)
            ],
        ).with_inputs('items')
    ]


def procedure_boundary_examples() -> list[dspy.Example]:
    return [
        dspy.Example(
            start_block=_item('Proof: suppose x is positive.').target_block,
            items=[_item('Hence the claim follows.')],
            decisions=[
                ProcedureBoundaryDecision(
                    position=0, is_procedure_boundary=False
                )
            ],
        ).with_inputs('start_block', 'items')
    ]
