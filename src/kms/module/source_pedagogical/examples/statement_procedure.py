"""Few-shot examples for independent component roles and local partitions."""

import dspy

from kms.core.model.block_types import BlockType
from kms.core.model.content import MultimodalContent
from kms.core.model.context import SourceBlockContext
from kms.core.model.source_pedagogical.pedagogical import (
    PedagogicalComponentInput,
    PedagogicalMember,
    PedagogicalRoleDecision,
    ProcedurePartitionDecision,
    StatementPartitionDecision,
)


def _component(position: int, *contents: str) -> PedagogicalComponentInput:
    return PedagogicalComponentInput(
        position=position,
        members=[
            PedagogicalMember(
                position=member_position,
                source_block=SourceBlockContext(
                    block_type=BlockType.PARAGRAPH,
                    material=MultimodalContent(text=content),
                ),
            )
            for member_position, content in enumerate(contents)
        ],
    )


def pedagogical_role_examples() -> list[dspy.Example]:
    """Distinguish posed notation, standalone working, mixed and neither roles."""
    return [
        dspy.Example(
            items=[
                _component(0, 'Exercise 4: $x^2 + 1$.'),
                _component(
                    1,
                    'Proof. Apply the distributive law.',
                    'Therefore the identity holds.',
                ),
                _component(
                    2,
                    'Exercise 5: Find the roots of $x^2 - 1$.',
                    'Factor $x^2 - 1 = (x - 1)(x + 1)$.',
                    'Thus the roots are $-1$ and $1$.',
                ),
                _component(3, 'Chapter 4 — Review'),
            ],
            decisions=[
                PedagogicalRoleDecision(
                    position=0, has_statement=True, has_procedure=False
                ),
                PedagogicalRoleDecision(
                    position=1, has_statement=False, has_procedure=True
                ),
                PedagogicalRoleDecision(
                    position=2, has_statement=True, has_procedure=True
                ),
                PedagogicalRoleDecision(
                    position=3, has_statement=False, has_procedure=False
                ),
            ],
            content_images=[],
        ).with_inputs('items', 'content_images')
    ]


def statement_partition_examples() -> list[dspy.Example]:
    """Select statement portions without borrowing another component's members."""
    return [
        dspy.Example(
            items=[
                _component(
                    0,
                    'Exercise 1.2.1: Sketch the slope field.',
                    r"(a) Use $y' = e^{x-y}$.",
                    r"(b) Use $y' = x^2$.",
                ),
                _component(
                    1,
                    r"Example 1.2.1: Attempt to solve $y' = 1/x$, $y(0) = 0$.",
                    r'Integrate to find the general solution $y = \ln |x| + C$.',
                ),
                _component(
                    2, 'Find $2 + 3$: counting on three from two gives $5$.'
                ),
            ],
            decisions=[
                StatementPartitionDecision(
                    position=0, statement_positions=[0, 1, 2]
                ),
                StatementPartitionDecision(position=1, statement_positions=[0]),
                StatementPartitionDecision(position=2, statement_positions=[0]),
            ],
            content_images=[],
        ).with_inputs('items', 'content_images')
    ]


def procedure_partition_examples() -> list[dspy.Example]:
    """Select visible working, allowing a combined member in both partitions."""
    return [
        dspy.Example(
            items=[
                _component(
                    0,
                    'Exercise 5: Find the roots of $x^2 - 1$.',
                    'Factor $x^2 - 1 = (x - 1)(x + 1)$.',
                    'Thus the roots are $-1$ and $1$.',
                ),
                _component(
                    1, 'Find $2 + 3$: counting on three from two gives $5$.'
                ),
                _component(
                    2,
                    'Proof. Apply the distributive law.',
                    'Therefore the identity holds.',
                ),
            ],
            decisions=[
                ProcedurePartitionDecision(
                    position=0, procedure_positions=[1, 2]
                ),
                ProcedurePartitionDecision(position=1, procedure_positions=[0]),
                ProcedurePartitionDecision(
                    position=2, procedure_positions=[0, 1]
                ),
            ],
            content_images=[],
        ).with_inputs('items', 'content_images')
    ]
