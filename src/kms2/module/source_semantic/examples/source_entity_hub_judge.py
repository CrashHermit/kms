"""Runtime examples for source-local entity identity judgments."""

import dspy

from kms2.core.model.block_types import BlockType
from kms2.core.model.context import SourceBlockContext
from kms2.core.model.source_extraction import SourceEvidenceInput
from kms2.core.model.source_semantic.source_entity_hub import (
    SourceEntityHubJudgeDecision,
    SourceEntityHubJudgeInput,
)


def _evidence(
    target: str,
    context_before: str,
    context_after: str,
) -> SourceEvidenceInput:
    return SourceEvidenceInput(
        context_before=[
            SourceBlockContext(
                block_type=BlockType.PARAGRAPH, content=context_before
            )
        ],
        target_blocks=[
            SourceBlockContext(block_type=BlockType.PARAGRAPH, content=target)
        ],
        context_after=[
            SourceBlockContext(
                block_type=BlockType.PARAGRAPH, content=context_after
            )
        ],
    )


def source_entity_hub_judge_examples() -> list[dspy.Example]:
    """Keep repeated exercise-local symbols distinct from one identity."""
    return [
        dspy.Example(
            requests=[
                SourceEntityHubJudgeInput(
                    index=0,
                    left_name='f',
                    left_description='the function mapping x to x squared',
                    left_evidence=_evidence(
                        'Let f(x) = x^2.',
                        'Exercise 1 defines a function.',
                        'Evaluate f(3).',
                    ),
                    right_name='f',
                    right_description='the function mapping x to x squared',
                    right_evidence=_evidence(
                        'Evaluate f(3).',
                        'Exercise 1 defines f(x) = x^2.',
                        'Find the derivative of f.',
                    ),
                ),
                SourceEntityHubJudgeInput(
                    index=1,
                    left_name='f',
                    left_description='the function mapping x to x squared',
                    left_evidence=_evidence(
                        'Let f(x) = x^2.',
                        'Exercise 1 defines a function.',
                        'Evaluate f(3).',
                    ),
                    right_name='f',
                    right_description='the force on the object',
                    right_evidence=_evidence(
                        'The force is denoted by f.',
                        'Exercise 2 studies the object motion.',
                        'Determine its acceleration.',
                    ),
                ),
                SourceEntityHubJudgeInput(
                    index=2,
                    left_name='f',
                    left_description='a real-valued function of a real variable',
                    left_evidence=_evidence(
                        'Let f(x) = x^2.',
                        'Exercise 1.',
                        'Evaluate f(3).',
                    ),
                    right_name='f',
                    right_description='a real-valued function of a real variable',
                    right_evidence=_evidence(
                        'Let f(x) = sin(x).',
                        'Exercise 2.',
                        'Find the derivative of f.',
                    ),
                ),
                SourceEntityHubJudgeInput(
                    index=3,
                    left_name='derivative',
                    left_description='the instantaneous rate of change of a function',
                    left_evidence=_evidence(
                        'Find the derivative of f.',
                        'Exercise 2 defines f(x) = sin(x).',
                        'Interpret the instantaneous rate of change.',
                    ),
                    right_name='derivative',
                    right_description='the instantaneous rate of change of a function',
                    right_evidence=_evidence(
                        'Compute the derivative of g.',
                        'Exercise 3 defines g(x) = x^3.',
                        'Interpret its instantaneous rate of change.',
                    ),
                ),
            ],
            decisions=[
                SourceEntityHubJudgeDecision(index=0, belongs_in_same_hub=True),
                SourceEntityHubJudgeDecision(
                    index=1, belongs_in_same_hub=False
                ),
                SourceEntityHubJudgeDecision(
                    index=2, belongs_in_same_hub=False
                ),
                SourceEntityHubJudgeDecision(index=3, belongs_in_same_hub=True),
            ],
        ).with_inputs('requests')
    ]
