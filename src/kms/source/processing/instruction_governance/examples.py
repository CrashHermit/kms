"""Few-shot examples for independently contextualized governance packets."""

import dspy

from kms.media.models import Material
from kms.source.context import SourceBlockContext
from kms.source.models.source import BlockType
from kms.source.processing.instruction_governance.models import (
    InstructionGovernanceDecision,
    InstructionGovernanceInput,
)


def _block(content: str) -> SourceBlockContext:
    return SourceBlockContext(
        block_type=BlockType.PARAGRAPH, material=Material.from_text(content)
    )


def instruction_governance_examples() -> list[dspy.Example]:
    """Return fresh contrasting candidates under shared directive anchors."""
    return [
        dspy.Example(
            instruction_blocks=[
                _block('For the following exercises, find the gradient.')
            ],
            items=[
                InstructionGovernanceInput(
                    position=0,
                    context_before=[],
                    statement_blocks=[
                        _block('280. Find the gradient of'),
                        _block(r'$$f(x, y) = \frac{14 - x^2 - y^2}{3}.$$'),
                        _block('Then, find the gradient at point $P(1, 2)$.'),
                    ],
                    context_after=[],
                ),
                InstructionGovernanceInput(
                    position=1,
                    context_before=[],
                    statement_blocks=[
                        _block(
                            r'282. Find the gradient of $f(x, y, z)$ at $P$ and '
                            r'the directional derivative in the direction of $\mathbf{u}$.'
                        )
                    ],
                    context_after=[],
                ),
                InstructionGovernanceInput(
                    position=2,
                    context_before=[
                        _block(
                            'Directional derivatives: for the following exercises use the direction from $P$ to $Q$.'
                        )
                    ],
                    statement_blocks=[
                        _block('284. $f(x, y) = x^2 + 3y^2, P(1, 1), Q(4, 5)$')
                    ],
                    context_after=[],
                ),
            ],
            decisions=[
                InstructionGovernanceDecision(position=0, governs=True),
                InstructionGovernanceDecision(position=1, governs=True),
                InstructionGovernanceDecision(position=2, governs=False),
            ],
        ).with_inputs('instruction_blocks', 'items'),
        dspy.Example(
            instruction_blocks=[
                _block(
                    'For the following exercises, find the directional derivative of the function at point $P$ in the direction of $Q$.'
                )
            ],
            items=[
                InstructionGovernanceInput(
                    position=0,
                    context_before=[],
                    statement_blocks=[
                        _block('284. $f(x, y) = x^2 + 3y^2, P(1, 1), Q(4, 5)$')
                    ],
                    context_after=[],
                )
            ],
            decisions=[InstructionGovernanceDecision(position=0, governs=True)],
        ).with_inputs('instruction_blocks', 'items'),
        dspy.Example(
            instruction_blocks=[
                _block(
                    r'For the following exercises, find the derivative of the function at $P$ in the direction of $\mathbf{u}$.'
                )
            ],
            items=[
                InstructionGovernanceInput(
                    position=0,
                    context_before=[],
                    statement_blocks=[
                        _block(
                            r'286. $f(x, y) = -7x + 2y, P(2, -4), \mathbf{u} = 4\mathbf{i} - 3\mathbf{j}$'
                        )
                    ],
                    context_after=[],
                ),
                InstructionGovernanceInput(
                    position=1,
                    context_before=[_block('Maximum rate of change')],
                    statement_blocks=[
                        _block(
                            '294. $f(x, y) = x^2 + xy + y^2$ at point $(-5, -4)$ in the direction the function increases most rapidly'
                        )
                    ],
                    context_after=[],
                ),
                InstructionGovernanceInput(
                    position=2,
                    context_before=[
                        _block('Level curves and gradient vectors')
                    ],
                    statement_blocks=[
                        _block(
                            '288. [T] Use technology to sketch the level curve of $f(x, y) = 4x - 2y + 3$ that passes through $P(1, 2)$ and draw the gradient vector at $P$.'
                        )
                    ],
                    context_after=[],
                ),
                InstructionGovernanceInput(
                    position=3,
                    context_before=[
                        _block('Level curves and gradient vectors')
                    ],
                    statement_blocks=[
                        _block(
                            '289. [T] Use technology to sketch the level curve of $f(x, y) = x^2 + 4y^2$ that passes through $P(-2, 0)$ and draw the gradient vector at $P$.'
                        )
                    ],
                    context_after=[],
                ),
            ],
            decisions=[
                InstructionGovernanceDecision(
                    position=position, governs=position == 0
                )
                for position in range(4)
            ],
        ).with_inputs('instruction_blocks', 'items'),
        dspy.Example(
            instruction_blocks=[
                _block(
                    'For the following exercises, find the maximum rate of change of $f$ at the given point and the direction in which it occurs.'
                )
            ],
            items=[
                InstructionGovernanceInput(
                    position=0,
                    context_before=[],
                    statement_blocks=[
                        _block('299. $f(x, y) = x^2 + y^2$ at point $(1, 2)$')
                    ],
                    context_after=[],
                )
            ],
            decisions=[InstructionGovernanceDecision(position=0, governs=True)],
        ).with_inputs('instruction_blocks', 'items'),
        dspy.Example(
            instruction_blocks=[
                _block('For the following exercises, find equations of')
            ],
            items=[
                InstructionGovernanceInput(
                    position=0,
                    context_before=[],
                    statement_blocks=[
                        _block(
                            '302. The level surface $$f(x, y, z) = 12$$ for $$f(x, y, z) = 4x^2 - 2y^2 + z^2$$ at point $(2, 2, 2)$.'
                        )
                    ],
                    context_after=[],
                )
            ],
            decisions=[InstructionGovernanceDecision(position=0, governs=True)],
        ).with_inputs('instruction_blocks', 'items'),
    ]
