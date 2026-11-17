"""Curated demonstrations for atomic target decomposition."""

import dspy

from kms.learning.probe.goal.models import ClarifiedGoal


def target_decomposition_examples() -> list[dspy.Example]:
    """Return examples distinguishing a focused goal from a compound one."""
    return [
        dspy.Example(
            goal=ClarifiedGoal(
                goal=(
                    'As a Canadian high-school student, I want to understand '
                    'why seasons occur, even though I have heard Earth is '
                    'closer to the Sun in January.'
                )
            ),
            targets=[
                'Earth’s axial tilt as the cause of seasons, including why '
                'Earth’s distance from the Sun does not explain seasonal '
                'changes, for a Canadian high-school student.'
            ],
        ).with_inputs('goal'),
        dspy.Example(
            goal=ClarifiedGoal(
                goal=(
                    'As a Canadian high-school student, I want to understand '
                    'how simple and compound interest differ and how to '
                    'calculate a compound-interest balance, using $1,000 at '
                    '5% annually for 3 years as an example.'
                )
            ),
            targets=[
                'Differences between simple and compound interest in how '
                'interest is calculated, in a savings context relevant to a '
                'Canadian high-school student.',
                'Calculation of a savings balance under compound interest, '
                'using the $1,000 initial amount, 5% annual rate, and 3-year '
                'period as an example for a Canadian high-school student.',
            ],
        ).with_inputs('goal'),
    ]
