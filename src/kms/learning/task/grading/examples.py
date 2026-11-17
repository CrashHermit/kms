"""Fixed examples for distinct correctness and reasoning evidence."""

from datetime import UTC, datetime

import dspy

from kms.learning.examples.image_assets import learning_image
from kms.learning.references import GroundingReferences, SolutionTarget
from kms.learning.task.grading.models import TaskGradingRequest
from kms.learning.task.models import (
    Confidence,
    FreeAnswer,
    LearningObservation,
    ResponseInterpretation,
    ReviewRating,
    Solution,
    Task,
    TaskGrade,
    TaskResponse,
    UnknownTaskAnswer,
)
from kms.media.models import ImagePart, Material, TextPart

_EXAMPLE_TIME = datetime(2020, 1, 1, tzinfo=UTC)
_TASK_UUID = '00000000-0000-4000-8000-000000000001'


def _task() -> Task:
    graph = learning_image('function_graph.png')
    prompt = Material(
        parts=(
            TextPart(
                text='Use the graph to identify f(2). Explain your approach if useful.'
            ),
            ImagePart(asset=graph),
        )
    )
    return Task(
        uuid=_TASK_UUID,
        created_at=_EXAMPLE_TIME,
        updated_at=_EXAMPLE_TIME,
        understanding="Distinguish a function's value from its limit.",
        prompt=prompt,
        target=SolutionTarget(uuid='00000000-0000-4000-8000-000000000002'),
        grounding=GroundingReferences(),
        solution=Solution(
            uuid='00000000-0000-4000-8000-000000000002',
            created_at=_EXAMPLE_TIME,
            updated_at=_EXAMPLE_TIME,
            material=Material(
                parts=(
                    TextPart(
                        text='The filled point at x=2 has y=5, so f(2)=5. '
                    ),
                    ImagePart(asset=graph),
                    TextPart(
                        text='The open point at y=15 shows the limit, not the value.'
                    ),
                )
            ),
            assessment_criteria=[
                'Identifies the filled point as f(2)=5.',
                'If reasoning is provided, distinguishes the open point at 15 from the function value.',
            ],
            grounding=GroundingReferences(),
        ),
    )


def _algebra_task() -> Task:
    return Task(
        uuid='00000000-0000-4000-8000-000000000004',
        created_at=_EXAMPLE_TIME,
        understanding='Solve a linear equation using equality-preserving operations.',
        prompt=Material.from_text('Solve 2x+3=7. Show your work if useful.'),
        target=SolutionTarget(uuid='00000000-0000-4000-8000-000000000005'),
        grounding=GroundingReferences(),
        solution=Solution(
            uuid='00000000-0000-4000-8000-000000000005',
            created_at=_EXAMPLE_TIME,
            material=Material.from_text(
                'Subtract 3 from both sides to get 2x=4, then divide by 2 to get x=2.'
            ),
            assessment_criteria=[
                'Identifies x=2.',
                'Accepts any valid method; written work is optional.',
            ],
            grounding=GroundingReferences(),
        ),
    )


def _request(
    answer: FreeAnswer | UnknownTaskAnswer, *, task: Task | None = None
) -> TaskGradingRequest:
    task = _task() if task is None else task
    return TaskGradingRequest(
        task=task,
        response=TaskResponse(
            uuid='00000000-0000-4000-8000-000000000003',
            created_at=_EXAMPLE_TIME,
            updated_at=_EXAMPLE_TIME,
            task_uuid=task.uuid,
            answer=answer,
        ),
    )


def _grade(
    correctness: str,
    reasoning_verdict: str,
    explanation: str,
    rating: ReviewRating,
    *,
    demonstrated: list[str] | None = None,
    unresolved: list[str] | None = None,
    gaps: list[LearningObservation] | None = None,
    misconceptions: list[LearningObservation] | None = None,
) -> TaskGrade:
    return TaskGrade(
        correctness=correctness,
        interpretation=ResponseInterpretation(
            reasoning_verdict=reasoning_verdict,
            explanation=explanation,
            demonstrated_understandings=demonstrated or [],
            gaps=gaps or [],
            possible_misconceptions=misconceptions or [],
            unresolved_questions=unresolved or [],
        ),
        review_rating=rating,
    )


def task_grading_examples() -> list[dspy.Example]:
    """Demonstrate alternate methods, uncertain work, and image-backed unknowns."""
    return [
        dspy.Example(
            request=_request(
                FreeAnswer(
                    material=Material.from_text(
                        'I struggled at first. Dividing both sides by 2 gives '
                        'x+1.5=3.5; subtracting 1.5 gives x=2.'
                    ),
                    confidence=Confidence.LOW,
                ),
                task=_algebra_task(),
            ),
            grade=_grade(
                'correct',
                'sound',
                'The learner reports difficulty but completes a valid divide-first method, different from the reference.',
                ReviewRating.HARD,
                demonstrated=[
                    'Solves the equation using valid inverse operations.'
                ],
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=_request(
                FreeAnswer(
                    material=Material.from_text(
                        '5, because the open circle marks the function value.'
                    ),
                    confidence=Confidence.HIGH,
                )
            ),
            grade=_grade(
                'correct',
                'unsound',
                'The value is correct, but the explanation assigns it to the open point.',
                ReviewRating.AGAIN,
                unresolved=[
                    'Can the learner distinguish the open and filled points?'
                ],
                misconceptions=[
                    LearningObservation(
                        description='May confuse open and filled points.',
                        evidence='The learner says the open circle marks the function value.',
                        uncertainty='One response may reflect a wording mistake or a guess.',
                        follow_up='Ask which plotted point defines f(2).',
                    )
                ],
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=_request(
                FreeAnswer(
                    material=Material.from_text(
                        '15; the graph approaches height 15 near x=2.'
                    ),
                    confidence=Confidence.LOW,
                )
            ),
            grade=_grade(
                'incorrect',
                'mixed',
                'The response identifies the limit but not the defined value.',
                ReviewRating.AGAIN,
                demonstrated=['Identifies the approaching height as 15.'],
                unresolved=[
                    'Does the learner recognize the filled point as f(2)?'
                ],
                gaps=[
                    LearningObservation(
                        description='Distinguish the limiting height from the function value.',
                        evidence='The learner gives 15 because nearby values approach 15.',
                        uncertainty='The response does not establish how the filled point is interpreted.',
                        follow_up='Check what the filled point indicates.',
                    )
                ],
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=_request(
                FreeAnswer(
                    material=Material.from_text('5'),
                    confidence=Confidence.MEDIUM,
                )
            ),
            grade=_grade(
                'correct',
                'insufficient_evidence',
                'The answer is correct; the bare response gives no reasoning evidence.',
                ReviewRating.GOOD,
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=_request(
                FreeAnswer(
                    material=Material.from_text('15'),
                    confidence=Confidence.HIGH,
                )
            ),
            grade=_grade(
                'incorrect',
                'insufficient_evidence',
                'The value 15 is incorrect. High confidence does not establish why the learner gave that answer.',
                ReviewRating.AGAIN,
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=_request(
                UnknownTaskAnswer(
                    material=Material(
                        parts=(
                            TextPart(
                                text='The graph approaches 15, but I am unsure whether that is f(2).'
                            ),
                            ImagePart(
                                asset=learning_image('function_graph.png')
                            ),
                        )
                    )
                )
            ),
            grade=_grade(
                'unknown',
                'mixed',
                'The learner identifies the approaching height but is unsure about the defined value.',
                ReviewRating.AGAIN,
                demonstrated=['Recognizes the limit height as 15.'],
                unresolved=['Can the learner read the filled point as f(2)=5?'],
            ),
        ).with_inputs('request'),
    ]
