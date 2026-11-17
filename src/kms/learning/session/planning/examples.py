"""Deterministic demonstrations of unified learning-session planning."""

from datetime import UTC, datetime

import dspy

from kms.learning.examples.image_assets import learning_image
from kms.learning.references import GroundingReferences, PropositionTarget
from kms.learning.session.models import (
    AssessmentTarget,
    LearningFinished,
    SessionContext,
    SessionStep,
    TeachingTarget,
)
from kms.learning.session.planning.models import LearningPlanningRequest
from kms.learning.task.models import (
    Confidence,
    FreeAnswer,
    LearningObservation,
    ResponseInterpretation,
    ReviewRating,
    Solution,
    Task,
    TaskAssessment,
    TaskGrade,
    TaskResponse,
)
from kms.learning.teaching.models import Teaching
from kms.media.models import ImagePart, Material, TextPart

_EXAMPLE_TIME = datetime(2020, 1, 1, tzinfo=UTC)
_GRAPH_TARGET = PropositionTarget(uuid='example-function-value')


def _task() -> Task:
    return Task(
        uuid='example-task',
        created_at=_EXAMPLE_TIME,
        understanding='Read f(2) from the graph, separately from the limit.',
        prompt=Material(
            parts=(
                TextPart(text='For the displayed graph, what is f(2)? '),
                ImagePart(asset=learning_image('function_graph.png')),
            )
        ),
        target=_GRAPH_TARGET,
        grounding=GroundingReferences(unit_uuids=['example-unit']),
        solution=Solution(
            uuid='example-solution',
            created_at=_EXAMPLE_TIME,
            material=Material.from_text(
                'The filled point at x=2 has y-coordinate 5, so f(2)=5.'
            ),
            assessment_criteria=[
                'Identifies the filled point as the function value.'
            ],
            grounding=GroundingReferences(unit_uuids=['example-unit']),
        ),
    )


def _incorrect_assessment() -> tuple[TaskResponse, TaskAssessment]:
    response = TaskResponse(
        uuid='example-conflict-response',
        created_at=_EXAMPLE_TIME,
        task_uuid='example-task',
        answer=FreeAnswer(
            material=Material.from_text(
                'f(2)=5. I guessed; the open circle gives the actual value.'
            ),
            confidence=Confidence.HIGH,
        ),
    )
    assessment = TaskAssessment(
        uuid='example-conflict-assessment',
        created_at=_EXAMPLE_TIME,
        response_uuid=response.uuid,
        grade=TaskGrade(
            correctness='correct',
            review_rating=ReviewRating.AGAIN,
            interpretation=ResponseInterpretation(
                reasoning_verdict='unsound',
                explanation='The value is correct, but the explanation assigns it to the open point.',
                demonstrated_understandings=[],
                gaps=[
                    LearningObservation(
                        description='Identify which graph marker defines f(2).',
                        evidence='The learner attributes the function value to the open point.',
                        uncertainty='This is one response and may reflect a guess.',
                        follow_up='Ask how the filled and open points differ.',
                    )
                ],
                possible_misconceptions=[],
                unresolved_questions=[
                    'Whether the distinction is understood consistently.'
                ],
            ),
        ),
    )
    return response, assessment


def _context(
    goal: str,
    target: str,
    *,
    steps: list[SessionStep],
    responses: list[TaskResponse] | None = None,
    assessments: list[TaskAssessment] | None = None,
) -> SessionContext:
    return SessionContext(
        goal=goal,
        targets=[target],
        steps=steps,
        responses=responses or [],
        assessments=assessments or [],
    )


def learning_planning_examples() -> list[dspy.Example]:
    """Demonstrate assessment, targeted teaching, progress, and bounded finish."""
    task = _task()
    response, assessment = _incorrect_assessment()
    wrong_response = response.model_copy(
        update={
            'uuid': 'example-wrong-response',
            'answer': FreeAnswer(
                material=Material.from_text('f(2)=15.'),
                confidence=Confidence.HIGH,
            ),
        }
    )
    wrong_assessment = assessment.model_copy(
        update={
            'uuid': 'example-wrong-assessment',
            'response_uuid': wrong_response.uuid,
            'grade': assessment.grade.model_copy(
                update={'correctness': 'incorrect'}
            ),
        }
    )
    wrong_context = _context(
        'Distinguish a function value at a point from its limit.',
        task.understanding,
        steps=[task],
        responses=[wrong_response],
        assessments=[wrong_assessment],
    )
    wrong_teaching = TeachingTarget(
        understanding=task.understanding,
        objective='Identify the y-coordinate of the filled point at x=2.',
        explanation='The answer is incorrect; teach the specific graph-reading distinction.',
        assessment_uuids=[wrong_assessment.uuid],
        target=_GRAPH_TARGET,
    )
    supported_response = response.model_copy(
        update={
            'uuid': 'example-supported-response',
            'answer': FreeAnswer(
                material=Material.from_text(
                    'f(2)=5 because the filled point at x=2 is at y=5.'
                ),
                confidence=Confidence.HIGH,
            ),
        }
    )
    supported_assessment = assessment.model_copy(
        update={
            'uuid': 'example-supported-assessment',
            'response_uuid': supported_response.uuid,
            'grade': assessment.grade.model_copy(
                update={
                    'review_rating': ReviewRating.GOOD,
                    'interpretation': assessment.grade.interpretation.model_copy(
                        update={
                            'reasoning_verdict': 'sound',
                            'explanation': 'The learner identifies the filled point and connects it to f(2).',
                            'demonstrated_understandings': [
                                LearningObservation(
                                    description='Read the function value from the filled point.',
                                    evidence='The learner identifies the filled point at y=5.',
                                    uncertainty='None is evident in this response.',
                                    follow_up='',
                                )
                            ],
                            'gaps': [],
                            'unresolved_questions': [],
                        }
                    ),
                }
            ),
        }
    )
    supported = _context(
        'Distinguish a function value at a point from its limit.',
        task.understanding,
        steps=[task],
        responses=[supported_response],
        assessments=[supported_assessment],
    )

    initial = _context(
        'Distinguish a function value at a point from its limit.',
        task.understanding,
        steps=[],
    )
    incorrect = _context(
        initial.goal,
        task.understanding,
        steps=[task],
        responses=[response],
        assessments=[assessment],
    )
    teaching_decision = TeachingTarget(
        understanding=task.understanding,
        objective='Explain which plotted point defines f(2).',
        explanation='The answer is correct but the cited reasoning is unsound; address that distinction.',
        assessment_uuids=[assessment.uuid],
        target=_GRAPH_TARGET,
    )
    teaching_step = Teaching(
        uuid='example-teaching',
        created_at=_EXAMPLE_TIME,
        understanding=task.understanding,
        objective=teaching_decision.objective,
        target=_GRAPH_TARGET,
        material=Material(
            parts=(
                TextPart(
                    text='The filled point gives the function value; the open point indicates the limit.'
                ),
                ImagePart(asset=learning_image('function_graph.png')),
            )
        ),
        grounding=GroundingReferences(unit_uuids=['example-unit']),
        assessment_uuids=[assessment.uuid],
    )
    taught = _context(
        initial.goal,
        task.understanding,
        steps=[task, teaching_step],
        responses=[response],
        assessments=[assessment],
    )
    progressed = _context(
        'Explain how axial tilt changes sunlight across seasons.',
        'Relate axial tilt to sunlight over a year.',
        steps=[],
    )
    finished = _context(
        initial.goal,
        task.understanding,
        steps=[task],
        responses=[response],
        assessments=[assessment],
    )
    return [
        dspy.Example(
            request=LearningPlanningRequest(session=initial),
            decision=AssessmentTarget(
                understanding=initial.targets[0],
                explanation='Begin by assessing one atomic understanding from the goal.',
                assessment_uuids=[],
                target=None,
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=LearningPlanningRequest(session=incorrect),
            decision=teaching_decision,
        ).with_inputs('request'),
        dspy.Example(
            request=LearningPlanningRequest(session=wrong_context),
            decision=wrong_teaching,
        ).with_inputs('request'),
        dspy.Example(
            request=LearningPlanningRequest(session=supported),
            decision=AssessmentTarget(
                understanding='Explain how an open point relates to the limit.',
                explanation='The target is supported; advance to a relevant distinction not yet assessed.',
                assessment_uuids=[supported_assessment.uuid],
                target=None,
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=LearningPlanningRequest(
                session=incorrect,
                target=_GRAPH_TARGET,
            ),
            decision=TeachingTarget(
                understanding=task.understanding,
                objective='Separate the filled function-value point from the open limit point.',
                explanation='The answer is correct, but its reasoning attributes the value to the open point.',
                assessment_uuids=[assessment.uuid],
                target=_GRAPH_TARGET,
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=LearningPlanningRequest(
                session=taught,
                target=_GRAPH_TARGET,
            ),
            decision=AssessmentTarget(
                understanding=task.understanding,
                explanation='Check the same target after teaching before moving on.',
                assessment_uuids=[assessment.uuid],
                target=_GRAPH_TARGET,
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=LearningPlanningRequest(session=progressed),
            decision=AssessmentTarget(
                understanding=progressed.targets[0],
                explanation='Continue with one relevant atomic understanding.',
                assessment_uuids=[],
                target=None,
            ),
        ).with_inputs('request'),
        dspy.Example(
            request=LearningPlanningRequest(session=finished),
            decision=LearningFinished(
                summary='The answer was correct, but the explanation conflicts with the graph. Whether the learner distinguishes the markers remains unresolved.',
                assessment_uuids=[assessment.uuid],
            ),
        ).with_inputs('request'),
    ]
