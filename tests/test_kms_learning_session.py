"""Session context selection retains only the requested interaction records."""

from kms.learning.references import GroundingReferences, SolutionTarget
from kms.learning.session.context import select_session_context
from kms.learning.session.models import (
    AssessmentTarget,
    LearningDecisionRecord,
    LearningSession,
    SessionHistory,
)
from kms.learning.task.models import (
    Confidence,
    FreeAnswer,
    ResponseInterpretation,
    ReviewRating,
    Solution,
    Task,
    TaskAssessment,
    TaskGrade,
    TaskResponse,
)
from kms.media.models import Material


def test_session_context_selects_steps_and_dependent_records() -> None:
    def task(uuid: str) -> Task:
        solution = Solution(
            material=Material.from_text('Reference explanation.'),
            assessment_criteria=['Distinguishes the concepts.'],
            grounding=GroundingReferences(),
        )
        return Task(
            uuid=uuid,
            understanding='target',
            prompt=Material.from_text('Explain the distinction.'),
            target=SolutionTarget(uuid=solution.uuid),
            grounding=GroundingReferences(),
            solution=solution,
        )

    first, second = task('first'), task('second')
    decision = LearningDecisionRecord(
        uuid='decision',
        decision=AssessmentTarget(
            understanding='target', explanation='', assessment_uuids=[]
        ),
    )
    responses = [
        TaskResponse(
            uuid=f'r-{item.uuid}',
            task_uuid=item.uuid,
            answer=FreeAnswer(
                material=Material.from_text('My explanation.'),
                confidence=Confidence.MEDIUM,
            ),
        )
        for item in (first, second)
    ]
    assessments = [
        TaskAssessment(
            uuid=f'a-{response.uuid}',
            response_uuid=response.uuid,
            grade=TaskGrade(
                correctness='correct',
                interpretation=ResponseInterpretation(
                    reasoning_verdict='sound',
                    explanation='The response explains the distinction.',
                    demonstrated_understandings=['Distinguishes the concepts.'],
                    gaps=[],
                    possible_misconceptions=[],
                    unresolved_questions=[],
                ),
                review_rating=ReviewRating.GOOD,
            ),
        )
        for response in responses
    ]
    history = SessionHistory(
        session=LearningSession(goal='goal', targets=['target']),
        steps=[first, decision, second],
        responses=responses,
        assessments=assessments,
    )
    context = select_session_context(history, ['second', 'decision'])
    assert [step.uuid for step in context.steps] == ['decision', 'second']
    assert context.responses == [responses[1]]
    assert context.assessments == [assessments[1]]
    assert context.goal == 'goal' and context.targets == ['target']
    initial = select_session_context(history, [])
    assert initial.goal == 'goal' and initial.targets == ['target']
    assert initial.steps == initial.responses == initial.assessments == []
