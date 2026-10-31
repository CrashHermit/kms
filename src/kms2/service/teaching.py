"""Application service for goal-driven teaching-session workflows."""

from datetime import UTC, datetime

from kms2.core.model.teaching.question import (
    FreeResponseSpecification,
    FreeResponseView,
    MultipleChoiceView,
    TeachingQuestion,
    TeachingQuestionView,
)
from kms2.core.model.teaching.session import (
    TeachingPhase,
    TeachingSession,
    TeachingSessionSnapshot,
    TeachingSessionView,
    TeachingStatus,
    TeachingTurnCommit,
    TeachingTurnView,
)
from kms2.core.model.teaching.turn import TeachingRole, TeachingTurn
from kms2.database.teaching.repository import TeachingRepository


class TeachingService:
    """Coordinate teaching-session domain transitions and projections."""

    def __init__(self, repository: TeachingRepository) -> None:
        self._teaching = repository

    async def start_session(
        self,
        user_uuid: str,
        goal: str,
    ) -> TeachingSessionView:
        """Start an empty goal-driven session for one user."""
        now = datetime.now(UTC)
        session = TeachingSession(
            goal=goal,
            status=TeachingStatus.AWAITING_TEACHER,
            phase=TeachingPhase.PROBING,
            revision=0,
            created_at=now,
            updated_at=now,
        )
        snapshot = await self._teaching.create_session(user_uuid, session)
        return self._view(snapshot)

    async def list_sessions(self, user_uuid: str) -> list[TeachingSession]:
        """List one user's teaching sessions."""
        return await self._teaching.list_sessions(user_uuid)

    async def load_session(
        self,
        user_uuid: str,
        session_uuid: str,
    ) -> TeachingSessionView:
        """Load one session as a learner-safe view."""
        snapshot = await self._teaching.load_session(user_uuid, session_uuid)
        return self._view(snapshot)

    async def submit_response(
        self,
        user_uuid: str,
        session_uuid: str,
        content: str,
        *,
        expected_revision: int,
    ) -> TeachingSessionView:
        """Append one learner response without grading or inference."""
        turn = TeachingTurn(
            role=TeachingRole.LEARNER,
            content=content,
            created_at=datetime.now(UTC),
        )
        snapshot = await self._teaching.append_response(
            user_uuid,
            session_uuid,
            expected_revision,
            turn,
        )
        return self._view(snapshot)

    async def end_session(
        self,
        user_uuid: str,
        session_uuid: str,
        *,
        expected_revision: int,
    ) -> TeachingSessionView:
        """End one active session while retaining its complete history."""
        snapshot = await self._teaching.end_session(
            user_uuid,
            session_uuid,
            expected_revision,
            datetime.now(UTC),
        )
        return self._view(snapshot)

    async def record_teaching_turn(
        self,
        user_uuid: str,
        session_uuid: str,
        content: str,
        commit: TeachingTurnCommit,
        *,
        expected_revision: int,
    ) -> TeachingSessionSnapshot:
        """Persist a complete teacher result for backend orchestration."""
        turn = TeachingTurn(
            role=TeachingRole.TEACHER,
            content=content,
            created_at=datetime.now(UTC),
        )
        return await self._teaching.record_turn(
            user_uuid,
            session_uuid,
            expected_revision,
            turn,
            commit,
        )

    @classmethod
    def _view(cls, snapshot: TeachingSessionSnapshot) -> TeachingSessionView:
        """Project a private snapshot without exposing answer metadata."""
        return TeachingSessionView(
            session=snapshot.session,
            turns=[
                TeachingTurnView(
                    turn=record.turn,
                    question=(
                        cls._question_view(record.question)
                        if record.question is not None
                        else None
                    ),
                    question_step_uuid=record.question_step_uuid,
                    responds_to_question_uuid=record.responds_to_question_uuid,
                    presented_plan_uuid=record.presented_plan_uuid,
                    accepted_plan_uuid=record.accepted_plan_uuid,
                )
                for record in snapshot.turns
            ],
            current_assessment=snapshot.current_assessment,
            current_plan=snapshot.current_plan,
            current_plan_acceptance=snapshot.current_plan_acceptance,
            current_step_uuid=snapshot.current_step_uuid,
            pending_question_uuid=snapshot.pending_question_uuid,
        )

    @staticmethod
    def _question_view(question: TeachingQuestion) -> TeachingQuestionView:
        """Project a typed question response without its private answer."""
        response = question.response
        if isinstance(response, FreeResponseSpecification):
            response_view = FreeResponseView()
        else:
            response_view = MultipleChoiceView(options=response.options)
        return TeachingQuestionView(
            uuid=question.uuid,
            kind=question.kind,
            prompt=question.prompt,
            response=response_view,
            created_at=question.created_at,
        )
