"""Application service for goal-driven teaching-session workflows."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from kms2.core.model.teaching.evidence import (
    TeachingCitationMedia,
    TeachingCitationView,
    TeachingEvidenceReference,
)
from kms2.core.model.teaching.question import question_view
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
from kms2.service.learner_observations import LearnerObservationService


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

    async def read_teaching_citations(
        self,
        user_uuid: str,
        session_uuid: str,
        references: list[TeachingEvidenceReference],
    ) -> list[TeachingCitationView]:
        """Read safe projections of evidence cited in one owned session."""
        return await self._teaching.read_teaching_citations(
            user_uuid,
            session_uuid,
            references,
        )

    async def read_teaching_citation_media(
        self,
        user_uuid: str,
        session_uuid: str,
        asset_uuid: str,
    ) -> TeachingCitationMedia:
        """Read actual media bytes for an image cited in the owned session."""
        path = await self._teaching.read_teaching_citation_media_path(
            user_uuid,
            session_uuid,
            asset_uuid,
        )
        data = await asyncio.to_thread(Path(path).read_bytes)
        return TeachingCitationMedia(
            media_type=_image_media_type(data),
            data=data,
        )

    async def load_snapshot(
        self,
        user_uuid: str,
        session_uuid: str,
    ) -> TeachingSessionSnapshot:
        """Load the private aggregate for backend teaching orchestration."""
        return await self._teaching.load_session(user_uuid, session_uuid)

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

    async def record_result(
        self,
        user_uuid: str,
        session_uuid: str,
        content: str,
        commit: TeachingTurnCommit,
        *,
        expected_revision: int,
    ) -> TeachingSessionView:
        """Persist one complete teacher result and return its safe projection."""
        snapshot = await self.record_teaching_turn(
            user_uuid,
            session_uuid,
            content,
            commit,
            expected_revision=expected_revision,
        )
        return self._view(snapshot)

    @staticmethod
    def _view(snapshot: TeachingSessionSnapshot) -> TeachingSessionView:
        """Project a private snapshot without exposing answer metadata."""
        return TeachingSessionView(
            session=snapshot.session,
            turns=[
                TeachingTurnView(
                    turn=record.turn,
                    question=(
                        question_view(record.question)
                        if record.question is not None
                        else None
                    ),
                    question_step_uuid=record.question_step_uuid,
                    responds_to_question_uuid=record.responds_to_question_uuid,
                    presented_plan_uuid=record.presented_plan_uuid,
                    accepted_plan_uuid=record.accepted_plan_uuid,
                    citations=record.citations,
                    knowledge_assertions=record.knowledge_assertions,
                    learner_observations=[
                        LearnerObservationService.view(observation)
                        for observation in record.learner_observations
                    ],
                )
                for record in snapshot.turns
            ],
            current_assessment=snapshot.current_assessment,
            current_plan=snapshot.current_plan,
            current_plan_acceptance=snapshot.current_plan_acceptance,
            current_step_uuid=snapshot.current_step_uuid,
            pending_question_uuid=snapshot.pending_question_uuid,
        )


def _image_media_type(data: bytes) -> str:
    """Identify the actual stored image format from its bytes."""
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if data.startswith((b'GIF87a', b'GIF89a')):
        return 'image/gif'
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return 'image/webp'
    if b'<svg' in data[:1024]:
        return 'image/svg+xml'
    raise ValueError('Cited image has an unsupported media format')
