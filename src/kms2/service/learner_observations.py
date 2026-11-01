"""Prepare append-only learner observations and expose safe views."""

import uuid
from datetime import UTC, datetime

from kms2.core.model.teaching.observation import (
    LearnerObservation,
    LearnerObservationDraft,
    LearnerObservationRecord,
    LearnerObservationView,
    LearnerObservationWrite,
)
from kms2.core.model.teaching.session import TeachingSessionSnapshot
from kms2.database.teaching.observation_repository import (
    LearnerObservationRepository,
)


class LearnerObservationService:
    """Assign durable identities and resolve evidence to learner turns."""

    def __init__(self, repository: LearnerObservationRepository) -> None:
        self._repository = repository

    async def read(
        self,
        user_uuid: str,
        observation_uuids: list[str],
    ) -> list[LearnerObservationRecord]:
        """Read user-owned observations in requested order."""
        return await self._repository.read(user_uuid, observation_uuids)

    def prepare(
        self,
        drafts: list[LearnerObservationDraft],
        snapshot: TeachingSessionSnapshot,
        visible_turn_indexes: list[int],
        embeddings: list[list[float]],
    ) -> list[LearnerObservationWrite]:
        """Assign observation identities and resolve visible learner evidence."""
        return [
            LearnerObservationWrite(
                observation=LearnerObservation(
                    uuid=str(uuid.uuid4()),
                    kind=draft.kind,
                    topic=draft.topic,
                    description=draft.description,
                    embedding=embedding,
                    created_at=datetime.now(UTC),
                ),
                evidence_turn_uuids=list(
                    dict.fromkeys(
                        snapshot.turns[visible_turn_indexes[position]].turn.uuid
                        for position in draft.evidence_turn_positions
                    )
                ),
            )
            for draft, embedding in zip(drafts, embeddings, strict=True)
        ]

    @staticmethod
    def view(record: LearnerObservationRecord) -> LearnerObservationView:
        """Project an internal observation without its embedding."""
        observation = record.observation
        return LearnerObservationView(
            uuid=observation.uuid,
            kind=observation.kind,
            topic=observation.topic,
            description=observation.description,
            created_at=observation.created_at,
            evidence=record.evidence,
        )
