"""User-owned learner observation persistence and cosine retrieval."""

from collections.abc import Callable

from neo4j import AsyncSession, AsyncTransaction

from kms.core.model.teaching.observation import (
    LearnerObservationRecord,
    LearnerObservationSearchCandidate,
    LearnerObservationWrite,
)
from kms.database.teaching.queries.observation import (
    READ_OWNED_LEARNER_OBSERVATIONS,
    SEARCH_OWNED_LEARNER_OBSERVATIONS,
    WRITE_LEARNER_OBSERVATION,
)
from kms.database.teaching.records import (
    learner_observation_from_record,
)


class LearnerObservationRepository:
    """Read observations reachable through one user's ownership edge."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def search(
        self,
        user_uuid: str,
        query_embedding: list[float],
        *,
        candidate_limit: int,
    ) -> list[LearnerObservationSearchCandidate]:
        """Return owner-scoped cosine candidates before reranking."""
        async with self._session_factory() as session:
            result = await session.run(
                SEARCH_OWNED_LEARNER_OBSERVATIONS,
                user_uuid=user_uuid,
                query_embedding=query_embedding,
                candidate_limit=candidate_limit,
            )
            rows = await result.data()
        return [
            LearnerObservationSearchCandidate(
                record=learner_observation_from_record(row['record']),
                similarity=row['similarity'],
            )
            for row in rows
        ]

    async def read(
        self,
        user_uuid: str,
        observation_uuids: list[str],
    ) -> list[LearnerObservationRecord]:
        """Read requested owned observations in request order."""
        if not observation_uuids:
            return []
        async with self._session_factory() as session:
            result = await session.run(
                READ_OWNED_LEARNER_OBSERVATIONS,
                user_uuid=user_uuid,
                observation_uuids=observation_uuids,
            )
            rows = await result.data()
        if len(rows) != len(observation_uuids):
            raise ValueError(
                'Requested learner observation is not owned or no longer available'
            )
        return [learner_observation_from_record(row['record']) for row in rows]


async def write_learner_observations(
    transaction: AsyncTransaction,
    user_uuid: str,
    teacher_turn_uuid: str,
    writes: list[LearnerObservationWrite],
) -> None:
    """Write a batch with one strict query result per observation."""
    for item in writes:
        observation = item.observation
        result = await transaction.run(
            WRITE_LEARNER_OBSERVATION,
            user_uuid=user_uuid,
            teacher_turn_uuid=teacher_turn_uuid,
            evidence_turn_uuids=item.evidence_turn_uuids,
            uuid=observation.uuid,
            kind=observation.kind.value,
            topic=observation.topic,
            description=observation.description,
            embedding=observation.embedding,
            created_at=observation.created_at,
        )
        await result.single(strict=True)
