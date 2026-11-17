"""Owner-scoped persistence for FSRS memory and review provenance."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from neo4j import AsyncSession, AsyncTransaction
from pydantic import TypeAdapter

from kms.knowledge.models import KnowledgeContext
from kms.learning.memory.models import (
    MemoryHistory,
    MemoryReview,
    MemoryState,
    ReviewInput,
    ReviewTargetContext,
)
from kms.learning.memory.queries import (
    LIST_DUE,
    LOCK_MEMORY_OWNER,
    READ_MEMORY,
    READ_REVIEW_INPUT,
    READ_TARGET_ARGUMENT,
    READ_TARGET_PROPOSITION,
    READ_TARGET_SOLUTION_TASK,
    REQUIRE_MEMORY_OWNER,
    SAVE_REVIEW,
)
from kms.learning.memory.records import (
    hydrate_argument,
    hydrate_memory,
    hydrate_proposition,
    hydrate_review,
)
from kms.learning.references import LearningTarget
from kms.learning.session.queries import READ_OWNED_TASK
from kms.learning.session.records import (
    hydrate_task,
    native_datetime,
    read_payload,
)
from kms.learning.task.models import TaskGrade


class MemoryRepository:
    """Persist and query learner memory with owner-scoped graph access."""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    @asynccontextmanager
    async def serialized(
        self, user_uuid: str
    ) -> AsyncIterator[AsyncTransaction]:
        """Hold a short owner write lock until the transaction completes."""
        async with self._session_factory() as session:
            async with await session.begin_transaction() as transaction:
                result = await transaction.run(
                    LOCK_MEMORY_OWNER, user_uuid=user_uuid
                )
                await result.single(strict=True)
                try:
                    yield transaction
                except BaseException:
                    await transaction.rollback()
                    raise
                else:
                    await transaction.commit()

    async def read_review_input(
        self,
        transaction: AsyncTransaction,
        user_uuid: str,
        session_uuid: str,
        assessment_uuid: str,
    ) -> ReviewInput:
        """Read target, response timestamp, and rating from owned graph data."""
        result = await transaction.run(
            READ_REVIEW_INPUT,
            user_uuid=user_uuid,
            session_uuid=session_uuid,
            assessment_uuid=assessment_uuid,
        )
        row = dict(await result.single(strict=True))
        grade = TaskGrade.model_validate(
            read_payload(row['assessment_payload_json'])['grade']
        )
        memory_record = row['memory']
        memory = (
            hydrate_memory(dict(memory_record), row['target'])
            if memory_record is not None
            else None
        )
        return ReviewInput(
            target=TypeAdapter(LearningTarget).validate_python(
                dict(row['target'])
            ),
            response_uuid=row['response_uuid'],
            assessment_uuid=row['assessment_uuid'],
            rating=grade.review_rating,
            reviewed_at=native_datetime(row['reviewed_at']).astimezone(UTC),
            memory=memory,
        )

    async def save_review(
        self,
        transaction: AsyncTransaction,
        user_uuid: str,
        memory: MemoryState,
        review: MemoryReview,
    ) -> MemoryState:
        """Atomically save a card update and its response-based review event."""
        result = await transaction.run(
            SAVE_REVIEW,
            user_uuid=user_uuid,
            memory_uuid=memory.uuid,
            target_kind=memory.target.kind,
            target_uuid=memory.target.uuid,
            card_json=memory.card_json,
            due_at=memory.due_at,
            review_count=memory.review_count,
            review_uuid=review.uuid,
            assessment_uuid=review.assessment_uuid,
            rating=review.rating.value,
            reviewed_at=review.reviewed_at,
            sequence=review.sequence,
            created_at=review.created_at,
            memory_created_at=memory.created_at,
            updated_at=memory.updated_at,
        )
        await result.single(strict=True)
        return memory

    async def read_memory(
        self, user_uuid: str, target: LearningTarget
    ) -> MemoryHistory:
        """Read one owned target's canonical card and ordered review history."""
        async with self._session_factory() as session:
            result = await session.run(
                READ_MEMORY,
                user_uuid=user_uuid,
                target_kind=target.kind,
                target_uuid=target.uuid,
            )
            row = dict(await result.single(strict=True))
        memory = hydrate_memory(dict(row['memory']), row['target'])
        reviews = [
            hydrate_review(
                {
                    **dict(review['review']),
                    'assessment_uuid': review['assessment_uuid'],
                },
                memory.uuid,
            )
            for review in row['reviews']
            if review['review'] is not None
        ]
        reviews.sort(key=lambda item: item.sequence)
        return MemoryHistory(memory=memory, reviews=reviews)

    async def list_due(
        self, user_uuid: str, as_of: datetime
    ) -> list[MemoryState]:
        """List due cards in stable due-time and typed-target order."""
        async with self._session_factory() as session:
            owner_result = await session.run(
                REQUIRE_MEMORY_OWNER, user_uuid=user_uuid
            )
            await owner_result.single(strict=True)
            result = await session.run(
                LIST_DUE, user_uuid=user_uuid, as_of=as_of
            )
            rows = await result.data()
        return [
            hydrate_memory(dict(row['memory']), row['target']) for row in rows
        ]

    async def read_target(
        self, user_uuid: str, target: LearningTarget
    ) -> ReviewTargetContext:
        """Resolve the selected target and only the context it requires."""
        async with self._session_factory() as session:
            if target.kind == 'proposition':
                result = await session.run(
                    READ_TARGET_PROPOSITION,
                    user_uuid=user_uuid,
                    target_uuid=target.uuid,
                )
                row = dict(await result.single(strict=True))
                proposition = hydrate_proposition(dict(row['proposition']))
                knowledge = KnowledgeContext(
                    matched=[proposition],
                    propositions=[proposition],
                    arguments=[],
                )
                return ReviewTargetContext(
                    target=target,
                    understanding=proposition.statement,
                    knowledge=knowledge,
                )

            if target.kind == 'argument':
                result = await session.run(
                    READ_TARGET_ARGUMENT,
                    user_uuid=user_uuid,
                    target_uuid=target.uuid,
                )
                row = dict(await result.single(strict=True))
                premises = [
                    hydrate_proposition(dict(item)) for item in row['premises']
                ]
                conclusion = hydrate_proposition(dict(row['conclusion']))
                argument = hydrate_argument(
                    dict(row['argument']),
                    row['premises'],
                    row['conclusion'],
                )
                propositions = {
                    item.uuid: item for item in [*premises, conclusion]
                }
                return ReviewTargetContext(
                    target=target,
                    understanding=conclusion.statement,
                    knowledge=KnowledgeContext(
                        matched=[],
                        propositions=sorted(
                            propositions.values(), key=lambda item: item.uuid
                        ),
                        arguments=[argument],
                    ),
                )

            result = await session.run(
                READ_TARGET_SOLUTION_TASK,
                user_uuid=user_uuid,
                target_uuid=target.uuid,
            )
            row = dict(await result.single(strict=True))
            result = await session.run(
                READ_OWNED_TASK,
                user_uuid=user_uuid,
                session_uuid=row['session_uuid'],
                task_uuid=row['task_uuid'],
            )
            task = hydrate_task(dict(await result.single(strict=True)))
        return ReviewTargetContext(
            target=target,
            understanding=task.understanding,
            knowledge=KnowledgeContext(
                matched=[], propositions=[], arguments=[]
            ),
            solution_task=task,
        )
