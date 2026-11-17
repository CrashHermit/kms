"""Apply graded learner responses to FSRS memory state."""

from datetime import UTC, datetime
from uuid import UUID

from fsrs import Card, Rating, Scheduler

from kms.learning.memory.models import (
    MemoryReview,
    MemoryReviewState,
    MemoryState,
    memory_uuid,
    review_uuid,
)
from kms.learning.memory.repository import MemoryRepository
from kms.learning.task.models import ReviewRating

_RATINGS = {
    ReviewRating.AGAIN: Rating.Again,
    ReviewRating.HARD: Rating.Hard,
    ReviewRating.GOOD: Rating.Good,
    ReviewRating.EASY: Rating.Easy,
}


class MemoryReviewNode:
    """Update one owner's FSRS state and persist its review atomically."""

    def __init__(
        self, scheduler: Scheduler, repository: MemoryRepository
    ) -> None:
        self._scheduler = scheduler
        self._repository = repository

    async def run(self, state: MemoryReviewState) -> dict[str, MemoryState]:
        async with self._repository.serialized(
            state['user_uuid']
        ) as transaction:
            review_input = await self._repository.read_review_input(
                transaction,
                state['user_uuid'],
                state['session_uuid'],
                state['assessment_uuid'],
            )
            identifier = memory_uuid(state['user_uuid'], review_input.target)
            if review_input.memory is None:
                card = Card(
                    card_id=int(UUID(identifier)), due=review_input.reviewed_at
                )
                sequence = 1
                created_at = datetime.now(UTC)
            else:
                card = Card.from_json(review_input.memory.card_json)
                if (
                    card.last_review is not None
                    and review_input.reviewed_at < card.last_review
                ):
                    raise ValueError(
                        'Review precedes the latest recorded attempt'
                    )
                sequence = review_input.memory.review_count + 1
                created_at = review_input.memory.created_at

            card, _ = self._scheduler.review_card(
                card,
                _RATINGS[review_input.rating],
                review_datetime=review_input.reviewed_at,
            )
            memory = MemoryState(
                uuid=identifier,
                created_at=created_at,
                updated_at=datetime.now(UTC),
                target=review_input.target,
                card_json=card.to_json(),
                due_at=card.due,
                review_count=sequence,
            )
            review = MemoryReview(
                uuid=review_uuid(
                    state['user_uuid'], review_input.response_uuid
                ),
                memory_uuid=identifier,
                assessment_uuid=review_input.assessment_uuid,
                rating=review_input.rating,
                reviewed_at=review_input.reviewed_at,
                sequence=sequence,
            )
            persisted = await self._repository.save_review(
                transaction, state['user_uuid'], memory, review
            )
            return {'memory': persisted}
