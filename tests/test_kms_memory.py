"""Behavioral tests for deterministic FSRS review application."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest
from fsrs import Card, Rating, Scheduler

from kms.learning.memory.models import (
    MemoryReview,
    MemoryReviewState,
    MemoryState,
    ReviewInput,
    memory_uuid,
    review_uuid,
)
from kms.learning.memory.node import MemoryReviewNode
from kms.learning.references import ArgumentTarget, PropositionTarget
from kms.learning.task.models import ReviewRating


def test_memory_and_review_ids_follow_persisted_identity_contract() -> None:
    target = PropositionTarget(uuid='same')
    identity = json.dumps(
        ['user', 'proposition', 'same'], separators=(',', ':')
    )
    assert memory_uuid('user', target) == str(
        uuid5(NAMESPACE_URL, f'kms:learning-memory:{identity}')
    )
    assert memory_uuid('user', target) != memory_uuid(
        'user', ArgumentTarget(uuid='same')
    )
    response_identity = json.dumps(['user', 'response'], separators=(',', ':'))
    assert review_uuid('user', 'response') == str(
        uuid5(NAMESPACE_URL, f'kms:learning-review:{response_identity}')
    )


class MemoryRepositoryFake:
    """Small owner-transaction fake that rejects duplicate review identities."""

    def __init__(self) -> None:
        self.review_input: ReviewInput
        self.memory: MemoryState | None = None
        self.reviews: list[MemoryReview] = []

    @asynccontextmanager
    async def serialized(self, user_uuid: str) -> AsyncIterator[object]:
        yield self

    async def read_review_input(
        self,
        transaction: object,
        user_uuid: str,
        session_uuid: str,
        assessment_uuid: str,
    ) -> ReviewInput:
        return self.review_input

    async def save_review(
        self,
        transaction: object,
        user_uuid: str,
        memory: MemoryState,
        review: MemoryReview,
    ) -> MemoryState:
        if any(item.uuid == review.uuid for item in self.reviews):
            raise ValueError('duplicate review UUID')
        self.memory = memory
        self.reviews.append(review)
        return memory


def review_input(
    *,
    when: datetime,
    rating: ReviewRating,
    response_uuid: str,
    assessment_uuid: str,
    memory: MemoryState | None = None,
) -> ReviewInput:
    return ReviewInput(
        target=PropositionTarget(uuid='p-1'),
        response_uuid=response_uuid,
        assessment_uuid=assessment_uuid,
        rating=rating,
        reviewed_at=when,
        memory=memory,
    )


async def _review_node_matches_pinned_fsrs_rating(
    rating: ReviewRating, expected: Rating
) -> None:
    timestamp = datetime(2026, 1, 2, 12, tzinfo=UTC)
    repository = MemoryRepositoryFake()
    repository.review_input = review_input(
        when=timestamp,
        rating=rating,
        response_uuid='response-1',
        assessment_uuid='assessment-1',
    )
    scheduler = Scheduler(enable_fuzzing=False)
    node = MemoryReviewNode(scheduler, repository)

    result = await node.run(
        {
            'user_uuid': 'user-1',
            'session_uuid': 'session-1',
            'assessment_uuid': 'assessment-1',
        }
    )

    identifier = memory_uuid('user-1', PropositionTarget(uuid='p-1'))
    expected_card = Card(card_id=int(UUID(identifier)), due=timestamp)
    expected_card, review_log = scheduler.review_card(
        expected_card, expected, review_datetime=timestamp
    )
    memory = result['memory']
    assert memory.uuid == identifier
    assert memory.card_json == expected_card.to_json()
    assert memory.due_at == expected_card.due
    assert memory.review_count == 1
    assert repository.reviews[0].rating == rating
    assert repository.reviews[0].reviewed_at == timestamp
    assert repository.reviews[0].sequence == 1
    assert review_log.rating == expected


async def _review_node_serializes_chronological_fsrs_state() -> None:
    repository = MemoryRepositoryFake()
    scheduler = Scheduler(enable_fuzzing=False)
    node = MemoryReviewNode(scheduler, repository)
    ratings = [ReviewRating.GOOD, ReviewRating.EASY, ReviewRating.HARD]
    expected_card: Card | None = None
    start = datetime(2026, 2, 1, 9, tzinfo=UTC)
    first_created_at: datetime | None = None

    for index, rating in enumerate(ratings):
        timestamp = start + timedelta(days=index * 4)
        repository.review_input = review_input(
            when=timestamp,
            rating=rating,
            response_uuid=f'response-{index}',
            assessment_uuid=f'assessment-{index}',
            memory=repository.memory,
        )
        result = await node.run(
            {
                'user_uuid': 'user-1',
                'session_uuid': f'session-{index}',
                'assessment_uuid': f'assessment-{index}',
            }
        )
        if expected_card is None:
            expected_card = Card(
                card_id=int(
                    UUID(memory_uuid('user-1', PropositionTarget(uuid='p-1')))
                ),
                due=timestamp,
            )
        expected_card, _ = scheduler.review_card(
            expected_card,
            {
                ReviewRating.AGAIN: Rating.Again,
                ReviewRating.HARD: Rating.Hard,
                ReviewRating.GOOD: Rating.Good,
                ReviewRating.EASY: Rating.Easy,
            }[rating],
            review_datetime=timestamp,
        )
        assert result['memory'].card_json == expected_card.to_json()
        assert result['memory'].due_at == expected_card.due
        if first_created_at is None:
            first_created_at = result['memory'].created_at
        assert result['memory'].created_at == first_created_at
        assert result['memory'].review_count == index + 1

    assert [review.sequence for review in repository.reviews] == [1, 2, 3]
    assert len({review.memory_uuid for review in repository.reviews}) == 1
    assert [review.uuid for review in repository.reviews] == [
        review_uuid('user-1', f'response-{index}') for index in range(3)
    ]


async def _duplicate_response_does_not_replace_memory_state() -> None:
    repository = MemoryRepositoryFake()
    node = MemoryReviewNode(Scheduler(enable_fuzzing=False), repository)
    timestamp = datetime(2026, 3, 1, 9, tzinfo=UTC)
    repository.review_input = review_input(
        when=timestamp,
        rating=ReviewRating.GOOD,
        response_uuid='same-response',
        assessment_uuid='first-grade',
    )
    state: MemoryReviewState = {
        'user_uuid': 'user-1',
        'session_uuid': 'session-1',
        'assessment_uuid': 'first-grade',
    }
    previous = (await node.run(state))['memory']
    repository.review_input = review_input(
        when=timestamp + timedelta(days=2),
        rating=ReviewRating.EASY,
        response_uuid='same-response',
        assessment_uuid='regrade',
        memory=previous,
    )

    with pytest.raises(ValueError, match='duplicate review UUID'):
        await node.run({**state, 'assessment_uuid': 'regrade'})

    assert repository.memory == previous
    assert len(repository.reviews) == 1


async def _review_before_last_attempt_is_rejected_without_save() -> None:
    repository = MemoryRepositoryFake()
    node = MemoryReviewNode(Scheduler(enable_fuzzing=False), repository)
    timestamp = datetime(2026, 4, 1, 9, tzinfo=UTC)
    repository.review_input = review_input(
        when=timestamp,
        rating=ReviewRating.GOOD,
        response_uuid='newer-response',
        assessment_uuid='grade-1',
    )
    state: MemoryReviewState = {
        'user_uuid': 'user-1',
        'session_uuid': 'session-1',
        'assessment_uuid': 'grade-1',
    }
    previous = (await node.run(state))['memory']
    repository.review_input = review_input(
        when=timestamp - timedelta(seconds=1),
        rating=ReviewRating.AGAIN,
        response_uuid='older-response',
        assessment_uuid='grade-2',
        memory=previous,
    )

    with pytest.raises(
        ValueError, match='Review precedes the latest recorded attempt'
    ):
        await node.run({**state, 'assessment_uuid': 'grade-2'})

    assert repository.memory == previous
    assert len(repository.reviews) == 1


@pytest.mark.parametrize(
    ('rating', 'expected'),
    [
        (ReviewRating.AGAIN, Rating.Again),
        (ReviewRating.HARD, Rating.Hard),
        (ReviewRating.GOOD, Rating.Good),
        (ReviewRating.EASY, Rating.Easy),
    ],
)
def test_review_node_matches_pinned_fsrs_rating(
    rating: ReviewRating, expected: Rating
) -> None:
    asyncio.run(_review_node_matches_pinned_fsrs_rating(rating, expected))


def test_review_node_serializes_chronological_fsrs_state() -> None:
    asyncio.run(_review_node_serializes_chronological_fsrs_state())


def test_duplicate_response_does_not_replace_memory_state() -> None:
    asyncio.run(_duplicate_response_does_not_replace_memory_state())


def test_review_before_last_attempt_is_rejected_without_save() -> None:
    asyncio.run(_review_before_last_attempt_is_rejected_without_save())
