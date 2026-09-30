import asyncio
import json
from datetime import UTC, datetime

from fsrs import Card, Rating, Scheduler

from kms2.core.model.learning import ReviewRating
from kms2.database.learning.queries import (
    ADD_CARD_TO_DECK,
    LOAD_REVIEW_SNAPSHOT,
    RECORD_REVIEW,
)
from kms2.database.learning.repository import LearningRepository
from kms2.service.learning import LearningService


class _Result:
    def __init__(self, *, row=None):
        self.row = row

    async def single(self, *, strict=False):
        return self.row

    async def data(self):
        return []

    async def consume(self):
        return None


class _Session:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def run(self, query, **parameters):
        self.calls.append((query, parameters))
        return self.responses[query]


def test_add_card_initializes_one_shared_fsrs_card():
    session = _Session(
        {
            ADD_CARD_TO_DECK: _Result(
                row={
                    'card_uuid': 'card-1',
                    'question': 'Question',
                    'answer': 'Answer',
                }
            )
        }
    )
    service = LearningService(LearningRepository(lambda: session))

    card = asyncio.run(service.add_card_to_deck('user-1', 'deck-1', 'card-1'))

    assert card.card_uuid == 'card-1'
    _, parameters = session.calls[0]
    persisted_card = Card.from_json(parameters['fsrs_card_json'])
    assert parameters['review_count'] == 0
    assert persisted_card.due == parameters['due_at']


def test_good_review_persists_fsrs_transition_and_first_event():
    scheduler = Scheduler()
    card = Card()
    review_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    session = _Session(
        {
            LOAD_REVIEW_SNAPSHOT: _Result(
                row={
                    'card_uuid': 'card-1',
                    'scheduler_json': scheduler.to_json(),
                    'fsrs_card_json': card.to_json(),
                    'review_count': 0,
                }
            ),
            RECORD_REVIEW: _Result(
                row={
                    'card_uuid': 'card-1',
                    'due_at': review_time,
                    'review_count': 1,
                    'review_index': 1,
                }
            ),
        }
    )
    service = LearningService(LearningRepository(lambda: session))

    receipt = asyncio.run(
        service.review_card(
            'user-1',
            'deck-1',
            'card-1',
            ReviewRating.GOOD,
            review_time,
        )
    )

    assert receipt.review_index == 1
    assert [query for query, _ in session.calls] == [
        LOAD_REVIEW_SNAPSHOT,
        RECORD_REVIEW,
    ]
    _, parameters = session.calls[1]
    assert parameters['fsrs_card_before_json'] == card.to_json()
    after = Card.from_json(parameters['fsrs_card_after_json'])
    expected, expected_log = Scheduler.from_json(
        scheduler.to_json()
    ).review_card(
        Card.from_json(card.to_json()),
        Rating.Good,
        review_datetime=review_time,
    )
    assert after.to_json() == expected.to_json()
    assert json.loads(parameters['fsrs_review_log_json']) == json.loads(
        expected_log.to_json()
    )
