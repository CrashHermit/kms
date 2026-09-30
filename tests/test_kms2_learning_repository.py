import asyncio
from datetime import UTC, datetime

from kms2.core.model.learning import ReviewEvent, UserCardReview
from kms2.database.learning.queries.card import ADD_CARD_TO_DECK
from kms2.database.learning.queries.review import RECORD_REVIEW
from kms2.database.learning.repository import LearningRepository


class _Result:
    def __init__(self, *, rows=None, row=None):
        self.rows = rows or []
        self.row = row

    async def data(self):
        return self.rows

    async def single(self, *, strict=False):
        return self.row

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


def _review() -> UserCardReview:
    return UserCardReview(
        fsrs_card_json='card-json',
        due_at=datetime(2026, 1, 1, tzinfo=UTC),
        review_count=0,
    )


def test_add_card_to_deck_persists_initial_review_state():
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
    repository = LearningRepository(lambda: session)

    card = asyncio.run(
        repository.add_card_to_deck('user-1', 'deck-1', 'card-1', _review())
    )

    assert card.card_uuid == 'card-1'
    _, parameters = session.calls[0]
    assert parameters['user_uuid'] == 'user-1'
    assert parameters['deck_uuid'] == 'deck-1'
    assert parameters['card_uuid'] == 'card-1'
    assert parameters['review_count'] == 0


def test_record_review_maps_ordered_receipt_and_event_payload():
    session = _Session(
        {
            RECORD_REVIEW: _Result(
                row={
                    'card_uuid': 'card-1',
                    'due_at': datetime(2026, 1, 2, tzinfo=UTC),
                    'review_count': 1,
                    'review_index': 1,
                }
            )
        }
    )
    repository = LearningRepository(lambda: session)
    event = ReviewEvent(
        review_index=1,
        scheduler_json='scheduler-after',
        fsrs_card_before_json='before',
        fsrs_card_after_json='after',
        fsrs_review_log_json='log',
    )

    receipt = asyncio.run(
        repository.record_review(
            'user-1',
            'deck-1',
            'card-1',
            'after',
            datetime(2026, 1, 2, tzinfo=UTC),
            'scheduler-after',
            event,
        )
    )

    assert receipt.review_index == 1
    _, parameters = session.calls[0]
    assert parameters['fsrs_card_before_json'] == 'before'
    assert parameters['fsrs_card_after_json'] == 'after'
    assert parameters['fsrs_review_log_json'] == 'log'
