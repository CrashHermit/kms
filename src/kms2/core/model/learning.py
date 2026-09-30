"""Graph models and service contracts for user card learning."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from kms2.core.model.base import Vertex


class UserCardReview(Vertex):
    """One user's shared FSRS state for one source flashcard."""

    fsrs_card_json: str
    due_at: datetime
    review_count: int


class ReviewEvent(Vertex):
    """Immutable FSRS transition attached to a user card review."""

    review_index: int
    scheduler_json: str
    fsrs_card_before_json: str
    fsrs_card_after_json: str
    fsrs_review_log_json: str


class ReviewRating(StrEnum):
    """Ratings accepted by the learning service."""

    AGAIN = 'again'
    HARD = 'hard'
    GOOD = 'good'
    EASY = 'easy'


class _LearningContract(BaseModel):
    """Base configuration for non-persisted learning contracts."""

    model_config = ConfigDict(extra='forbid')


class DeckCard(_LearningContract):
    """A source flashcard available through a deck or source ownership."""

    card_uuid: str
    question: str
    answer: str


class DueDeckCard(DeckCard):
    """A deck card whose shared user review is due."""

    due_at: datetime


class ReviewSnapshot(_LearningContract):
    """Current user scheduler and card state used for one review."""

    card_uuid: str
    scheduler_json: str
    fsrs_card_json: str
    review_count: int


class ReviewReceipt(_LearningContract):
    """Persisted result of one FSRS review transition."""

    card_uuid: str
    due_at: datetime
    review_count: int
    review_index: int


__all__ = [
    'DeckCard',
    'DueDeckCard',
    'ReviewEvent',
    'ReviewRating',
    'ReviewReceipt',
    'ReviewSnapshot',
    'UserCardReview',
]
