"""Application service for user decks and FSRS card scheduling."""

from datetime import datetime

from fsrs import Card, Rating, Scheduler

from kms2.core.model.learning import (
    DeckCard,
    DueDeckCard,
    ReviewEvent,
    ReviewRating,
    ReviewReceipt,
    UserCardReview,
)
from kms2.core.model.user import Deck
from kms2.database.learning.repository import LearningRepository


class LearningService:
    """Coordinate graph persistence and FSRS state transitions."""

    def __init__(self, repository: LearningRepository) -> None:
        self._learning = repository

    async def list_decks(self, user_uuid: str) -> list[Deck]:
        """List a user's decks."""
        return await self._learning.list_decks(user_uuid)

    async def create_deck(self, user_uuid: str, name: str) -> Deck:
        """Create a named deck for a user."""
        return await self._learning.create_deck(user_uuid, Deck(name=name))

    async def list_assignable_cards(self, user_uuid: str) -> list[DeckCard]:
        """List source cards the user may add to a deck."""
        return await self._learning.list_assignable_cards(user_uuid)

    async def add_card_to_deck(
        self,
        user_uuid: str,
        deck_uuid: str,
        card_uuid: str,
    ) -> DeckCard:
        """Add a card and initialize shared FSRS state once."""
        card = Card()
        review = self._new_review(card)
        return await self._learning.add_card_to_deck(
            user_uuid,
            deck_uuid,
            card_uuid,
            review,
        )

    async def remove_card_from_deck(
        self,
        user_uuid: str,
        deck_uuid: str,
        card_uuid: str,
    ) -> None:
        """Remove one deck membership without changing review history."""
        await self._learning.remove_card_from_deck(
            user_uuid,
            deck_uuid,
            card_uuid,
        )

    async def list_deck_cards(
        self,
        user_uuid: str,
        deck_uuid: str,
    ) -> list[DeckCard]:
        """List cards in one deck."""
        return await self._learning.list_deck_cards(user_uuid, deck_uuid)

    async def list_due_deck_cards(
        self,
        user_uuid: str,
        deck_uuid: str,
        now: datetime,
    ) -> list[DueDeckCard]:
        """List cards due at or before the supplied UTC time."""
        return await self._learning.list_due_deck_cards(
            user_uuid, deck_uuid, now
        )

    async def list_review_events(
        self,
        user_uuid: str,
        card_uuid: str,
    ) -> list[ReviewEvent]:
        """List one user's card history in review order."""
        return await self._learning.list_review_events(user_uuid, card_uuid)

    async def review_card(
        self,
        user_uuid: str,
        deck_uuid: str,
        card_uuid: str,
        rating: ReviewRating,
        review_datetime: datetime,
        review_duration: int | None = None,
    ) -> ReviewReceipt:
        """Apply one rating through FSRS and persist its complete transition."""
        snapshot = await self._learning.load_review_snapshot(
            user_uuid,
            deck_uuid,
            card_uuid,
        )
        scheduler = Scheduler.from_json(snapshot.scheduler_json)
        card = Card.from_json(snapshot.fsrs_card_json)
        updated_card, review_log = scheduler.review_card(
            card,
            self._fsrs_rating(rating),
            review_datetime=review_datetime,
            review_duration=review_duration,
        )
        scheduler_json = scheduler.to_json()
        updated_card_json = updated_card.to_json()
        event = ReviewEvent(
            review_index=snapshot.review_count + 1,
            scheduler_json=scheduler_json,
            fsrs_card_before_json=snapshot.fsrs_card_json,
            fsrs_card_after_json=updated_card_json,
            fsrs_review_log_json=review_log.to_json(),
        )
        return await self._learning.record_review(
            user_uuid,
            deck_uuid,
            card_uuid,
            updated_card_json,
            updated_card.due,
            scheduler_json,
            event,
        )

    @staticmethod
    def _new_review(card: Card) -> UserCardReview:
        return UserCardReview(
            fsrs_card_json=card.to_json(),
            due_at=card.due,
            review_count=0,
        )

    @staticmethod
    def _fsrs_rating(rating: ReviewRating) -> Rating:
        return {
            ReviewRating.AGAIN: Rating.Again,
            ReviewRating.HARD: Rating.Hard,
            ReviewRating.GOOD: Rating.Good,
            ReviewRating.EASY: Rating.Easy,
        }[rating]
