"""Persistence access for decks, source cards, and user review state.

Deck creation, card assignment, snapshot loads, and review recording require one
matching user-scoped record. Missing or duplicate matches raise Neo4j's
``ResultNotSingleError`` at this boundary.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any

from neo4j import Record
from neo4j.time import DateTime

from kms2.core.model.learning import (
    DeckCard,
    DueDeckCard,
    ReviewEvent,
    ReviewReceipt,
    ReviewSnapshot,
    UserCardReview,
)
from kms2.core.model.user import Deck
from kms2.database.learning.queries import (
    ADD_CARD_TO_DECK,
    CREATE_DECK,
    LIST_ASSIGNABLE_CARDS,
    LIST_DECK_CARDS,
    LIST_DECKS,
    LIST_DUE_DECK_CARDS,
    LIST_REVIEW_EVENTS,
    LOAD_REVIEW_SNAPSHOT,
    RECORD_REVIEW,
    REMOVE_CARD_FROM_DECK,
)


class LearningRepository:
    """Persist user-scoped deck membership and review transitions."""

    def __init__(self, session_factory: Callable[..., Any]) -> None:
        self._session_factory = session_factory

    async def list_decks(self, user_uuid: str) -> list[Deck]:
        """List one user's decks in stable display order."""
        rows = await self._data(LIST_DECKS, user_uuid=user_uuid)
        return [Deck(uuid=row['uuid'], name=row['name']) for row in rows]

    async def create_deck(self, user_uuid: str, deck: Deck) -> Deck:
        """Create a deck owned by one user."""
        row = await self._single(
            CREATE_DECK,
            user_uuid=user_uuid,
            deck_uuid=deck.uuid,
            deck_name=deck.name,
        )
        return Deck(uuid=row['uuid'], name=row['name'])

    async def list_assignable_cards(self, user_uuid: str) -> list[DeckCard]:
        """List cards reachable from sources owned by one user."""
        rows = await self._data(LIST_ASSIGNABLE_CARDS, user_uuid=user_uuid)
        return [DeckCard.model_validate(row) for row in rows]

    async def add_card_to_deck(
        self,
        user_uuid: str,
        deck_uuid: str,
        card_uuid: str,
        new_review: UserCardReview,
    ) -> DeckCard:
        """Add a card and initialize its shared review state if needed."""
        row = await self._single(
            ADD_CARD_TO_DECK,
            user_uuid=user_uuid,
            deck_uuid=deck_uuid,
            card_uuid=card_uuid,
            review_uuid=new_review.uuid,
            fsrs_card_json=new_review.fsrs_card_json,
            due_at=new_review.due_at,
            review_count=new_review.review_count,
        )
        return DeckCard.model_validate(row)

    async def remove_card_from_deck(
        self,
        user_uuid: str,
        deck_uuid: str,
        card_uuid: str,
    ) -> None:
        """Remove only one deck membership for a card."""
        await self._consume(
            REMOVE_CARD_FROM_DECK,
            user_uuid=user_uuid,
            deck_uuid=deck_uuid,
            card_uuid=card_uuid,
        )

    async def list_deck_cards(
        self,
        user_uuid: str,
        deck_uuid: str,
    ) -> list[DeckCard]:
        """List cards in one user's deck."""
        rows = await self._data(
            LIST_DECK_CARDS,
            user_uuid=user_uuid,
            deck_uuid=deck_uuid,
        )
        return [DeckCard.model_validate(row) for row in rows]

    async def list_due_deck_cards(
        self,
        user_uuid: str,
        deck_uuid: str,
        now: datetime,
    ) -> list[DueDeckCard]:
        """List due cards in one user's deck in due order."""
        rows = await self._data(
            LIST_DUE_DECK_CARDS,
            user_uuid=user_uuid,
            deck_uuid=deck_uuid,
            now=now,
        )
        return [
            DueDeckCard.model_validate(_native_datetime_row(row))
            for row in rows
        ]

    async def load_review_snapshot(
        self,
        user_uuid: str,
        deck_uuid: str,
        card_uuid: str,
    ) -> ReviewSnapshot:
        """Load the user-wide scheduler and card state for one deck card."""
        row = await self._single(
            LOAD_REVIEW_SNAPSHOT,
            user_uuid=user_uuid,
            deck_uuid=deck_uuid,
            card_uuid=card_uuid,
        )
        return ReviewSnapshot.model_validate(row)

    async def record_review(
        self,
        user_uuid: str,
        deck_uuid: str,
        card_uuid: str,
        updated_card_json: str,
        due_at: datetime,
        scheduler_json: str,
        event: ReviewEvent,
    ) -> ReviewReceipt:
        """Persist one FSRS state transition and append its event."""
        row = await self._single(
            RECORD_REVIEW,
            user_uuid=user_uuid,
            deck_uuid=deck_uuid,
            card_uuid=card_uuid,
            fsrs_card_json=updated_card_json,
            due_at=due_at,
            scheduler_json=scheduler_json,
            event_uuid=event.uuid,
            fsrs_card_before_json=event.fsrs_card_before_json,
            fsrs_card_after_json=event.fsrs_card_after_json,
            fsrs_review_log_json=event.fsrs_review_log_json,
        )
        return ReviewReceipt.model_validate(_native_datetime_row(row))

    async def list_review_events(
        self,
        user_uuid: str,
        card_uuid: str,
    ) -> list[ReviewEvent]:
        """List one user's immutable card review events in order."""
        rows = await self._data(
            LIST_REVIEW_EVENTS,
            user_uuid=user_uuid,
            card_uuid=card_uuid,
        )
        return [ReviewEvent.model_validate(row) for row in rows]

    async def _data(
        self, query: str, **parameters: object
    ) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            result = await session.run(query, **parameters)
            return await result.data()

    async def _single(self, query: str, **parameters: object) -> Record:
        async with self._session_factory() as session:
            result = await session.run(query, **parameters)
            return await result.single(strict=True)

    async def _consume(self, query: str, **parameters: object) -> None:
        async with self._session_factory() as session:
            result = await session.run(query, **parameters)
            await result.consume()


def _native_datetime(value: datetime | DateTime) -> datetime:
    """Convert a Neo4j temporal value to the domain datetime type."""
    return value.to_native() if isinstance(value, DateTime) else value


def _native_datetime_row(row: Any) -> dict[str, Any]:
    """Convert Neo4j temporal fields in one result row to domain values."""
    values = dict(row)
    values['due_at'] = _native_datetime(values['due_at'])
    return values
