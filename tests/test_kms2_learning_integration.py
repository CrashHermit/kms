"""Opt-in Neo4j verification for generic source-learning artifacts."""

import asyncio
import os
import uuid
from datetime import UTC, datetime

import pytest

from kms2.composition.services import build_learning_service, build_user_service
from kms2.config.settings import Settings
from kms2.core.model.learning import ReviewRating
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.database.source_learning.repository import SourceLearningRepository

_REQUIRED_ENVIRONMENT = (
    'KMS2_DATABASE__URI',
    'KMS2_DATABASE__USERNAME',
    'KMS2_DATABASE__PASSWORD',
    'KMS2_DATABASE__DATABASE',
)


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j integration environment is not enabled',
)
def test_generic_learning_cards_are_user_scoped_and_reset_per_source():
    async def exercise() -> None:
        database = DatabaseClient(Settings().database)
        suffix = uuid.uuid4().hex
        source_a_uuid = f'kms2-learning-it-source-a-{suffix}'
        source_b_uuid = f'kms2-learning-it-source-b-{suffix}'
        card_a_uuid = f'kms2-learning-it-card-a-{suffix}'
        card_b_uuid = f'kms2-learning-it-card-b-{suffix}'
        try:
            await schema.ensure_structural_schema(database.session)
            users = build_user_service(database)
            learning = build_learning_service(database)
            user_a = await users.create_user(f'Alex {suffix}')
            user_b = await users.create_user(f'Sam {suffix}')
            async with database.session() as session:
                await session.run(
                    """
                    MATCH (user:User {uuid: $user_uuid})
                    CREATE (source:Source {uuid: $source_uuid, key: $key})
                    CREATE (user)-[:OWNS_SOURCE]->(source)
                    CREATE (source)-[:HAS_FLASHCARD]->
                           (card:SourceFlashcard {
                               uuid: $card_uuid,
                               question: $question,
                               answer: $answer
                           })
                    """,
                    user_uuid=user_a.uuid,
                    source_uuid=source_a_uuid,
                    key='alex.pdf',
                    card_uuid=card_a_uuid,
                    question='What does Alex know?',
                    answer='Alex knows the fact.',
                )
                await session.run(
                    """
                    MATCH (user:User {uuid: $user_uuid})
                    CREATE (source:Source {uuid: $source_uuid, key: $key})
                    CREATE (user)-[:OWNS_SOURCE]->(source)
                    CREATE (source)-[:HAS_FLASHCARD]->
                           (card:SourceFlashcard {
                               uuid: $card_uuid,
                               question: $question,
                               answer: $answer
                           })
                    """,
                    user_uuid=user_b.uuid,
                    source_uuid=source_b_uuid,
                    key='sam.pdf',
                    card_uuid=card_b_uuid,
                    question='What does Sam know?',
                    answer='Sam knows the fact.',
                )

            assert [
                card.card_uuid
                for card in await learning.list_assignable_cards(user_a.uuid)
            ] == [card_a_uuid]
            assert [
                card.card_uuid
                for card in await learning.list_assignable_cards(user_b.uuid)
            ] == [card_b_uuid]

            default_deck = (await learning.list_decks(user_a.uuid))[0]
            default_deck_b = (await learning.list_decks(user_b.uuid))[0]
            await learning.add_card_to_deck(
                user_a.uuid,
                default_deck.uuid,
                card_a_uuid,
            )
            await learning.add_card_to_deck(
                user_b.uuid,
                default_deck_b.uuid,
                card_b_uuid,
            )
            await learning.review_card(
                user_a.uuid,
                default_deck.uuid,
                card_a_uuid,
                ReviewRating.GOOD,
                datetime(2026, 9, 29, 12, 0, tzinfo=UTC),
            )
            await learning.review_card(
                user_b.uuid,
                default_deck_b.uuid,
                card_b_uuid,
                ReviewRating.GOOD,
                datetime(2026, 9, 29, 12, 0, tzinfo=UTC),
            )

            source_learning = SourceLearningRepository(database.session)
            await source_learning.clear_source_learning(source_a_uuid)
            assert (
                await learning.list_deck_cards(user_a.uuid, default_deck.uuid)
                == []
            )
            assert [
                card.card_uuid
                for card in await learning.list_deck_cards(
                    user_b.uuid, default_deck_b.uuid
                )
            ] == [card_b_uuid]
            assert (
                await learning.list_review_events(user_a.uuid, card_a_uuid)
                == []
            )
            assert (
                len(await learning.list_review_events(user_b.uuid, card_b_uuid))
                == 1
            )
        finally:
            async with database.session() as session:
                await session.run(
                    """
                    MATCH (source:Source)
                    WHERE source.uuid IN $source_uuids
                    DETACH DELETE source
                    """,
                    source_uuids=[source_a_uuid, source_b_uuid],
                )
                await session.run(
                    """
                    MATCH (user:User)
                    WHERE user.uuid IN $user_uuids
                    DETACH DELETE user
                    """,
                    user_uuids=[user_a.uuid, user_b.uuid]
                    if 'user_a' in locals()
                    else [],
                )
            await database.close()

    asyncio.run(exercise())
