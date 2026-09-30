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
def test_learning_is_user_scoped_and_source_rebuild_resets_history():
    async def exercise() -> None:
        database = DatabaseClient(Settings().database)
        suffix = uuid.uuid4().hex
        source_a_uuid = f'kms2-learning-it-source-a-{suffix}'
        source_b_uuid = f'kms2-learning-it-source-b-{suffix}'
        card_a_uuid = f'kms2-learning-it-card-a-{suffix}'
        card_b_uuid = f'kms2-learning-it-card-b-{suffix}'
        fact_a_uuid = f'kms2-learning-it-fact-a-{suffix}'
        fact_b_uuid = f'kms2-learning-it-fact-b-{suffix}'
        user_a = None
        user_b = None
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
                    CREATE (source)-[:HAS_LEARNING_FACT]->
                           (fact:SourceEntityLearningFact {
                               uuid: $fact_uuid, text: $fact_text
                           })
                    CREATE (source)-[:HAS_FLASHCARD]->
                           (card:SourceFlashcard {
                               uuid: $card_uuid,
                               question: $question,
                               answer: $answer
                           })
                    CREATE (card)-[:DERIVED_FROM]->(fact)
                    """,
                    user_uuid=user_a.uuid,
                    source_uuid=source_a_uuid,
                    fact_uuid=fact_a_uuid,
                    card_uuid=card_a_uuid,
                    key='alex.pdf',
                    fact_text='Alex fact',
                    question='What does Alex know?',
                    answer='Alex knows the fact.',
                )
                await session.run(
                    """
                    MATCH (user:User {uuid: $user_uuid})
                    CREATE (source:Source {uuid: $source_uuid, key: $key})
                    CREATE (user)-[:OWNS_SOURCE]->(source)
                    CREATE (source)-[:HAS_LEARNING_FACT]->
                           (fact:SourceEntityLearningFact {
                               uuid: $fact_uuid, text: $fact_text
                           })
                    CREATE (source)-[:HAS_FLASHCARD]->
                           (card:SourceFlashcard {
                               uuid: $card_uuid,
                               question: $question,
                               answer: $answer
                           })
                    CREATE (card)-[:DERIVED_FROM]->(fact)
                    """,
                    user_uuid=user_b.uuid,
                    source_uuid=source_b_uuid,
                    fact_uuid=fact_b_uuid,
                    card_uuid=card_b_uuid,
                    key='sam.pdf',
                    fact_text='Sam fact',
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
            second_deck = await learning.create_deck(user_a.uuid, 'Second')
            await learning.add_card_to_deck(
                user_a.uuid,
                default_deck.uuid,
                card_a_uuid,
            )
            await learning.add_card_to_deck(
                user_a.uuid,
                second_deck.uuid,
                card_a_uuid,
            )
            await learning.review_card(
                user_a.uuid,
                default_deck.uuid,
                card_a_uuid,
                ReviewRating.GOOD,
                datetime.now(UTC),
            )

            events = await learning.list_review_events(user_a.uuid, card_a_uuid)
            assert [event.review_index for event in events] == [1]
            future = datetime(2200, 1, 1, tzinfo=UTC)
            assert [
                card.card_uuid
                for card in await learning.list_due_deck_cards(
                    user_a.uuid,
                    default_deck.uuid,
                    future,
                )
            ] == [card_a_uuid]
            assert [
                card.card_uuid
                for card in await learning.list_due_deck_cards(
                    user_a.uuid,
                    second_deck.uuid,
                    future,
                )
            ] == [card_a_uuid]

            await SourceLearningRepository(
                database.session
            ).clear_source_learning(source_a_uuid)
            assert (
                await learning.list_deck_cards(user_a.uuid, default_deck.uuid)
                == []
            )
            assert (
                await learning.list_deck_cards(user_a.uuid, second_deck.uuid)
                == []
            )
            assert (
                await learning.list_review_events(user_a.uuid, card_a_uuid)
                == []
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
                    OPTIONAL MATCH (user)-[:HAS_DECK]->(deck:Deck)
                    OPTIONAL MATCH (user)-[:HAS_SCHEDULER_SETTINGS]->
                        (settings:UserSchedulerSettings)
                    WITH collect(user) + collect(deck) + collect(settings)
                         AS nodes
                    UNWIND nodes AS node
                    WITH node
                    WHERE node IS NOT NULL
                    DETACH DELETE node
                    """,
                    user_uuids=[
                        user.uuid
                        for user in (user_a, user_b)
                        if user is not None
                    ],
                )
            await database.close()

    asyncio.run(exercise())
