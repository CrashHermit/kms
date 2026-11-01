"""Opt-in Neo4j verification for generic source-learning artifacts."""

import asyncio
import os
import uuid
from datetime import UTC, datetime

import pytest

from kms2.composition.services import build_learning_service, build_user_service
from kms2.config.inference import ContextWindowSettings
from kms2.config.settings import Settings
from kms2.core.model.block_types import BlockType
from kms2.core.model.learning import ReviewRating
from kms2.core.model.source import Source
from kms2.core.model.source_learning.flashcard import (
    SourceFlashcard,
    SourceFlashcardOccurrence,
)
from kms2.core.windowing import TokenBudget
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.database.source.source_block_repository import SourceBlockRepository
from kms2.database.source.source_entity_repository import (
    SourceEntityRepository,
)
from kms2.database.source.source_graph_repository import SourceGraphRepository
from kms2.database.source_learning.repository import SourceLearningRepository
from kms2.langgraph.source_learning.state import SourceLearningState
from kms2.node.source_learning.atomic import SourceAtomicFlashcardNode

_REQUIRED_ENVIRONMENT = (
    'KMS2_DATABASE__URI',
    'KMS2_DATABASE__USERNAME',
    'KMS2_DATABASE__PASSWORD',
    'KMS2_DATABASE__DATABASE',
)


class _UnitCounter:
    def count_texts(self, texts):
        return [1 for _ in texts]


def _atomic_loader(database, repository, context_window):
    return SourceAtomicFlashcardNode(
        repository,
        None,
        context_window,
        SourceBlockRepository(database.session),
        budget=TokenBudget(_UnitCounter(), 1000),
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

            cards_before = await learning.list_deck_cards(
                user_a.uuid, default_deck.uuid
            )
            reviews_before = await learning.list_review_events(
                user_a.uuid, card_a_uuid
            )
            entities = SourceEntityRepository(database.session)
            await entities.replace_source_entities(source_a_uuid, [])
            await entities.replace_source_entity_hubs(source_a_uuid, [], [])
            assert (
                await learning.list_deck_cards(user_a.uuid, default_deck.uuid)
                == cards_before
            )
            assert (
                await learning.list_review_events(user_a.uuid, card_a_uuid)
                == reviews_before
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
            if 'user_a' in locals():
                source_learning = SourceLearningRepository(database.session)
                await source_learning.clear_source_learning(source_a_uuid)
                await source_learning.clear_source_learning(source_b_uuid)
                async with database.session() as session:
                    await session.run(
                        """
                        MATCH (user:User)
                        WHERE user.uuid IN $user_uuids
                        OPTIONAL MATCH (user)-[:HAS_DECK]->(deck:Deck)
                        OPTIONAL MATCH (user)-[:HAS_SCHEDULER_SETTINGS]->
                                       (settings:UserSchedulerSettings)
                        WITH collect(DISTINCT user) AS users,
                             collect(DISTINCT deck) AS decks,
                             collect(DISTINCT settings) AS settings
                        FOREACH (deck IN decks | DETACH DELETE deck)
                        FOREACH (setting IN settings | DETACH DELETE setting)
                        FOREACH (user IN users | DETACH DELETE user)
                        """,
                        user_uuids=[user_a.uuid, user_b.uuid],
                    )
                    await session.run(
                        """
                        MATCH (source:Source)
                        WHERE source.uuid IN $source_uuids
                        DETACH DELETE source
                        """,
                        source_uuids=[source_a_uuid, source_b_uuid],
                    )
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j integration environment is not enabled',
)
def test_block_learning_round_trips_evidence_without_entity_links():
    async def exercise() -> None:
        database = DatabaseClient(Settings().database)
        suffix = uuid.uuid4().hex
        source_uuid = f'kms2-block-learning-source-{suffix}'
        block_uuids = [
            f'kms2-block-learning-block-{suffix}-{position}'
            for position in range(3)
        ]
        node_uuids = [source_uuid, *block_uuids]
        repository = SourceLearningRepository(database.session)
        atomic = SourceFlashcard(
            uuid=f'kms2-block-learning-atomic-{suffix}',
            question='What property does the passage establish?',
            answer='Every stated condition is required.',
        )
        coherent = SourceFlashcard(
            uuid=f'kms2-block-learning-coherent-{suffix}',
            question='Which condition bounds the established property?',
            answer='The source states the required condition.',
        )
        try:
            async with database.session() as session:
                await session.run(
                    """
                    CREATE (source:Source {uuid: $source_uuid})
                    CREATE (page:SourcePage {index: 0})
                    CREATE (first:SourceBlock {
                        uuid: $before_uuid,
                        block_type: $paragraph_type,
                        content: 'A named object is introduced.'
                    })
                    CREATE (target:SourceBlock {
                        uuid: $target_uuid,
                        block_type: $paragraph_type,
                        content: 'The object satisfies the stated condition.'
                    })
                    CREATE (after:SourceBlock {
                        uuid: $after_uuid,
                        block_type: $paragraph_type,
                        content: 'This sentence explains the reference.'
                    })
                    CREATE (source)-[:HAS_PAGE]->(page)
                    CREATE (page)-[:CONTAINS_BLOCK]->(first)
                    CREATE (page)-[:CONTAINS_BLOCK]->(target)
                    CREATE (page)-[:CONTAINS_BLOCK]->(after)
                    CREATE (source)-[:FIRST_BLOCK]->(first)
                    CREATE (first)-[:NEXT_BLOCK]->(target)
                    CREATE (target)-[:NEXT_BLOCK]->(after)
                    """,
                    source_uuid=source_uuid,
                    paragraph_type=BlockType.PARAGRAPH.value,
                    before_uuid=block_uuids[0],
                    target_uuid=block_uuids[1],
                    after_uuid=block_uuids[2],
                )
            requests = await repository.load_atomic_flashcard_requests(
                source_uuid
            )
            assert [request.source_block_uuid for request in requests] == [
                block_uuids[0],
                block_uuids[1],
                block_uuids[2],
            ]
            target_request = requests[1]
            assert target_request.input.evidence.source_text == (
                'The object satisfies the stated condition.'
            )
            assert (
                target_request.input.evidence.source_context.context_before
                == []
            )
            assert (
                target_request.input.evidence.source_context.context_after == []
            )
            loaded = await _atomic_loader(
                database,
                repository,
                ContextWindowSettings(backward_budget=1, forward_budget=1),
            ).load(SourceLearningState(source_uuid=source_uuid))
            loaded_request = next(
                request
                for request in loaded['atomic_requests']
                if request.source_block_uuid == block_uuids[1]
            )
            evidence = loaded_request.input.evidence.source_context
            assert [block.content for block in evidence.context_before] == [
                'A named object is introduced.'
            ]
            assert [block.content for block in evidence.target_blocks] == [
                'The object satisfies the stated condition.'
            ]
            assert [block.content for block in evidence.context_after] == [
                'This sentence explains the reference.'
            ]
            assert all(
                not hasattr(block, 'uuid')
                for block in [
                    *evidence.context_before,
                    *evidence.target_blocks,
                    *evidence.context_after,
                ]
            )

            await repository.persist_flashcards(
                source_uuid,
                [
                    SourceFlashcardOccurrence(
                        card=atomic,
                        source_block_uuids=[block_uuids[1]],
                    )
                ],
            )
            await repository.persist_flashcards(
                source_uuid,
                [
                    SourceFlashcardOccurrence(
                        card=coherent,
                        source_block_uuids=[block_uuids[1]],
                        derived_card_uuids=[atomic.uuid],
                    )
                ],
            )
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                          -[:HAS_FLASHCARD]->(card:SourceFlashcard)
                    OPTIONAL MATCH (card)-[about:ABOUT_HUB]->()
                    OPTIONAL MATCH (card)-[:SUPPORTED_BY]->(
                        block:SourceBlock
                    )
                    OPTIONAL MATCH (card)-[:DERIVED_FROM]->(
                        parent:SourceFlashcard
                    )
                    RETURN card.uuid AS card_uuid,
                           count(DISTINCT about) AS about_count,
                           collect(DISTINCT block.uuid)
                               AS source_block_uuids,
                           collect(DISTINCT parent.uuid) AS parent_uuids
                    ORDER BY card_uuid
                    """,
                    source_uuid=source_uuid,
                )
                rows = await result.data()
            assert len(rows) == 2
            assert all(row['about_count'] == 0 for row in rows)
            assert all(
                row['source_block_uuids'] == [block_uuids[1]] for row in rows
            )
            coherent_row = next(
                row for row in rows if row['card_uuid'] == coherent.uuid
            )
            assert coherent_row['parent_uuids'] == [atomic.uuid]
        finally:
            await repository.clear_source_learning(source_uuid)
            async with database.session() as session:
                await session.run(
                    """
                    MATCH (node)
                    WHERE node.uuid IN $node_uuids
                    DETACH DELETE node
                    """,
                    node_uuids=node_uuids,
                )
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j integration environment is not enabled',
)
def test_learning_builds_independent_cross_page_windows_without_stored_context():
    async def exercise() -> None:
        database = DatabaseClient(Settings().database)
        suffix = uuid.uuid4().hex
        source_uuid = f'kms2-block-window-source-{suffix}'
        block_uuids = [
            f'kms2-block-window-block-{suffix}-{index}' for index in range(20)
        ]
        rows = [
            {
                'uuid': block_uuid,
                'page_index': index // 10,
                'content': f'Window evidence block {index}.',
            }
            for index, block_uuid in enumerate(block_uuids)
        ]
        repository = SourceLearningRepository(database.session)
        try:
            async with database.session() as session:
                await session.run(
                    """
                    CREATE (source:Source {uuid: $source_uuid})
                    WITH source
                    UNWIND [0, 1] AS page_index
                    CREATE (page:SourcePage {index: page_index})
                    CREATE (source)-[:HAS_PAGE]->(page)
                    WITH DISTINCT source
                    UNWIND $blocks AS row
                    MATCH (source)-[:HAS_PAGE]->(page:SourcePage {
                        index: row.page_index
                    })
                    CREATE (page)-[:CONTAINS_BLOCK]->(
                        block:SourceBlock {
                            uuid: row.uuid,
                            block_type: $paragraph_type,
                            content: row.content
                        }
                    )
                    WITH DISTINCT source
                    UNWIND range(0, 18) AS index
                    MATCH (left:SourceBlock {uuid: $block_uuids[index]})
                    MATCH (right:SourceBlock {uuid: $block_uuids[index + 1]})
                    CREATE (left)-[:NEXT_BLOCK]->(right)
                    WITH DISTINCT source
                    MATCH (first:SourceBlock {uuid: $block_uuids[0]})
                    CREATE (source)-[:FIRST_BLOCK]->(first)
                    """,
                    source_uuid=source_uuid,
                    blocks=rows,
                    block_uuids=block_uuids,
                    paragraph_type=BlockType.PARAGRAPH.value,
                )
            requests = await repository.load_atomic_flashcard_requests(
                source_uuid
            )
            assert [request.source_block_uuid for request in requests] == (
                block_uuids
            )
            assert all(
                not request.input.evidence.source_context.context_before
                and not request.input.evidence.source_context.context_after
                for request in requests
            )
            loaded = await _atomic_loader(
                database,
                repository,
                ContextWindowSettings(backward_budget=3, forward_budget=3),
            ).load(SourceLearningState(source_uuid=source_uuid))
            requests = loaded['atomic_requests']
            for request_index in (0, 10, 19):
                request = requests[request_index]
                evidence = request.input.evidence.source_context
                assert [block.content for block in evidence.target_blocks] == [
                    f'Window evidence block {request_index}.'
                ]
                assert [block.content for block in evidence.context_before] == [
                    f'Window evidence block {index}.'
                    for index in range(max(0, request_index - 3), request_index)
                ]
                assert [block.content for block in evidence.context_after] == [
                    f'Window evidence block {index}.'
                    for index in range(
                        request_index + 1, min(20, request_index + 4)
                    )
                ]
                assert all(
                    not hasattr(block, 'uuid')
                    for block in [
                        *evidence.context_before,
                        *evidence.target_blocks,
                        *evidence.context_after,
                    ]
                )
        finally:
            await SourceGraphRepository(database.session).replace_source(
                Source(uuid=source_uuid, key='window-fixture.pdf'),
                [],
            )
            async with database.session() as session:
                await session.run(
                    'MATCH (source:Source {uuid: $uuid}) DETACH DELETE source',
                    uuid=source_uuid,
                )
            await database.close()

    asyncio.run(exercise())
