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
def test_proposition_learning_round_trips_evidence_without_entity_links():
    async def exercise() -> None:
        database = DatabaseClient(Settings().database)
        suffix = uuid.uuid4().hex
        source_uuid = f'kms2-proposition-learning-source-{suffix}'
        block_uuids = [
            f'kms2-proposition-learning-block-{suffix}-{position}'
            for position in range(3)
        ]
        proposition_uuid = f'kms2-proposition-learning-proposition-{suffix}'
        hub_uuid = f'kms2-proposition-learning-hub-{suffix}'
        node_uuids = [
            source_uuid,
            *block_uuids,
            proposition_uuid,
            hub_uuid,
        ]
        repository = SourceLearningRepository(database.session)
        atomic = SourceFlashcard(
            uuid=f'kms2-proposition-learning-atomic-{suffix}',
            question='What property does the passage establish?',
            answer='Every stated condition is required.',
        )
        coherent = SourceFlashcard(
            uuid=f'kms2-proposition-learning-coherent-{suffix}',
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
                    CREATE (target)-[:MEMBER_OF]->(
                        proposition:SourceProposition {
                            uuid: $proposition_uuid,
                            text: 'The object satisfies the stated condition.'
                        }
                    )
                    CREATE (proposition)-[:IN_SOURCE_HUB]->(
                        hub:SourcePropositionHub {uuid: $hub_uuid}
                    )
                    """,
                    source_uuid=source_uuid,
                    paragraph_type=BlockType.PARAGRAPH.value,
                    before_uuid=block_uuids[0],
                    target_uuid=block_uuids[1],
                    after_uuid=block_uuids[2],
                    proposition_uuid=proposition_uuid,
                    hub_uuid=hub_uuid,
                )

            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (hub:SourcePropositionHub {uuid: $hub_uuid})
                    RETURN keys(hub) AS keys
                    """,
                    hub_uuid=hub_uuid,
                )
                hub = await result.single(strict=True)
            assert hub['keys'] == ['uuid']
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                    OPTIONAL MATCH (source)-[:HAS_PROPOSITION]->(direct)
                    WITH source, count(DISTINCT direct) AS direct_ownership
                    MATCH (source)-[:HAS_PAGE]->(:SourcePage)
                        -[:CONTAINS_BLOCK]->(target:SourceBlock)
                        -[:MEMBER_OF]->(proposition:SourceProposition {
                            uuid: $proposition_uuid
                        })
                    RETURN keys(proposition) AS proposition_keys,
                           direct_ownership,
                           target.uuid AS owning_block_uuid
                    """,
                    source_uuid=source_uuid,
                    proposition_uuid=proposition_uuid,
                )
                occurrence_graph = await result.single(strict=True)
            assert set(occurrence_graph['proposition_keys']) == {
                'uuid',
                'text',
            }
            assert 'ordinal' not in occurrence_graph['proposition_keys']
            assert 'source_uuid' not in occurrence_graph['proposition_keys']
            assert occurrence_graph['direct_ownership'] == 0
            assert occurrence_graph['owning_block_uuid'] == block_uuids[1]
            requests = await repository.load_atomic_flashcard_requests(
                source_uuid
            )
            assert len(requests) == 1
            request = requests[0]
            assert request.source_block_uuid == block_uuids[1]
            assert request.input.evidence.source_context.context_before == []
            assert request.input.evidence.source_context.context_after == []
            loaded = await _atomic_loader(
                database,
                repository,
                ContextWindowSettings(backward_budget=1, forward_budget=1),
            ).load(SourceLearningState(source_uuid=source_uuid))
            request = loaded['atomic_requests'][0]
            assert request.source_proposition_uuid == proposition_uuid
            assert request.input.context.proposition_statement == (
                'The object satisfies the stated condition.'
            )
            assert request.input.evidence.proposition_text == (
                'The object satisfies the stated condition.'
            )
            evidence = request.input.evidence.source_context
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
                        source_proposition_uuids=[proposition_uuid],
                    )
                ],
            )
            await repository.persist_flashcards(
                source_uuid,
                [
                    SourceFlashcardOccurrence(
                        card=coherent,
                        source_proposition_uuids=[proposition_uuid],
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
                        proposition:SourceProposition
                    )
                    OPTIONAL MATCH (card)-[:DERIVED_FROM]->(
                        parent:SourceFlashcard
                    )
                    RETURN card.uuid AS card_uuid,
                           count(DISTINCT about) AS about_count,
                           collect(DISTINCT proposition.uuid)
                               AS proposition_uuids,
                           collect(DISTINCT parent.uuid) AS parent_uuids
                    ORDER BY card_uuid
                    """,
                    source_uuid=source_uuid,
                )
                rows = await result.data()
            assert len(rows) == 2
            assert all(row['about_count'] == 0 for row in rows)
            assert all(
                row['proposition_uuids'] == [proposition_uuid] for row in rows
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
        source_uuid = f'kms2-proposition-window-source-{suffix}'
        hub_uuid = f'kms2-proposition-window-hub-{suffix}'
        block_uuids = [
            f'kms2-proposition-window-block-{suffix}-{index}'
            for index in range(20)
        ]
        proposition_uuids = [
            f'kms2-proposition-window-proposition-{suffix}-z',
            f'kms2-proposition-window-proposition-{suffix}-m',
            f'kms2-proposition-window-proposition-{suffix}-a',
            f'kms2-proposition-window-proposition-{suffix}-y',
        ]
        windows = [(0, 0, 3), (10, 7, 13), (10, 7, 13), (19, 16, 19)]
        rows = [
            {
                'uuid': uuid,
                'index': index,
                'page_index': index // 10,
                'content': f'Window evidence block {index}.',
            }
            for index, uuid in enumerate(block_uuids)
        ]
        proposition_rows = [
            {
                'uuid': proposition_uuid,
                'text': f'Occurrence statement {sequence_index}.',
                'owner': block_uuids[owner],
            }
            for sequence_index, (
                proposition_uuid,
                (owner, start, end),
            ) in enumerate(zip(proposition_uuids, windows, strict=True))
        ]
        repository = SourceLearningRepository(database.session)
        try:
            async with database.session() as session:
                await session.run(
                    """
                    CREATE (source:Source {uuid: $source_uuid})
                    CREATE (hub:SourcePropositionHub {uuid: $hub_uuid})
                    WITH source, hub
                    UNWIND [0, 1] AS page_index
                    CREATE (page:SourcePage {index: page_index})
                    CREATE (source)-[:HAS_PAGE]->(page)
                    WITH DISTINCT source, hub
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
                    WITH DISTINCT source, hub
                    UNWIND range(0, 18) AS index
                    MATCH (left:SourceBlock {uuid: $block_uuids[index]})
                    MATCH (right:SourceBlock {uuid: $block_uuids[index + 1]})
                    CREATE (left)-[:NEXT_BLOCK]->(right)
                    WITH DISTINCT source, hub
                    MATCH (first:SourceBlock {uuid: $block_uuids[0]})
                    CREATE (source)-[:FIRST_BLOCK]->(first)
                    WITH source, hub
                    UNWIND $propositions AS row
                    MATCH (source)-[:HAS_PAGE]->(:SourcePage)
                        -[:CONTAINS_BLOCK]->(owner:SourceBlock {uuid: row.owner})
                    CREATE (owner)-[:MEMBER_OF]->(
                        proposition:SourceProposition {
                            uuid: row.uuid,
                            text: row.text
                        }
                    )
                    CREATE (proposition)-[:IN_SOURCE_HUB]->(hub)
                    """,
                    source_uuid=source_uuid,
                    hub_uuid=hub_uuid,
                    blocks=rows,
                    block_uuids=block_uuids,
                    propositions=proposition_rows,
                    paragraph_type=BlockType.PARAGRAPH.value,
                )

            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
                        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->
                        (proposition:SourceProposition)
                    OPTIONAL MATCH (proposition)-[reference]->()
                    WITH count(DISTINCT proposition) AS occurrence_count,
                         collect(DISTINCT type(reference)) AS reference_types
                    RETURN occurrence_count, reference_types
                    """,
                    source_uuid=source_uuid,
                )
                counts = await result.single(strict=True)
            assert counts['reference_types'] == ['IN_SOURCE_HUB']
            assert counts['occurrence_count'] == 4

            requests = await repository.load_atomic_flashcard_requests(
                source_uuid
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
            ordered_indices = (0, 2, 1, 3)
            assert [
                request.source_proposition_uuid for request in requests
            ] == [proposition_uuids[index] for index in ordered_indices]
            for sequence_index, request_index in enumerate(ordered_indices):
                owner, start, end = windows[request_index]
                request = requests[sequence_index]
                evidence = request.input.evidence.source_context
                assert request.input.context.proposition_statement == (
                    'Occurrence statement 0.'
                )
                assert request.input.evidence.proposition_text == (
                    f'Occurrence statement {request_index}.'
                )
                assert [block.content for block in evidence.target_blocks] == [
                    f'Window evidence block {owner}.'
                ]
                assert [block.content for block in evidence.context_before] == [
                    f'Window evidence block {index}.'
                    for index in range(start, owner)
                ]
                assert [block.content for block in evidence.context_after] == [
                    f'Window evidence block {index}.'
                    for index in range(owner + 1, end + 1)
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
                await session.run(
                    'MATCH (hub:SourcePropositionHub {uuid: $uuid}) DETACH DELETE hub',
                    uuid=hub_uuid,
                )
            await database.close()

    asyncio.run(exercise())
