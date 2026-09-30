import asyncio
import os
import uuid
from collections import Counter
from datetime import UTC, datetime

import pytest

from kms2.composition.services import build_learning_service, build_user_service
from kms2.config.settings import Settings
from kms2.core.model.learning import ReviewRating
from kms2.core.model.source import Source
from kms2.core.model.source_learning.entity import (
    SourceEntityFlashcardResult,
    SourceEntityLearningFactResult,
)
from kms2.core.model.source_learning.event import (
    SourceEventFlashcardResult,
    SourceEventLearningFactResult,
)
from kms2.core.model.source_learning.predicate import (
    SourcePredicateFlashcardResult,
    SourcePredicateLearningFactResult,
)
from kms2.core.model.source_learning.triplet import (
    SourceTripletFlashcardResult,
    SourceTripletLearningFactResult,
)
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.database.source.source_graph_repository import SourceGraphRepository
from kms2.database.source_learning.repository import SourceLearningRepository
from kms2.database.source_semantic.source_entity_repository import (
    SourceEntityRepository,
)
from kms2.database.source_semantic.source_event_repository import (
    SourceEventRepository,
)
from kms2.database.source_semantic.source_fact_repository import (
    SourceFactRepository,
)
from kms2.database.source_semantic.source_predicate_repository import (
    SourcePredicateRepository,
)
from kms2.database.source_semantic.source_triplet_repository import (
    SourceTripletRepository,
)

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
        source_learning = SourceLearningRepository(database.session)
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
            default_deck_b = (await learning.list_decks(user_b.uuid))[0]
            second_deck = await learning.create_deck(user_a.uuid, 'Second')
            review_instant = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
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
                review_instant,
            )
            await learning.add_card_to_deck(
                user_b.uuid,
                default_deck_b.uuid,
                card_b_uuid,
            )
            await learning.review_card(
                user_b.uuid,
                default_deck_b.uuid,
                card_b_uuid,
                ReviewRating.GOOD,
                review_instant,
            )
            b_deck_cards_before = await learning.list_deck_cards(
                user_b.uuid, default_deck_b.uuid
            )
            b_events_before = await learning.list_review_events(
                user_b.uuid, card_b_uuid
            )
            assert [card.card_uuid for card in b_deck_cards_before] == [
                card_b_uuid
            ]
            assert [event.review_index for event in b_events_before] == [1]

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

            await source_learning.clear_source_learning(source_a_uuid)
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
            assert (
                await learning.list_deck_cards(user_b.uuid, default_deck_b.uuid)
                == b_deck_cards_before
            )
            assert (
                await learning.list_review_events(user_b.uuid, card_b_uuid)
                == b_events_before
            )
        finally:
            await source_learning.clear_source_learning(source_a_uuid)
            await source_learning.clear_source_learning(source_b_uuid)
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


_SOURCE_LEARNING_FIXTURE_SHAPES = {
    'entity': (
        'SourceEntity',
        'SourceEntityHub',
        'SourceEntity',
        'canonical_name',
        'SourceEntityLearningFact',
    ),
    'event': (
        'SourceEvent',
        'SourceEventHub',
        'SourceEvent',
        'name',
        'SourceEventLearningFact',
    ),
    'predicate': (
        'SourceEntity',
        'SourcePredicateHub',
        'SourcePredicate',
        'predicate',
        'SourcePredicateLearningFact',
    ),
    'triplet': (
        'SourceEntity',
        'SourceTripletHub',
        'SourceTriplet',
        'canonical_name',
        'SourceTripletLearningFact',
    ),
}

_SOURCE_LEARNING_RESULT_MODELS = {
    'entity': (SourceEntityLearningFactResult, SourceEntityFlashcardResult),
    'event': (SourceEventLearningFactResult, SourceEventFlashcardResult),
    'predicate': (
        SourcePredicateLearningFactResult,
        SourcePredicateFlashcardResult,
    ),
    'triplet': (SourceTripletLearningFactResult, SourceTripletFlashcardResult),
}


async def _seed_source_learning_fixture(
    session,
    source_uuid: str,
    family: str,
    owner_uuid: str | None = None,
) -> dict[str, object]:
    """Create ordered source evidence and one typed source-local hub."""
    endpoint_label, hub_label, member_label, name_property, _ = (
        _SOURCE_LEARNING_FIXTURE_SHAPES[family]
    )
    ids = {
        'source_uuid': source_uuid,
        'page_uuid': f'{source_uuid}-page',
        'block_1': f'{source_uuid}-block-1',
        'block_2': f'{source_uuid}-block-2',
        'fact_1': f'{source_uuid}-fact-1',
        'fact_2': f'{source_uuid}-fact-2',
        'target_1': f'{source_uuid}-target-1',
        'target_2': f'{source_uuid}-target-2',
        'triplet_1': f'{source_uuid}-triplet-1',
        'triplet_2': f'{source_uuid}-triplet-2',
        'triplet_3': f'{source_uuid}-triplet-3',
        'subject_1': f'{source_uuid}-subject-1',
        'subject_2': f'{source_uuid}-subject-2',
        'subject_3': f'{source_uuid}-subject-3',
        'object_1': f'{source_uuid}-object-1',
        'object_2': f'{source_uuid}-object-2',
        'object_3': f'{source_uuid}-object-3',
        'predicate_1': f'{source_uuid}-predicate-1',
        'predicate_2': f'{source_uuid}-predicate-2',
        'predicate_3': f'{source_uuid}-predicate-3',
        'hub_uuid': f'{source_uuid}-hub',
    }
    ids['all_uuids'] = list(ids.values())

    result = await session.run(
        """
        CREATE (source:Source {
            uuid: $source_uuid,
            key: $source_uuid,
            metadata: '{}'
        })
        CREATE (page:SourcePage {
            uuid: $page_uuid, index: 0, markdown: 'Source evidence'
        })
        CREATE (first_block:SourceBlock {
            uuid: $block_1, block_type: 'markdown',
            content: 'First evidence block'
        })
        CREATE (second_block:SourceBlock {
            uuid: $block_2, block_type: 'markdown',
            content: 'Second evidence block'
        })
        CREATE (source)-[:HAS_PAGE]->(page)
        CREATE (source)-[:FIRST_PAGE]->(page)
        CREATE (source)-[:FIRST_BLOCK]->(first_block)
        CREATE (source)-[:LAST_BLOCK]->(second_block)
        CREATE (page)-[:FIRST_BLOCK]->(first_block)
        CREATE (page)-[:LAST_BLOCK]->(second_block)
        CREATE (page)-[:CONTAINS_BLOCK]->(first_block)
        CREATE (page)-[:CONTAINS_BLOCK]->(second_block)
        CREATE (first_block)-[:NEXT_BLOCK]->(second_block)
        """,
        source_uuid=source_uuid,
        page_uuid=ids['page_uuid'],
        block_1=ids['block_1'],
        block_2=ids['block_2'],
    )
    await result.consume()
    if owner_uuid is not None:
        result = await session.run(
            """
            MATCH (user:User {uuid: $user_uuid})
            MATCH (source:Source {uuid: $source_uuid})
            CREATE (user)-[:OWNS_SOURCE]->(source)
            """,
            user_uuid=owner_uuid,
            source_uuid=source_uuid,
        )
        await result.consume()

    facts = [
        {
            'uuid': ids['fact_1'],
            'text': 'Fact one appears in both source blocks.',
            'target_uuid': ids['target_1'],
            'block_uuids': [ids['block_1'], ids['block_2']],
        },
        {
            'uuid': ids['fact_2'],
            'text': 'Fact two appears in the second source block.',
            'target_uuid': ids['target_2'],
            'block_uuids': [ids['block_2']],
        },
    ]
    result = await session.run(
        """
        UNWIND $facts AS row
        CREATE (fact:SourceFact {uuid: row.uuid, text: row.text})
        CREATE (target:SourceFactTarget {uuid: row.target_uuid})
        CREATE (fact)-[:HAS_TARGET]->(target)
        WITH target, row
        UNWIND row.block_uuids AS block_uuid
        MATCH (source_block:SourceBlock {uuid: block_uuid})
        CREATE (target)-[:HAS_SOURCE_BLOCK]->(source_block)
        """,
        facts=facts,
    )
    await result.consume()

    triplets = [
        {
            'uuid': ids['triplet_1'],
            'fact_uuid': ids['fact_1'],
            'block_uuid': ids['block_1'],
            'subject_uuid': ids['subject_1'],
            'subject_name': 'Subject one',
            'predicate_uuid': ids['predicate_1'],
            'predicate': 'relates to',
            'object_uuid': ids['object_1'],
            'object_name': 'Object one',
        },
        {
            'uuid': ids['triplet_2'],
            'fact_uuid': ids['fact_2'],
            'block_uuid': ids['block_2'],
            'subject_uuid': ids['subject_2'],
            'subject_name': 'Subject two',
            'predicate_uuid': ids['predicate_2'],
            'predicate': 'causes',
            'object_uuid': ids['object_2'],
            'object_name': 'Object two',
        },
        {
            'uuid': ids['triplet_3'],
            'fact_uuid': ids['fact_1'],
            'block_uuid': ids['block_2'],
            'subject_uuid': ids['subject_3'],
            'subject_name': 'Subject three',
            'predicate_uuid': ids['predicate_3'],
            'predicate': 'mentions',
            'object_uuid': ids['object_3'],
            'object_name': 'Object three',
        },
    ]
    result = await session.run(
        f"""
        UNWIND $triplets AS row
        MATCH (fact:SourceFact {{uuid: row.fact_uuid}})
        CREATE (triplet:SourceTriplet {{uuid: row.uuid}})
        CREATE (subject:{endpoint_label} {{
            uuid: row.subject_uuid,
            source_uuid: $source_uuid,
            source_block_uuid: row.block_uuid,
            name: row.subject_name
        }})
        CREATE (predicate:SourcePredicate {{
            uuid: row.predicate_uuid,
            source_uuid: $source_uuid,
            source_block_uuid: row.block_uuid,
            predicate: row.predicate
        }})
        CREATE (object:{endpoint_label} {{
            uuid: row.object_uuid,
            source_uuid: $source_uuid,
            source_block_uuid: row.block_uuid,
            name: row.object_name
        }})
        CREATE (fact)-[:HAS_TRIPLET]->(triplet)
        CREATE (triplet)-[:HAS_SUBJECT]->(subject)
        CREATE (triplet)-[:HAS_PREDICATE]->(predicate)
        CREATE (triplet)-[:HAS_OBJECT]->(object)
        """,
        source_uuid=source_uuid,
        triplets=triplets,
    )
    await result.consume()

    if family in ('entity', 'event'):
        member_uuids = [
            ids[f'{role}_{index}']
            for index in range(1, 4)
            for role in ('subject', 'object')
        ]
    elif family == 'predicate':
        member_uuids = [ids[f'predicate_{index}'] for index in range(1, 4)]
    else:
        member_uuids = [ids[f'triplet_{index}'] for index in range(1, 4)]

    result = await session.run(
        f"""
        CREATE (hub:{hub_label} {{
            uuid: $hub_uuid,
            source_uuid: $source_uuid,
            {name_property}: $hub_name,
            description: $hub_description
        }})
        WITH hub
        UNWIND $member_uuids AS member_uuid
        MATCH (member:{member_label} {{uuid: member_uuid}})
        CREATE (member)-[:IN_SOURCE_HUB]->(hub)
        """,
        hub_uuid=ids['hub_uuid'],
        source_uuid=source_uuid,
        hub_name=f'Shared {family} hub',
        hub_description=f'A non-empty {family} hub description.',
        member_uuids=member_uuids,
    )
    await result.consume()
    return ids


async def _persist_source_learning_card(
    repository: SourceLearningRepository,
    family: str,
    ids: dict[str, object],
) -> None:
    """Persist one selected learning fact and its derived flashcard."""
    inputs = await getattr(repository, f'load_{family}_learning_inputs')(
        ids['source_uuid']
    )
    fact_result_model, card_result_model = _SOURCE_LEARNING_RESULT_MODELS[
        family
    ]
    fact_result = fact_result_model(
        facts=[
            {
                'text': f'Selected {family} learning fact.',
                'source_fact_uuids': [ids['fact_1'], ids['fact_2']],
                'triplet_uuids': [ids['triplet_1'], ids['triplet_2']],
            }
        ]
    )
    persisted = await getattr(repository, f'persist_{family}_learning_facts')(
        ids['source_uuid'], inputs, [fact_result]
    )
    assert persisted == 1

    card_inputs = await getattr(repository, f'load_{family}_flashcard_inputs')(
        ids['source_uuid']
    )
    card_result = card_result_model(
        question=f'What is the selected {family} evidence?',
        answer=f'The selected {family} evidence is preserved.',
    )
    persisted = await getattr(repository, f'persist_{family}_flashcards')(
        ids['source_uuid'], card_inputs, [card_result]
    )
    assert persisted == 1


async def _source_learning_uuid_sets(session, source_uuid: str) -> dict:
    result = await session.run(
        """
        MATCH (source:Source {uuid: $source_uuid})
        OPTIONAL MATCH (source)-[:HAS_LEARNING_FACT]->(learning_fact)
        WITH source, collect(DISTINCT learning_fact.uuid) AS learning_fact_uuids
        OPTIONAL MATCH (source)-[:HAS_FLASHCARD]->(card:SourceFlashcard)
        RETURN learning_fact_uuids,
               collect(DISTINCT card.uuid) AS card_uuids
        """,
        source_uuid=source_uuid,
    )
    record = await result.single(strict=True)
    return {
        'learning_fact_uuids': sorted(record['learning_fact_uuids']),
        'card_uuids': sorted(record['card_uuids']),
    }


async def _capture_source_learning_artifacts(
    session,
    user_uuid: str,
    source_uuid: str,
    family: str,
) -> dict:
    """Capture one source's learning, shared review, and immutable history."""
    learning_fact_label = _SOURCE_LEARNING_FIXTURE_SHAPES[family][4]
    result = await session.run(
        f"""
        MATCH (user:User {{uuid: $user_uuid}})-[:OWNS_SOURCE]->
              (source:Source {{uuid: $source_uuid}})
        MATCH (source)-[:HAS_LEARNING_FACT]->
              (learning_fact:{learning_fact_label})
        MATCH (source)-[:HAS_FLASHCARD]->(card:SourceFlashcard)
        MATCH (card)-[:DERIVED_FROM]->(learning_fact)
        MATCH (user)-[:HAS_CARD_REVIEW]->(review:UserCardReview)
              <-[:HAS_CARD_REVIEW]-(card)
        OPTIONAL MATCH (review)-[:HAS_REVIEW_EVENT]->(event:ReviewEvent)
        WITH source, learning_fact, card, review, event
        ORDER BY event.review_index
        RETURN source.uuid AS source_uuid,
               learning_fact.uuid AS learning_fact_uuid,
               card.uuid AS card_uuid,
               review.uuid AS review_uuid,
               review.fsrs_card_json AS review_fsrs_card_json,
               review.due_at AS review_due_at,
               review.review_count AS review_count,
               collect(CASE WHEN event IS NOT NULL THEN {{
                   uuid: event.uuid,
                   review_index: event.review_index,
                   scheduler_json: event.scheduler_json,
                   fsrs_card_before_json: event.fsrs_card_before_json,
                   fsrs_card_after_json: event.fsrs_card_after_json,
                   fsrs_review_log_json: event.fsrs_review_log_json
               }} END) AS review_events
        """,
        user_uuid=user_uuid,
        source_uuid=source_uuid,
    )
    record = await result.single(strict=True)
    return {
        'source_uuid': record['source_uuid'],
        'learning_fact_uuid': record['learning_fact_uuid'],
        'card_uuid': record['card_uuid'],
        'review_uuid': record['review_uuid'],
        'review_fsrs_card_json': record['review_fsrs_card_json'],
        'review_due_at': record['review_due_at'],
        'review_count': record['review_count'],
        'review_events': [dict(event) for event in record['review_events']],
    }


async def _capture_owner_graph_ids(session, user_uuid: str) -> dict:
    result = await session.run(
        """
        MATCH (user:User {uuid: $user_uuid})
        OPTIONAL MATCH (user)-[:HAS_DECK]->(deck:Deck)
        OPTIONAL MATCH (user)-[:HAS_SCHEDULER_SETTINGS]->
              (settings:UserSchedulerSettings)
        RETURN user.uuid AS user_uuid,
               collect(DISTINCT deck.uuid) AS deck_uuids,
               collect(DISTINCT {
                   uuid: settings.uuid,
                   scheduler_json: settings.scheduler_json
               }) AS scheduler_settings
        """,
        user_uuid=user_uuid,
    )
    record = await result.single(strict=True)
    return {
        'user_uuid': record['user_uuid'],
        'deck_uuids': sorted(record['deck_uuids']),
        'scheduler_settings': sorted(
            (dict(settings) for settings in record['scheduler_settings']),
            key=lambda settings: settings['uuid'],
        ),
    }


async def _capture_deck_memberships(
    session, user_uuid: str
) -> dict[str, list[str]]:
    result = await session.run(
        """
        MATCH (user:User {uuid: $user_uuid})-[:HAS_DECK]->(deck:Deck)
        OPTIONAL MATCH (deck)-[:HAS_CARD]->(card:SourceFlashcard)
        WITH deck, collect(card.uuid) AS card_uuids
        RETURN deck.uuid AS deck_uuid, card_uuids
        ORDER BY deck_uuid
        """,
        user_uuid=user_uuid,
    )
    rows = await result.data()
    return {
        row['deck_uuid']: sorted(
            card_uuid for card_uuid in row['card_uuids'] if card_uuid
        )
        for row in rows
    }


async def _present_node_uuids(session, node_uuids: list[str]) -> list[str]:
    result = await session.run(
        """
        MATCH (node)
        WHERE node.uuid IN $node_uuids
        RETURN collect(node.uuid) AS present_uuids
        """,
        node_uuids=node_uuids,
    )
    record = await result.single(strict=True)
    return sorted(record['present_uuids'])


@pytest.mark.parametrize('family', ('entity', 'event', 'predicate', 'triplet'))
@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j integration environment is not enabled',
)
def test_source_learning_round_trip_preserves_selected_directed_evidence(
    family: str,
):
    async def exercise() -> None:
        database = DatabaseClient(Settings().database)
        suffix = uuid.uuid4().hex
        source_uuid = f'kms2-query-learning-{family}-{suffix}'
        fixture_ids = None
        repository = SourceLearningRepository(database.session)
        try:
            await schema.ensure_structural_schema(database.session)
            async with database.session() as session:
                fixture_ids = await _seed_source_learning_fixture(
                    session, source_uuid, family
                )

            load_inputs = getattr(repository, f'load_{family}_learning_inputs')
            fact_inputs = await load_inputs(source_uuid)
            assert len(fact_inputs) == 1
            assert fact_inputs[0].hub_name == f'Shared {family} hub'
            assert fact_inputs[0].hub_description

            pair_values = {
                (fixture_ids['fact_1'], fixture_ids['triplet_1']): (
                    'Subject one',
                    'relates to',
                    'Object one',
                ),
                (fixture_ids['fact_2'], fixture_ids['triplet_2']): (
                    'Subject two',
                    'causes',
                    'Object two',
                ),
                (fixture_ids['fact_1'], fixture_ids['triplet_3']): (
                    'Subject three',
                    'mentions',
                    'Object three',
                ),
            }
            expected_roles = (
                ('subject', 'object')
                if family in ('entity', 'event')
                else (family,)
            )
            expected_initial = Counter(
                (fact_uuid, triplet_uuid, role)
                for fact_uuid, triplet_uuid in pair_values
                for role in expected_roles
            )
            observed_initial = Counter(
                (
                    evidence.source_fact_uuid,
                    evidence.triplet_uuid,
                    evidence.member_role,
                )
                for evidence in fact_inputs[0].evidence
            )
            assert observed_initial == expected_initial
            for evidence in fact_inputs[0].evidence:
                assert (
                    evidence.subject,
                    evidence.predicate,
                    evidence.object,
                ) == pair_values[
                    (evidence.source_fact_uuid, evidence.triplet_uuid)
                ]

            fact_result_model, card_result_model = (
                _SOURCE_LEARNING_RESULT_MODELS[family]
            )
            fact_result = fact_result_model(
                facts=[
                    {
                        'text': f'Selected {family} learning fact.',
                        'source_fact_uuids': [
                            fixture_ids['fact_1'],
                            fixture_ids['fact_2'],
                        ],
                        'triplet_uuids': [
                            fixture_ids['triplet_1'],
                            fixture_ids['triplet_2'],
                        ],
                    }
                ]
            )
            persist_facts = getattr(
                repository, f'persist_{family}_learning_facts'
            )
            assert (
                await persist_facts(source_uuid, fact_inputs, [fact_result])
                == 1
            )

            learning_fact_label = _SOURCE_LEARNING_FIXTURE_SHAPES[family][4]
            async with database.session() as session:
                result = await session.run(
                    f"""
                    MATCH (source:Source {{uuid: $source_uuid}})
                          -[:HAS_LEARNING_FACT]->
                          (learning_fact:{learning_fact_label})
                    CALL (learning_fact) {{
                        OPTIONAL MATCH (learning_fact)-[:SUPPORTED_BY]->
                                       (source_fact:SourceFact)
                        RETURN count(source_fact) AS source_fact_count,
                               collect(source_fact.uuid) AS source_fact_uuids
                    }}
                    CALL (learning_fact) {{
                        OPTIONAL MATCH
                            (learning_fact)-[:SUPPORTED_BY_TRIPLET]->
                            (triplet:SourceTriplet)
                        RETURN count(triplet) AS triplet_count,
                               collect(triplet.uuid) AS triplet_uuids
                    }}
                    RETURN learning_fact.uuid AS learning_fact_uuid,
                           source_fact_count,
                           source_fact_uuids,
                           triplet_count,
                           triplet_uuids
                    """,
                    source_uuid=source_uuid,
                )
                persisted_fact = await result.single(strict=True)
            assert persisted_fact['source_fact_count'] == 2
            assert sorted(persisted_fact['source_fact_uuids']) == sorted(
                (fixture_ids['fact_1'], fixture_ids['fact_2'])
            )
            assert persisted_fact['triplet_count'] == 2
            assert sorted(persisted_fact['triplet_uuids']) == sorted(
                (fixture_ids['triplet_1'], fixture_ids['triplet_2'])
            )

            load_cards = getattr(repository, f'load_{family}_flashcard_inputs')
            card_inputs = await load_cards(source_uuid)
            assert len(card_inputs) == 1
            expected_selected_pairs = {
                (fixture_ids['fact_1'], fixture_ids['triplet_1']),
                (fixture_ids['fact_2'], fixture_ids['triplet_2']),
            }
            expected_card_evidence = [
                (fact_uuid, triplet_uuid, role)
                for fact_uuid, triplet_uuid in sorted(expected_selected_pairs)
                for role in sorted(expected_roles)
            ]
            observed_card_evidence = [
                (
                    evidence.source_fact_uuid,
                    evidence.triplet_uuid,
                    evidence.member_role,
                )
                for evidence in card_inputs[0].evidence
            ]
            assert observed_card_evidence == expected_card_evidence
            for evidence in card_inputs[0].evidence:
                pair = (evidence.source_fact_uuid, evidence.triplet_uuid)
                assert pair in expected_selected_pairs
                assert (
                    evidence.subject,
                    evidence.predicate,
                    evidence.object,
                ) == pair_values[pair]

            card_result = card_result_model(
                question=f'What is the selected {family} evidence?',
                answer=f'The selected {family} evidence is preserved.',
            )
            persist_cards = getattr(repository, f'persist_{family}_flashcards')
            assert (
                await persist_cards(source_uuid, card_inputs, [card_result])
                == 1
            )
            async with database.session() as session:
                result = await session.run(
                    f"""
                    MATCH (source:Source {{uuid: $source_uuid}})
                          -[:HAS_FLASHCARD]->(card:SourceFlashcard)
                    MATCH (card)-[:DERIVED_FROM]->
                          (learning_fact:{learning_fact_label})
                    RETURN card.uuid AS card_uuid,
                           card.question AS question,
                           card.answer AS answer,
                           learning_fact.uuid AS learning_fact_uuid
                    """,
                    source_uuid=source_uuid,
                )
                card_record = await result.single(strict=True)
            assert card_record['question'] == card_result.question
            assert card_record['answer'] == card_result.answer
            assert (
                card_record['learning_fact_uuid']
                == persisted_fact['learning_fact_uuid']
            )

            async with database.session() as session:
                before_empty_result = await _source_learning_uuid_sets(
                    session, source_uuid
                )
            empty_result = fact_result_model(facts=[])
            assert (
                await persist_facts(source_uuid, fact_inputs, [empty_result])
                == 0
            )
            async with database.session() as session:
                after_empty_result = await _source_learning_uuid_sets(
                    session, source_uuid
                )
            assert after_empty_result == before_empty_result
        finally:
            try:
                await repository.clear_source_learning(source_uuid)
                if fixture_ids is not None:
                    async with database.session() as session:
                        result = await session.run(
                            """
                            MATCH (node)
                            WHERE node.uuid IN $fixture_uuids
                            DETACH DELETE node
                            """,
                            fixture_uuids=fixture_ids['all_uuids'],
                        )
                        await result.consume()
            finally:
                await database.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ('operation', 'family'),
    (
        ('source', 'entity'),
        ('facts', 'entity'),
        ('triplets', 'triplet'),
        ('entity_hubs', 'entity'),
        ('event_hubs', 'event'),
        ('predicate_hubs', 'predicate'),
        ('triplet_hubs', 'triplet'),
    ),
)
@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j integration environment is not enabled',
)
def test_source_replacement_invalidates_only_its_learning_snapshot(
    operation: str,
    family: str,
):
    async def exercise() -> None:
        database = DatabaseClient(Settings().database)
        suffix = uuid.uuid4().hex
        source_a_uuid = f'kms2-query-cleanup-a-{suffix}'
        source_b_uuid = f'kms2-query-cleanup-b-{suffix}'
        user = None
        fixture_a = None
        fixture_b = None
        source_learning = SourceLearningRepository(database.session)
        try:
            await schema.ensure_structural_schema(database.session)
            users = build_user_service(database)
            learning = build_learning_service(database)
            user = await users.create_user(f'Query cleanup {suffix}')
            async with database.session() as session:
                fixture_a = await _seed_source_learning_fixture(
                    session, source_a_uuid, family, user.uuid
                )
                fixture_b = await _seed_source_learning_fixture(
                    session, source_b_uuid, family, user.uuid
                )

            await _persist_source_learning_card(
                source_learning, family, fixture_a
            )
            await _persist_source_learning_card(
                source_learning, family, fixture_b
            )
            source_a_artifacts = await _source_learning_uuid_sets_for_source(
                database, source_a_uuid
            )
            source_b_artifacts = await _source_learning_uuid_sets_for_source(
                database, source_b_uuid
            )
            card_a_uuid = source_a_artifacts['card_uuids'][0]
            card_b_uuid = source_b_artifacts['card_uuids'][0]
            assignable_cards = await learning.list_assignable_cards(user.uuid)
            assert sorted(
                card.card_uuid for card in assignable_cards
            ) == sorted((card_a_uuid, card_b_uuid))

            decks = await learning.list_decks(user.uuid)
            default_deck = decks[0]
            second_deck = await learning.create_deck(
                user.uuid, f'Second {suffix}'
            )
            decks = [default_deck, second_deck]
            card_uuids = (card_a_uuid, card_b_uuid)
            for deck in decks:
                for card_uuid in card_uuids:
                    await learning.add_card_to_deck(
                        user.uuid, deck.uuid, card_uuid
                    )
            review_instant = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
            for card_uuid in card_uuids:
                await learning.review_card(
                    user.uuid,
                    default_deck.uuid,
                    card_uuid,
                    ReviewRating.GOOD,
                    review_instant,
                )

            async with database.session() as session:
                artifacts_a = await _capture_source_learning_artifacts(
                    session, user.uuid, source_a_uuid, family
                )
                artifacts_b = await _capture_source_learning_artifacts(
                    session, user.uuid, source_b_uuid, family
                )
                owner_ids_before = await _capture_owner_graph_ids(
                    session, user.uuid
                )
                memberships_before = await _capture_deck_memberships(
                    session, user.uuid
                )
                result = await session.run(
                    """
                    MATCH (source:Source)
                    WHERE source.uuid IN $source_uuids
                    RETURN collect(source.uuid) AS source_uuids
                    """,
                    source_uuids=[source_a_uuid, source_b_uuid],
                )
                source_ids_before = sorted(
                    (await result.single(strict=True))['source_uuids']
                )

            public_cards_before = {
                deck.uuid: [
                    card.card_uuid
                    for card in await learning.list_deck_cards(
                        user.uuid, deck.uuid
                    )
                ]
                for deck in decks
            }
            assert all(
                sorted(card_uuids) == sorted(public_cards_before[deck.uuid])
                for deck in decks
            )
            public_events_a_before = await learning.list_review_events(
                user.uuid, card_a_uuid
            )
            public_events_b_before = await learning.list_review_events(
                user.uuid, card_b_uuid
            )
            assert [event.review_index for event in public_events_a_before] == [
                1
            ]
            assert [event.review_index for event in public_events_b_before] == [
                1
            ]
            assert artifacts_b['review_count'] == 1
            assert [
                event['review_index'] for event in artifacts_b['review_events']
            ] == [1]

            source_replacement = SourceGraphRepository(database.session)
            fact_replacement = SourceFactRepository(database.session)
            triplet_replacement = SourceTripletRepository(database.session)
            entity_replacement = SourceEntityRepository(database.session)
            event_replacement = SourceEventRepository(database.session)
            predicate_replacement = SourcePredicateRepository(database.session)
            replacements = {
                'source': (
                    source_replacement.replace_source,
                    (
                        Source(uuid=source_a_uuid, key=source_a_uuid),
                        [],
                        [],
                        [],
                        [],
                    ),
                ),
                'facts': (
                    fact_replacement.replace_source_facts,
                    (source_a_uuid, []),
                ),
                'triplets': (
                    triplet_replacement.replace_source_triplets,
                    (source_a_uuid, []),
                ),
                'entity_hubs': (
                    entity_replacement.replace_source_entity_hubs,
                    (source_a_uuid, [], []),
                ),
                'event_hubs': (
                    event_replacement.replace_source_event_hubs,
                    (source_a_uuid, [], []),
                ),
                'predicate_hubs': (
                    predicate_replacement.replace_source_predicate_hubs,
                    (source_a_uuid, [], []),
                ),
                'triplet_hubs': (
                    triplet_replacement.replace_source_triplet_hubs,
                    (source_a_uuid, [], []),
                ),
            }
            replace, arguments = replacements[operation]
            await replace(*arguments)

            a_artifact_uuids = [
                artifacts_a['learning_fact_uuid'],
                artifacts_a['card_uuid'],
                artifacts_a['review_uuid'],
                *(event['uuid'] for event in artifacts_a['review_events']),
            ]
            async with database.session() as session:
                assert (
                    await _present_node_uuids(session, a_artifact_uuids) == []
                )
                artifacts_b_after = await _capture_source_learning_artifacts(
                    session, user.uuid, source_b_uuid, family
                )
                owner_ids_after = await _capture_owner_graph_ids(
                    session, user.uuid
                )
                memberships_after = await _capture_deck_memberships(
                    session, user.uuid
                )
                result = await session.run(
                    """
                    MATCH (source:Source)
                    WHERE source.uuid IN $source_uuids
                    RETURN collect(source.uuid) AS source_uuids
                    """,
                    source_uuids=[source_a_uuid, source_b_uuid],
                )
                source_ids_after = sorted(
                    (await result.single(strict=True))['source_uuids']
                )
            assert artifacts_b_after == artifacts_b
            assert owner_ids_after == owner_ids_before
            assert source_ids_after == source_ids_before
            assert memberships_after == {
                deck_uuid: [
                    card_uuid
                    for card_uuid in memberships
                    if card_uuid != artifacts_a['card_uuid']
                ]
                for deck_uuid, memberships in memberships_before.items()
            }
            assert (
                await learning.list_review_events(user.uuid, card_a_uuid) == []
            )
            assert (
                await learning.list_review_events(user.uuid, card_b_uuid)
                == public_events_b_before
            )
            assert {
                deck.uuid: [
                    card.card_uuid
                    for card in await learning.list_deck_cards(
                        user.uuid, deck.uuid
                    )
                ]
                for deck in decks
            } == {deck.uuid: [card_b_uuid] for deck in decks}

            await replace(*arguments)
            async with database.session() as session:
                assert (
                    await _present_node_uuids(session, a_artifact_uuids) == []
                )
                assert (
                    await _capture_source_learning_artifacts(
                        session, user.uuid, source_b_uuid, family
                    )
                    == artifacts_b
                )
                assert (
                    await _capture_owner_graph_ids(session, user.uuid)
                    == owner_ids_before
                )
                assert (
                    await _capture_deck_memberships(session, user.uuid)
                    == memberships_after
                )
        finally:
            try:
                await source_learning.clear_source_learning(source_a_uuid)
                await source_learning.clear_source_learning(source_b_uuid)
                fixture_uuids = []
                for fixture in (fixture_a, fixture_b):
                    if fixture is not None:
                        fixture_uuids.extend(fixture['all_uuids'])
                async with database.session() as session:
                    result = await session.run(
                        """
                        MATCH (node)
                        WHERE node.uuid IN $fixture_uuids
                        DETACH DELETE node
                        """,
                        fixture_uuids=fixture_uuids,
                    )
                    await result.consume()
                    result = await session.run(
                        """
                        MATCH (user:User)
                        WHERE user.uuid IN $user_uuids
                        OPTIONAL MATCH (user)-[:HAS_DECK]->(deck:Deck)
                        OPTIONAL MATCH (user)-[:HAS_SCHEDULER_SETTINGS]->
                              (settings:UserSchedulerSettings)
                        WITH collect(DISTINCT user)
                             + collect(DISTINCT deck)
                             + collect(DISTINCT settings) AS nodes
                        UNWIND nodes AS node
                        WITH node
                        WHERE node IS NOT NULL
                        DETACH DELETE node
                        """,
                        user_uuids=[user.uuid] if user is not None else [],
                    )
                    await result.consume()
            finally:
                await database.close()

    asyncio.run(exercise())


async def _source_learning_uuid_sets_for_source(
    database: DatabaseClient,
    source_uuid: str,
) -> dict[str, list[str]]:
    async with database.session() as session:
        return await _source_learning_uuid_sets(session, source_uuid)
