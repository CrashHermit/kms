"""Real Neo4j coverage for owner-scoped knowledge storage."""

import asyncio
import os
from uuid import uuid4

import pytest

from kms.config.settings import Settings
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.knowledge.models import Argument, Proposition
from kms.knowledge.storage import KnowledgeRepository

_REQUIRED_ENVIRONMENT = (
    'KMS_DATABASE__URI',
    'KMS_DATABASE__USERNAME',
    'KMS_DATABASE__PASSWORD',
    'KMS_DATABASE__DATABASE',
)


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_knowledge_storage_is_atomic_owner_scoped_and_provenanced():
    async def exercise() -> None:
        database = DatabaseClient(Settings().database)
        repository = KnowledgeRepository(database.session)
        suffix = uuid4().hex
        owner_uuid = f'knowledge-it-owner-{suffix}'
        other_uuid = f'knowledge-it-other-{suffix}'
        source_uuid = f'knowledge-it-source-{suffix}'
        unit_uuid = f'knowledge-it-unit-{suffix}'
        source_model = 'knowledge-test-model'
        constructed_model = source_model
        premise = Proposition(
            uuid=f'knowledge-it-premise-{suffix}',
            name='premise',
            statement='A premise',
        )
        conclusion = Proposition(
            uuid=f'knowledge-it-conclusion-{suffix}',
            name='conclusion',
            statement='A conclusion',
        )
        constructed_premise = Proposition(
            uuid=f'knowledge-it-constructed-premise-{suffix}',
            name='constructed premise',
            statement='A separate premise',
        )
        constructed_conclusion = Proposition(
            uuid=f'knowledge-it-constructed-conclusion-{suffix}',
            name='constructed conclusion',
            statement='A separate conclusion',
        )
        vector = [1.0, 0.0]
        try:
            await schema.ensure_structural_schema(database.session)
            async with database.session() as session:
                result = await session.run(
                    """
                    CREATE (owner:User {uuid: $owner_uuid})
                    CREATE (other:User {uuid: $other_uuid})
                    CREATE (source:Source {
                        uuid: $source_uuid, key: $source_uuid,
                        created_at: datetime(), updated_at: datetime(),
                        metadata: '{}', revision: 0
                    })
                    CREATE (unit:SourceUnit {
                        uuid: $unit_uuid, created_at: datetime(),
                        updated_at: datetime(), kind: 'content'
                    })
                    CREATE (owner)-[:OWNS_SOURCE {
                        created_at: datetime(), updated_at: datetime()
                    }]->(source)
                    CREATE (source)-[:HAS_UNIT {
                        created_at: datetime(), updated_at: datetime()
                    }]->(unit)
                    """,
                    owner_uuid=owner_uuid,
                    other_uuid=other_uuid,
                    source_uuid=source_uuid,
                    unit_uuid=unit_uuid,
                )
                await result.consume()

            stored = await repository.save(
                owner_uuid,
                [premise, conclusion],
                [vector, [0.8, 0.6]],
                [
                    Argument(
                        uuid=f'knowledge-it-source-argument-{suffix}',
                        premise_uuids=[premise.uuid],
                        conclusion_uuid=conclusion.uuid,
                    )
                ],
                source_uuid=source_uuid,
                target_unit_uuid=unit_uuid,
                embedding_model=source_model,
                existing_proposition_uuids=[],
                existing_argument_uuids=[],
            )
            assert len(stored) == 1
            assert stored[0].uuid == f'knowledge-it-source-argument-{suffix}'

            constructed = await repository.save(
                owner_uuid,
                [constructed_premise, constructed_conclusion],
                [vector, [0.0, 1.0]],
                [
                    Argument(
                        uuid=f'knowledge-it-constructed-argument-{suffix}',
                        premise_uuids=[constructed_premise.uuid],
                        conclusion_uuid=constructed_conclusion.uuid,
                    )
                ],
                source_uuid=None,
                target_unit_uuid=None,
                embedding_model=constructed_model,
                existing_proposition_uuids=[],
                existing_argument_uuids=[],
            )
            assert (
                constructed[0].uuid
                == f'knowledge-it-constructed-argument-{suffix}'
            )
            await repository.save(
                owner_uuid,
                [],
                [],
                [],
                source_uuid=source_uuid,
                target_unit_uuid=unit_uuid,
                embedding_model=constructed_model,
                existing_proposition_uuids=[constructed_premise.uuid],
                existing_argument_uuids=[constructed[0].uuid],
            )

            duplicate = await repository.save(
                owner_uuid,
                [],
                [],
                [
                    Argument(
                        uuid=f'knowledge-it-duplicate-argument-{suffix}',
                        premise_uuids=[premise.uuid],
                        conclusion_uuid=conclusion.uuid,
                    )
                ],
                source_uuid=None,
                target_unit_uuid=None,
                embedding_model=source_model,
                existing_proposition_uuids=[],
                existing_argument_uuids=[],
            )
            assert duplicate[0].uuid == stored[0].uuid
            foreign_proposition = Proposition(
                uuid=f'knowledge-it-foreign-proposition-{suffix}',
                name='foreign',
                statement='Owned by another user',
            )
            await repository.save(
                other_uuid,
                [foreign_proposition],
                [vector],
                [],
                source_uuid=None,
                target_unit_uuid=None,
                embedding_model=source_model,
                existing_proposition_uuids=[],
                existing_argument_uuids=[],
            )

            rolled_back = Proposition(
                uuid=f'knowledge-it-rolled-back-{suffix}',
                name='rollback',
                statement='Must not persist',
            )
            with pytest.raises(PermissionError, match='endpoints'):
                await repository.save(
                    owner_uuid,
                    [rolled_back],
                    [vector],
                    [
                        Argument(
                            premise_uuids=[
                                premise.uuid,
                                foreign_proposition.uuid,
                            ],
                            conclusion_uuid=rolled_back.uuid,
                        )
                    ],
                    source_uuid=None,
                    target_unit_uuid=None,
                    embedding_model=source_model,
                    existing_proposition_uuids=[],
                    existing_argument_uuids=[],
                )

            await repository.save(
                owner_uuid,
                [],
                [],
                [],
                source_uuid=source_uuid,
                target_unit_uuid=unit_uuid,
                embedding_model=source_model,
                existing_proposition_uuids=[premise.uuid],
                existing_argument_uuids=[stored[0].uuid],
            )

            # The test uses a tiny embedding dimension to prove exact cosine order.
            matches = await repository.search(
                owner_uuid,
                vector,
                limit=10,
                embedding_model=source_model,
            )
            expected_matches = sorted(
                [premise.uuid, conclusion.uuid, constructed_premise.uuid],
                key=lambda proposition_uuid: (
                    0
                    if proposition_uuid
                    in {premise.uuid, constructed_premise.uuid}
                    else 1,
                    proposition_uuid,
                ),
            )
            assert [
                proposition.uuid for proposition in matches
            ] == expected_matches + [constructed_conclusion.uuid]
            with pytest.raises(
                ValueError, match='Incompatible embedding model'
            ):
                await repository.search(
                    owner_uuid,
                    vector,
                    limit=10,
                    embedding_model='different-model',
                )

            context = await repository.context(
                owner_uuid, [premise], argument_limit=10
            )
            assert [argument.uuid for argument in context.arguments] == [
                stored[0].uuid
            ]
            assert 'origin' not in context.arguments[0].model_dump()
            assert {
                proposition.uuid for proposition in context.propositions
            } == {
                premise.uuid,
                conclusion.uuid,
            }
            assert [proposition.uuid for proposition in context.matched] == [
                premise.uuid
            ]
            ranked_context = await repository.context(
                owner_uuid, [conclusion, premise], argument_limit=10
            )
            assert [item.uuid for item in ranked_context.matched] == [
                conclusion.uuid,
                premise.uuid,
            ]
            foreign = await repository.search(
                other_uuid,
                vector,
                limit=10,
                embedding_model=source_model,
            )
            assert [proposition.uuid for proposition in foreign] == [
                foreign_proposition.uuid
            ]
            foreign_context = await repository.context(
                other_uuid, [premise], argument_limit=10
            )
            assert foreign_context.matched == []
            assert foreign_context.propositions == []
            assert foreign_context.arguments == []

            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (proposition:Proposition {uuid: $proposition_uuid})
                    OPTIONAL MATCH (proposition)-[provenance:FROM_SOURCE]->
                        (source:Source {uuid: $source_uuid})
                    MATCH (:User {uuid: $owner_uuid})-[:OWNS_ARGUMENT]->
                        (argument:Argument {uuid: $argument_uuid})
                    OPTIONAL MATCH (argument)-[:FROM_SOURCE]->(source)
                    RETURN count(DISTINCT provenance) AS proposition_provenance,
                           argument.origin AS origin,
                           count(DISTINCT source) AS argument_source
                    """,
                    proposition_uuid=premise.uuid,
                    source_uuid=source_uuid,
                    owner_uuid=owner_uuid,
                    argument_uuid=stored[0].uuid,
                )
                row = await result.single()
            assert row['proposition_provenance'] == 1
            assert row['argument_source'] == 1
            assert row['origin'] == 'source'
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (:User {uuid: $owner_uuid})-[:OWNS_ARGUMENT]->
                        (argument:Argument {uuid: $argument_uuid})
                    RETURN argument.origin AS origin
                    """,
                    owner_uuid=owner_uuid,
                    argument_uuid=constructed[0].uuid,
                )
                assert (await result.single())['origin'] == 'constructed'
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (proposition:Proposition {uuid: $proposition_uuid})
                        -[:FROM_SOURCE]->(source:Source {uuid: $source_uuid})
                    MATCH (argument:Argument {uuid: $argument_uuid})
                        -[:FROM_SOURCE]->(source)
                    RETURN count(DISTINCT proposition) AS propositions,
                           count(DISTINCT argument) AS arguments
                    """,
                    proposition_uuid=constructed_premise.uuid,
                    source_uuid=source_uuid,
                    argument_uuid=constructed[0].uuid,
                )
                row = await result.single()
            assert row['propositions'] == 1
            assert row['arguments'] == 1
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (node)
                    WHERE node.uuid IN $uuids
                    DETACH DELETE node
                    """,
                    uuids=[
                        owner_uuid,
                        other_uuid,
                        source_uuid,
                        unit_uuid,
                        premise.uuid,
                        conclusion.uuid,
                        constructed_premise.uuid,
                        constructed_conclusion.uuid,
                        foreign_proposition.uuid,
                        f'knowledge-it-constructed-argument-{suffix}',
                        f'knowledge-it-source-argument-{suffix}',
                    ],
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_knowledge_storage_serialized_locks_across_repositories_and_rolls_back():
    async def exercise() -> None:
        database = DatabaseClient(Settings().database)
        first_repository = KnowledgeRepository(database.session)
        second_repository = KnowledgeRepository(database.session)
        owner_uuid = f'knowledge-it-lock-owner-{uuid4().hex}'
        second_attempting = asyncio.Event()
        second_entered = asyncio.Event()

        async def hold_then_rollback() -> None:
            async with first_repository.serialized(owner_uuid):
                await second_attempting.wait()
                await asyncio.sleep(0.1)
                assert not second_entered.is_set()
                raise RuntimeError('rollback lock holder')

        async def wait_for_lock() -> None:
            second_attempting.set()
            async with second_repository.serialized(owner_uuid):
                second_entered.set()

        try:
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:User {uuid: $owner_uuid})',
                    owner_uuid=owner_uuid,
                )
                await result.consume()
            first = asyncio.create_task(hold_then_rollback())
            second = asyncio.create_task(wait_for_lock())
            with pytest.raises(RuntimeError, match='rollback lock holder'):
                await first
            await asyncio.wait_for(second, timeout=10)
            assert second_entered.is_set()
        finally:
            async with database.session() as session:
                result = await session.run(
                    'MATCH (user:User {uuid: $owner_uuid}) DETACH DELETE user',
                    owner_uuid=owner_uuid,
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())
