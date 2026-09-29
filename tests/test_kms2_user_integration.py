import asyncio
import os
import uuid

import pytest
from fsrs import Scheduler

from kms2.config.settings import Settings
from kms2.core.model.source import Source
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.database.source.source_catalog_repository import (
    SourceOwnershipConflict,
)
from kms2.database.source.source_graph_repository import SourceGraphRepository
from kms2.user_service import UserService

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
def test_user_decks_and_exclusive_source_ownership():
    async def exercise() -> None:
        database = DatabaseClient(Settings().database)
        suffix = uuid.uuid4().hex
        source_uuid = f'kms2-user-it-source-{suffix}'
        other_source_uuid = f'kms2-user-it-source-other-{suffix}'
        service = UserService(database.session)
        user_a = None
        user_b = None
        try:
            await schema.ensure_structural_schema(database.session)
            user_a = await service.create_user(f'Alex {suffix}')
            user_b = await service.create_user(f'Sam {suffix}')
            assert {user.uuid for user in await service.list_users()} >= {
                user_a.uuid,
                user_b.uuid,
            }
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (user:User {uuid: $user_uuid})-[:HAS_DECK]->
                          (deck:Deck)-[:HAS_SETTINGS]->(settings:DeckSettings)
                    RETURN deck.name AS deck_name,
                           settings.scheduler_json AS scheduler_json
                    """,
                    user_uuid=user_a.uuid,
                )
                deck_row = (await result.data())[0]
            assert deck_row['deck_name'] == 'Default'
            scheduler_json = deck_row['scheduler_json']
            assert (
                Scheduler.from_json(scheduler_json).to_json() == scheduler_json
            )

            async with database.session() as session:
                await session.run(
                    'CREATE (:Source {uuid: $uuid, key: $key})',
                    uuid=source_uuid,
                    key='legacy.pdf',
                )
                await session.run(
                    'CREATE (:Source {uuid: $uuid, key: $key})',
                    uuid=other_source_uuid,
                    key='owned.pdf',
                )
            await service.adopt_source(user_b.uuid, source_uuid)
            await service.attach_new_source(user_a.uuid, other_source_uuid)

            assert [
                source.uuid
                for source in await service.list_sources(user_a.uuid)
            ] == [other_source_uuid]
            assert [
                source.uuid
                for source in await service.list_sources(user_b.uuid)
            ] == [source_uuid]
            assert source_uuid not in {
                source.uuid for source in await service.list_unowned_sources()
            }

            with pytest.raises(SourceOwnershipConflict):
                await service.adopt_source(user_a.uuid, source_uuid)

            await SourceGraphRepository(database.session).replace_source(
                Source(uuid=source_uuid, key='legacy.pdf'),
                [],
                [],
                [],
                [],
            )
            assert [
                source.uuid
                for source in await service.list_sources(user_b.uuid)
            ] == [source_uuid]
            assert [
                source.uuid
                for source in await service.list_sources(user_a.uuid)
            ] == [other_source_uuid]
        finally:
            async with database.session() as session:
                await session.run(
                    """
                    MATCH (source:Source)
                    WHERE source.uuid IN $source_uuids
                    DETACH DELETE source
                    """,
                    source_uuids=[source_uuid, other_source_uuid],
                )
                await session.run(
                    """
                    MATCH (user:User)
                    WHERE user.uuid IN $user_uuids
                    OPTIONAL MATCH (user)-[:HAS_DECK]->(deck:Deck)
                    OPTIONAL MATCH (deck)-[:HAS_SETTINGS]->(
                        settings:DeckSettings
                    )
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
