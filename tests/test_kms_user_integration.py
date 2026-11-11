import asyncio
import os
import uuid

import pytest

from kms.config.settings import Settings
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.source.storage.catalog import (
    SourceCatalogRepository,
    SourceOwnershipConflict,
)
from kms.users.repository import UserRepository
from kms.users.service import UserService

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
def test_user_creation_and_exclusive_source_ownership():
    async def exercise() -> None:
        database = DatabaseClient(Settings().database)
        suffix = uuid.uuid4().hex
        source_uuid = f'kms2-user-it-source-{suffix}'
        other_source_uuid = f'kms2-user-it-source-other-{suffix}'
        legacy_source_uuid = f'kms2-user-it-legacy-source-{suffix}'
        users = UserRepository(database.session)
        user_service = UserService(users)
        catalog = SourceCatalogRepository(database.session)
        user_a = None
        user_b = None
        try:
            await schema.ensure_structural_schema(database.session)
            user_a = await user_service.create_user(f'Alex {suffix}')
            user_b = await user_service.create_user(f'Sam {suffix}')
            assert await users.read_user(user_a.uuid) == user_a
            listed_users = await user_service.list_users()
            assert (
                next(user for user in listed_users if user.uuid == user_a.uuid)
                == user_a
            )
            assert {user.uuid for user in listed_users} >= {
                user_b.uuid,
            }
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (user:User {uuid: $user_uuid})
                    OPTIONAL MATCH (user)-[:HAS_DECK]->(deck:Deck)
                    OPTIONAL MATCH (user)-[:HAS_SCHEDULER_SETTINGS]->
                        (settings:UserSchedulerSettings)
                    RETURN count(user) AS users,
                           count(deck) AS decks,
                           count(settings) AS scheduler_settings
                    """,
                    user_uuid=user_a.uuid,
                )
                row = (await result.data())[0]
            assert row == {
                'users': 1,
                'decks': 0,
                'scheduler_settings': 0,
            }

            async with database.session() as session:
                result = await session.run(
                    'CREATE (:Source {uuid: $uuid, key: $key, created_at: datetime()})',
                    uuid=legacy_source_uuid,
                    key='legacy.pdf',
                )
                await result.consume()
                result = await session.run(
                    """
                    CREATE (:Source {
                        uuid: $uuid,
                        key: $key,
                        created_at: datetime(),
                        identity_schema: 'source-content-v3',
                        metadata: '{}',
                        revision: 0
                    })
                    """,
                    uuid=source_uuid,
                    key='owned.pdf',
                )
                await result.consume()
                result = await session.run(
                    """
                    CREATE (:Source {
                        uuid: $uuid,
                        key: $key,
                        created_at: datetime(),
                        identity_schema: 'source-content-v3',
                        metadata: '{}',
                        revision: 0
                    })
                    """,
                    uuid=other_source_uuid,
                    key='other-owned.pdf',
                )
                await result.consume()
            await catalog.adopt_source(user_b.uuid, source_uuid)
            await catalog.adopt_source(user_a.uuid, other_source_uuid)

            with pytest.raises(
                ValueError,
                match='^Source format is obsolete; re-ingest the source$',
            ):
                await catalog.adopt_source(user_a.uuid, legacy_source_uuid)

            assert [
                source.uuid
                for source in await catalog.list_sources(user_a.uuid)
            ] == [other_source_uuid]
            assert [
                source.uuid
                for source in await catalog.list_sources(user_b.uuid)
            ] == [source_uuid]
            unowned_uuids = {
                source.uuid for source in await catalog.list_unowned_sources()
            }
            assert legacy_source_uuid in unowned_uuids
            assert source_uuid not in unowned_uuids

            with pytest.raises(SourceOwnershipConflict):
                await catalog.adopt_source(user_a.uuid, source_uuid)
        finally:
            async with database.session() as session:
                await session.run(
                    """
                    MATCH (source:Source)
                    WHERE source.uuid IN $source_uuids
                    DETACH DELETE source
                    """,
                    source_uuids=[
                        source_uuid,
                        other_source_uuid,
                        legacy_source_uuid,
                    ],
                )
                await session.run(
                    """
                    MATCH (user:User)
                    WHERE user.uuid IN $user_uuids
                    DETACH DELETE user
                    """,
                    user_uuids=[
                        user.uuid
                        for user in (user_a, user_b)
                        if user is not None
                    ],
                )
            await database.close()

    asyncio.run(exercise())
