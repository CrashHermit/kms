import asyncio
import os

import pytest

from kms2.config import Settings
from kms2.core.model import (
    Entity,
    Event,
    Predicate,
    RawAssertion,
    RawTriplet,
    Source,
    SourceBlock,
    SourcePage,
    VisualAsset,
)
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.database.semantic.repository import SemanticRepository
from kms2.database.source.repository import SourceRepository

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
def test_kms2_neo4j_materializes_and_replaces_source():
    async def exercise() -> None:
        database = DatabaseClient(Settings().database)
        source_uuid = 'kms2-integration-source'
        old_block_uuids = [
            'kms2-integration-block-1',
            'kms2-integration-block-2',
        ]
        old_asset_uuids = [
            'kms2-integration-asset-1',
            'kms2-integration-asset-2',
        ]
        source = Source(
            uuid=source_uuid,
            key='integration.pdf',
            metadata={'suite': 'kms2'},
        )
        first_block = SourceBlock(
            uuid=old_block_uuids[0],
            block_type='paragraph',
            content='first block',
            assets=[
                VisualAsset(uuid=old_asset_uuids[0], path='first.png'),
                VisualAsset(uuid=old_asset_uuids[1], path='second.png'),
            ],
        )
        second_block = SourceBlock(
            uuid=old_block_uuids[1],
            block_type='aside_text',
            content='aside block',
        )
        repository = SourceRepository(database.session)
        semantic_repository = SemanticRepository(database.session)
        initial_pages = [
            SourcePage(index=0),
            SourcePage(index=1, blocks=[first_block, second_block]),
        ]

        try:
            await schema.ensure_schema(database.session)
            await repository.replace_source(source, initial_pages)

            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block:SourceBlock)
                    OPTIONAL MATCH (block)-[:CONTAINS_VISUAL_ASSET]->(asset:VisualAsset)
                    RETURN count(DISTINCT source) AS sources,
                           count(DISTINCT page) AS pages,
                           count(DISTINCT block) AS blocks,
                           count(DISTINCT asset) AS assets
                    """,
                    source_uuid=source_uuid,
                )
                counts = await result.single()
                assert dict(counts) == {
                    'sources': 1,
                    'pages': 2,
                    'blocks': 2,
                    'assets': 2,
                }

                result = await session.run(
                    """
                    MATCH (block:SourceBlock)
                    WHERE block.uuid IN $block_uuids
                    RETURN block.uuid AS uuid, labels(block) AS labels
                    ORDER BY block.uuid
                    """,
                    block_uuids=old_block_uuids,
                )
                labels_by_uuid = {
                    record['uuid']: set(record['labels'])
                    async for record in result
                }
                assert labels_by_uuid == {
                    old_block_uuids[0]: {'SourceBlock', 'Text'},
                    old_block_uuids[1]: {'SourceBlock', 'AsideText'},
                }

                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                    OPTIONAL MATCH (source)-[has_page:HAS_PAGE]->()
                    WITH source, count(has_page) AS has_page
                    OPTIONAL MATCH (source)-[first_page:FIRST_PAGE]->()
                    WITH source, has_page, count(first_page) AS first_page
                    OPTIONAL MATCH (source)-[first_block:FIRST_BLOCK]->()
                    WITH source, has_page, first_page, count(first_block) AS first_block
                    OPTIONAL MATCH (source)-[last_block:LAST_BLOCK]->()
                    WITH source, has_page, first_page, first_block, count(last_block) AS last_block
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page)
                    OPTIONAL MATCH (page)-[next_page:NEXT_PAGE]->()
                    WITH source, has_page, first_page, first_block, last_block,
                         count(next_page) AS next_page
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page)
                    OPTIONAL MATCH (page)-[contains_block:CONTAINS_BLOCK]->()
                    WITH source, has_page, first_page, first_block, last_block,
                         next_page, count(contains_block) AS contains_block
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block)
                    OPTIONAL MATCH (block)-[next_block:NEXT_BLOCK]->()
                    WITH source, has_page, first_page, first_block, last_block,
                         next_page, contains_block, count(next_block) AS next_block
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block)
                    OPTIONAL MATCH (block)-[contains_asset:CONTAINS_VISUAL_ASSET]->()
                    WITH source, has_page, first_page, first_block, last_block,
                         next_page, contains_block, next_block,
                         count(contains_asset) AS contains_asset
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block)
                    OPTIONAL MATCH (block)-[first_asset:FIRST_VISUAL_ASSET]->()
                    WITH source, has_page, first_page, first_block, last_block,
                         next_page, contains_block, next_block, contains_asset,
                         count(first_asset) AS first_asset
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block)
                    OPTIONAL MATCH (block)-[last_asset:LAST_VISUAL_ASSET]->()
                    WITH source, has_page, first_page, first_block, last_block,
                         next_page, contains_block, next_block, contains_asset,
                         first_asset, count(last_asset) AS last_asset
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block)
                    OPTIONAL MATCH (block)-[:CONTAINS_VISUAL_ASSET]->(asset)
                    OPTIONAL MATCH (asset)-[next_asset:NEXT_VISUAL_ASSET]->()
                    RETURN has_page, first_page, first_block, last_block, next_page,
                           contains_block, next_block, contains_asset, first_asset,
                           last_asset, count(next_asset) AS next_asset
                    """,
                    source_uuid=source_uuid,
                )
                relationships = await result.single()
                assert dict(relationships) == {
                    'has_page': 2,
                    'first_page': 1,
                    'first_block': 1,
                    'last_block': 1,
                    'next_page': 1,
                    'contains_block': 2,
                    'next_block': 1,
                    'contains_asset': 2,
                    'first_asset': 1,
                    'last_asset': 1,
                    'next_asset': 1,
                }

                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                    RETURN source.uuid AS uuid, source.key AS key,
                           source.metadata AS metadata
                    """,
                    source_uuid=source_uuid,
                )
                source_record = await result.single()
                assert dict(source_record) == {
                    'uuid': source_uuid,
                    'key': 'integration.pdf',
                    'metadata': '{"suite": "kms2"}',
                }
            assertion = RawAssertion(
                triplet=RawTriplet(
                    uuid='kms2-integration-triplet',
                    source_uuid=source_uuid,
                    source_block_uuid=old_block_uuids[0],
                    subject_uuid='kms2-integration-event',
                    object_uuid='kms2-integration-entity',
                    predicate_uuid='kms2-integration-predicate',
                ),
                subject=Event(
                    uuid='kms2-integration-event',
                    source_uuid=source_uuid,
                    source_block_uuid=old_block_uuids[0],
                    name='integration event',
                ),
                object=Entity(
                    uuid='kms2-integration-entity',
                    source_uuid=source_uuid,
                    source_block_uuid=old_block_uuids[0],
                    name='integration entity',
                ),
                predicate=Predicate(
                    uuid='kms2-integration-predicate',
                    source_uuid=source_uuid,
                    source_block_uuid=old_block_uuids[0],
                    predicate='relates to',
                ),
            )
            await semantic_repository.replace_source_assertions(
                source_uuid,
                [assertion],
            )

            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (block:SourceBlock {uuid: $block_uuid})
                          -[:HAS_TRIPLET]->
                          (triplet:Triplet {uuid: $triplet_uuid})
                    MATCH (triplet)-[:HAS_SUBJECT]->(subject:Event)
                    MATCH (triplet)-[:HAS_OBJECT]->(object:Entity)
                    MATCH (triplet)-[:HAS_PREDICATE]->(predicate:Predicate)
                    RETURN count(*) AS relationships
                    """,
                    block_uuid=old_block_uuids[0],
                    triplet_uuid='kms2-integration-triplet',
                )
                semantic_relationships = await result.single()
                assert semantic_relationships['relationships'] == 1

                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                          -[:HAS_TRIPLET]->
                          (triplet:Triplet)
                    RETURN count(triplet) AS direct_source_triplets
                    """,
                    source_uuid=source_uuid,
                )
                direct_source_triplets = await result.single()
                assert direct_source_triplets['direct_source_triplets'] == 0

            await semantic_repository.replace_source_assertions(
                source_uuid,
                [],
            )

            replacement_block = SourceBlock(
                uuid='kms2-integration-block-3',
                block_type='paragraph',
                content='replacement block',
                assets=[
                    VisualAsset(
                        uuid='kms2-integration-asset-3',
                        path='replacement.png',
                    )
                ],
            )
            await repository.replace_source(
                source,
                [SourcePage(index=7, blocks=[replacement_block])],
            )

            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block:SourceBlock)
                    OPTIONAL MATCH (block)-[:CONTAINS_VISUAL_ASSET]->(asset:VisualAsset)
                    RETURN count(DISTINCT source) AS sources,
                           count(DISTINCT page) AS pages,
                           count(DISTINCT block) AS blocks,
                           count(DISTINCT asset) AS assets
                    """,
                    source_uuid=source_uuid,
                )
                counts = await result.single()
                assert dict(counts) == {
                    'sources': 1,
                    'pages': 1,
                    'blocks': 1,
                    'assets': 1,
                }

                result = await session.run(
                    """
                    MATCH (node)
                    WHERE node.uuid IN $stale_uuids
                    RETURN count(node) AS stale_nodes
                    """,
                    stale_uuids=old_block_uuids + old_asset_uuids,
                )
                stale = await result.single()
                assert stale['stale_nodes'] == 0

                result = await session.run(
                    """
                    MATCH (block:SourceBlock {uuid: $block_uuid})
                    RETURN labels(block) AS labels, block.content AS content
                    """,
                    block_uuid='kms2-integration-block-3',
                )
                replacement = await result.single()
                assert set(replacement['labels']) == {
                    'SourceBlock',
                    'Paragraph',
                }
                assert replacement['content'] == 'replacement block'
        finally:
            await database.close()

    asyncio.run(exercise())
