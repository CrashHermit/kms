import asyncio
import os

import pytest

from kms2.config import Settings
from kms2.core.model import (
    Instruction,
    Procedure,
    RawAssertion,
    RawTriplet,
    Source,
    SourceBlock,
    SourceEntity,
    SourceEntityDescriptionResult,
    SourceEvent,
    SourceEventDescriptionResult,
    SourcePage,
    SourcePredicate,
    SourcePredicateDescriptionResult,
    Statement,
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
        initial_instructions = [
            Instruction(
                uuid='kms2-integration-instruction-1',
                member_block_uuids=[old_block_uuids[0]],
                governed_statement_uuids=['kms2-integration-statement-1'],
            )
        ]
        initial_statements = [
            Statement(
                uuid='kms2-integration-statement-1',
                member_block_uuids=[old_block_uuids[1]],
                is_exercise=True,
            )
        ]
        initial_procedures = [
            Procedure(
                uuid='kms2-integration-procedure-1',
                member_block_uuids=[old_block_uuids[0]],
            )
        ]

        try:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=Settings().local_models.embedding.model.dimension,
            )
            await repository.replace_source(
                source,
                initial_pages,
                initial_instructions,
                initial_statements,
                initial_procedures,
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
                subject=SourceEvent(
                    uuid='kms2-integration-event',
                    source_uuid=source_uuid,
                    source_block_uuid=old_block_uuids[0],
                    name='integration event',
                ),
                object=SourceEntity(
                    uuid='kms2-integration-entity',
                    source_uuid=source_uuid,
                    source_block_uuid=old_block_uuids[0],
                    name='integration entity',
                ),
                predicate=SourcePredicate(
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
                    MATCH (triplet)-[:HAS_SUBJECT]->(subject:SourceEvent)
                    MATCH (triplet)-[:HAS_OBJECT]->(object:SourceEntity)
                    MATCH (triplet)-[:HAS_PREDICATE]->(predicate:SourcePredicate)
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
            replacement_instructions = [
                Instruction(
                    uuid='kms2-integration-instruction-2',
                    member_block_uuids=[replacement_block.uuid],
                    governed_statement_uuids=['kms2-integration-statement-2'],
                )
            ]
            replacement_statements = [
                Statement(
                    uuid='kms2-integration-statement-2',
                    member_block_uuids=[replacement_block.uuid],
                    is_exercise=False,
                )
            ]
            replacement_procedures = [
                Procedure(
                    uuid='kms2-integration-procedure-2',
                    member_block_uuids=[replacement_block.uuid],
                )
            ]
            await repository.replace_source(
                source,
                [SourcePage(index=7, blocks=[replacement_block])],
                replacement_instructions,
                replacement_statements,
                replacement_procedures,
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
                    stale_uuids=old_block_uuids
                    + old_asset_uuids
                    + [
                        'kms2-integration-instruction-1',
                        'kms2-integration-statement-1',
                        'kms2-integration-procedure-1',
                    ],
                )
                stale = await result.single()
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                          -[:HAS_INSTRUCTION]->
                          (instruction:Instruction)
                    OPTIONAL MATCH (member:SourceBlock)-[:MEMBER_OF]->(instruction)
                    OPTIONAL MATCH (instruction)-[:GOVERNS]->(statement:Statement)
                    RETURN instruction.uuid AS instruction_uuid,
                           collect(DISTINCT member.uuid) AS member_uuids,
                           collect(DISTINCT statement.uuid) AS statement_uuids
                    """,
                    source_uuid=source_uuid,
                )
                replacement_instruction = await result.single()
                assert replacement_instruction['instruction_uuid'] == (
                    'kms2-integration-instruction-2'
                )
                assert replacement_instruction['member_uuids'] == [
                    'kms2-integration-block-3'
                ]
                assert replacement_instruction['statement_uuids'] == [
                    'kms2-integration-statement-2'
                ]
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


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j integration environment is not enabled',
)
def test_kms2_neo4j_migrates_labels_and_queries_vectors():
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        source_repository = SourceRepository(database.session)
        semantic_repository = SemanticRepository(database.session)
        source_uuid = 'kms2-vector-source'
        block_uuids = ['kms2-vector-block-1', 'kms2-vector-block-2']

        try:
            async with database.session() as session:
                result = await session.run(
                    """
                    CREATE (:Entity {
                        uuid: 'kms2-vector-legacy-entity',
                        source_uuid: $source_uuid,
                        source_block_uuid: $block_uuid,
                        name: 'legacy entity'
                    })
                    CREATE (:Event {
                        uuid: 'kms2-vector-legacy-event',
                        source_uuid: $source_uuid,
                        source_block_uuid: $block_uuid,
                        name: 'legacy event'
                    })
                    CREATE (:Predicate {
                        uuid: 'kms2-vector-legacy-predicate',
                        source_uuid: $source_uuid,
                        source_block_uuid: $block_uuid,
                        predicate: 'legacy predicate'
                    })
                    """,
                    source_uuid=source_uuid,
                    block_uuid=block_uuids[0],
                )
                await result.consume()

            await schema.ensure_schema(
                database.session,
                embedding_dimension=settings.local_models.embedding.model.dimension,
            )

            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (node {uuid: $uuid})
                    RETURN labels(node) AS labels
                    """,
                    uuid='kms2-vector-legacy-entity',
                )
                assert set((await result.single())['labels']) == {
                    'SourceEntity'
                }
                result = await session.run(
                    """
                    MATCH (node {uuid: $uuid})
                    RETURN labels(node) AS labels
                    """,
                    uuid='kms2-vector-legacy-event',
                )
                assert set((await result.single())['labels']) == {'SourceEvent'}
                result = await session.run(
                    """
                    MATCH (node {uuid: $uuid})
                    RETURN labels(node) AS labels
                    """,
                    uuid='kms2-vector-legacy-predicate',
                )
                assert set((await result.single())['labels']) == {
                    'SourcePredicate'
                }

            await source_repository.replace_source(
                Source(uuid=source_uuid, key='vector-search.pdf'),
                [
                    SourcePage(
                        index=0,
                        blocks=[
                            SourceBlock(
                                uuid=block_uuids[0],
                                block_type='paragraph',
                                content='query block',
                                embedding=[1.0, 0.0],
                            ),
                            SourceBlock(
                                uuid=block_uuids[1],
                                block_type='paragraph',
                                content='nearest block',
                                embedding=[0.9, 0.1],
                            ),
                        ],
                    )
                ],
                [],
                [],
                [],
            )

            assertions = [
                RawAssertion(
                    triplet=RawTriplet(
                        uuid='kms2-vector-triplet-1',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[0],
                        subject_uuid='kms2-vector-entity-1',
                        object_uuid='kms2-vector-event-1',
                        predicate_uuid='kms2-vector-predicate-1',
                    ),
                    subject=SourceEntity(
                        uuid='kms2-vector-entity-1',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[0],
                        name='Alpha',
                    ),
                    object=SourceEvent(
                        uuid='kms2-vector-event-1',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[0],
                        name='Appears',
                    ),
                    predicate=SourcePredicate(
                        uuid='kms2-vector-predicate-1',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[0],
                        predicate='supports',
                    ),
                ),
                RawAssertion(
                    triplet=RawTriplet(
                        uuid='kms2-vector-triplet-2',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[1],
                        subject_uuid='kms2-vector-entity-2',
                        object_uuid='kms2-vector-event-2',
                        predicate_uuid='kms2-vector-predicate-2',
                    ),
                    subject=SourceEntity(
                        uuid='kms2-vector-entity-2',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[1],
                        name='Beta',
                    ),
                    object=SourceEvent(
                        uuid='kms2-vector-event-2',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[1],
                        name='Changes',
                    ),
                    predicate=SourcePredicate(
                        uuid='kms2-vector-predicate-2',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[1],
                        predicate='causes',
                    ),
                ),
            ]
            await semantic_repository.replace_source_assertions(
                source_uuid,
                assertions,
            )
            await semantic_repository.update_source_entity_description(
                source_uuid,
                [
                    SourceEntityDescriptionResult(
                        uuid='kms2-vector-entity-1',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[0],
                        name='Alpha',
                        description='first entity',
                        embedding=[1.0, 0.0],
                    ),
                    SourceEntityDescriptionResult(
                        uuid='kms2-vector-entity-2',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[1],
                        name='Beta',
                        description='second entity',
                        embedding=[0.9, 0.1],
                    ),
                ],
            )
            await semantic_repository.update_source_event_description(
                source_uuid,
                [
                    SourceEventDescriptionResult(
                        uuid='kms2-vector-event-1',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[0],
                        name='Appears',
                        description='first event',
                        embedding=[1.0, 0.0],
                    ),
                    SourceEventDescriptionResult(
                        uuid='kms2-vector-event-2',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[1],
                        name='Changes',
                        description='second event',
                        embedding=[0.9, 0.1],
                    ),
                ],
            )
            await semantic_repository.update_source_predicate_description(
                source_uuid,
                [
                    SourcePredicateDescriptionResult(
                        uuid='kms2-vector-predicate-1',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[0],
                        predicate='supports',
                        description='first predicate',
                        embedding=[1.0, 0.0],
                    ),
                    SourcePredicateDescriptionResult(
                        uuid='kms2-vector-predicate-2',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[1],
                        predicate='causes',
                        description='second predicate',
                        embedding=[0.9, 0.1],
                    ),
                ],
            )

            block_matches = await source_repository.find_similar_blocks(
                block_uuids[0],
                top_k=1,
            )
            entity_matches = (
                await semantic_repository.find_similar_source_entities(
                    'kms2-vector-entity-1',
                    top_k=1,
                )
            )
            event_matches = (
                await semantic_repository.find_similar_source_events(
                    'kms2-vector-event-1',
                    top_k=1,
                )
            )
            predicate_matches = (
                await semantic_repository.find_similar_source_predicates(
                    'kms2-vector-predicate-1',
                    top_k=1,
                )
            )

            assert block_matches[0].uuid == block_uuids[1]
            assert entity_matches[0].uuid == 'kms2-vector-entity-2'
            assert event_matches[0].uuid == 'kms2-vector-event-2'
            assert predicate_matches[0].uuid == 'kms2-vector-predicate-2'
            assert all(
                'embedding' not in match.model_dump()
                for matches in (
                    block_matches,
                    entity_matches,
                    event_matches,
                    predicate_matches,
                )
                for match in matches
            )
        finally:
            await database.close()

    asyncio.run(exercise())
