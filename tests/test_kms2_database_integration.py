import asyncio
import os

import pytest

from kms2.config.settings import Settings
from kms2.core.model.block import SourceBlock
from kms2.core.model.global_semantic.global_entity_hub import GlobalEntityHub
from kms2.core.model.global_semantic.global_event_hub import GlobalEventHub
from kms2.core.model.global_semantic.global_predicate_hub import (
    GlobalPredicateHub,
)
from kms2.core.model.global_semantic.global_procedure_hub import (
    GlobalProcedureHub,
)
from kms2.core.model.global_semantic.global_statement_hub import (
    GlobalStatementHub,
)
from kms2.core.model.global_semantic.global_triplet_hub import GlobalTripletHub
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.instruction import Instruction
from kms2.core.model.source_processing.pedagogical import (
    ProcedureDraft,
    StatementDraft,
)
from kms2.core.model.source_semantic.source_entity import (
    SourceEntity,
    SourceEntityDescriptionResult,
)
from kms2.core.model.source_semantic.source_entity_hub import (
    SourceEntityHubCandidate,
)
from kms2.core.model.source_semantic.source_event import (
    SourceEvent,
    SourceEventDescriptionResult,
)
from kms2.core.model.source_semantic.source_event_hub import (
    SourceEventHubCandidate,
)
from kms2.core.model.source_semantic.source_fact_extraction import (
    SourceFact,
    SourceFactContext,
    SourceFactTarget,
)
from kms2.core.model.source_semantic.source_predicate import (
    SourcePredicate,
    SourcePredicateDescriptionResult,
)
from kms2.core.model.source_semantic.source_predicate_hub import (
    SourcePredicateHubCandidate,
)
from kms2.core.model.source_semantic.source_triplet import (
    SourceTriplet,
    SourceTripletOccurrence,
)
from kms2.core.model.visual_asset import VisualAsset
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.database.global_semantic.global_entity_hub_repository import (
    GlobalEntityHubRepository,
)
from kms2.database.global_semantic.global_event_hub_repository import (
    GlobalEventHubRepository,
)
from kms2.database.global_semantic.global_predicate_hub_repository import (
    GlobalPredicateHubRepository,
)
from kms2.database.global_semantic.global_procedure_hub_repository import (
    GlobalProcedureHubRepository,
)
from kms2.database.global_semantic.global_statement_hub_repository import (
    GlobalStatementHubRepository,
)
from kms2.database.global_semantic.global_triplet_repository import (
    GlobalTripletRepository,
)
from kms2.database.source.source_block_repository import SourceBlockRepository
from kms2.database.source.source_graph_repository import SourceGraphRepository
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


def _embedding(first: float, second: float, dimension: int) -> list[float]:
    """Build a deterministic vector matching the configured index dimension."""
    return [first, second, *([0.0] * (dimension - 2))]


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
        source_fact_repository = SourceFactRepository(database.session)
        second_block = SourceBlock(
            uuid=old_block_uuids[1],
            block_type='aside_text',
            content='aside block',
        )
        source_graph_repository = SourceGraphRepository(database.session)
        source_triplet_repository = SourceTripletRepository(database.session)
        initial_pages = [
            SourcePage(index=0),
            SourcePage(index=1, blocks=[first_block, second_block]),
        ]
        initial_instructions = [
            Instruction(
                uuid='kms2-integration-instruction-1',
                member_block_uuids=old_block_uuids,
                governed_statement_uuids=['kms2-integration-statement-1'],
            )
        ]
        initial_statements = [
            StatementDraft(
                uuid='kms2-integration-statement-1',
                member_block_uuids=[old_block_uuids[1]],
                is_exercise=True,
            )
        ]
        initial_procedures = [
            ProcedureDraft(
                uuid='kms2-integration-procedure-1',
                member_block_uuids=[old_block_uuids[0]],
            )
        ]

        try:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=Settings().local_models.embedding.model.dimension,
            )
            await source_graph_repository.replace_source(
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
                    MATCH (statement:SourceStatement {uuid: $statement_uuid})
                    MATCH (procedure:SourceProcedure {uuid: $procedure_uuid})
                    RETURN statement.source_uuid AS statement_source_uuid,
                           procedure.source_uuid AS procedure_source_uuid
                    """,
                    statement_uuid='kms2-integration-statement-1',
                    procedure_uuid='kms2-integration-procedure-1',
                )
                ownership = await result.single()
                assert dict(ownership) == {
                    'statement_source_uuid': source_uuid,
                    'procedure_source_uuid': source_uuid,
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
                    old_block_uuids[0]: {'SourceBlock'},
                    old_block_uuids[1]: {'SourceBlock'},
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
            source_fact = SourceFact(
                uuid='kms2-integration-fact',
                target=SourceFactTarget(
                    uuid='kms2-integration-target',
                    source_blocks=[first_block],
                ),
                context_before=SourceFactContext(
                    uuid='kms2-integration-before'
                ),
                context_after=SourceFactContext(uuid='kms2-integration-after'),
                text='integration event relates to integration entity',
            )
            triplet_occurrence = SourceTripletOccurrence(
                fact=source_fact,
                triplet=SourceTriplet(uuid='kms2-integration-triplet'),
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
            await source_fact_repository.replace_source_facts(
                source_uuid,
                [source_fact],
            )
            await source_triplet_repository.replace_source_triplets(
                source_uuid,
                [triplet_occurrence],
            )

            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                          -[:FIRST_BLOCK]->
                          (first:SourceBlock)
                    MATCH (first)-[:NEXT_BLOCK*0..]->
                          (target_block:SourceBlock {uuid: $block_uuid})
                    MATCH (target_block)<-[:HAS_SOURCE_BLOCK]-(
                          target:SourceFactTarget)
                          <-[:HAS_TARGET]-
                          (fact:SourceFact {uuid: $source_fact_uuid})
                    MATCH (fact)-[:HAS_TRIPLET]->
                          (triplet:SourceTriplet {uuid: $triplet_uuid})
                    MATCH (triplet)-[:HAS_SUBJECT]->(subject:SourceEvent)
                    MATCH (triplet)-[:HAS_OBJECT]->(object:SourceEntity)
                    MATCH (triplet)-[:HAS_PREDICATE]->(predicate:SourcePredicate)
                    RETURN fact.text AS fact_text, count(*) AS relationships
                    """,
                    source_uuid=source_uuid,
                    block_uuid=old_block_uuids[0],
                    source_fact_uuid=source_fact.uuid,
                    triplet_uuid='kms2-integration-triplet',
                )
                semantic_relationships = await result.single()
                assert semantic_relationships['fact_text'] == source_fact.text
                assert semantic_relationships['relationships'] == 1

                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                          -[:HAS_FACT]->
                          (:SourceFact)-[:HAS_TRIPLET]->
                          (triplet:SourceTriplet)
                    RETURN count(triplet) AS direct_source_triplets
                    """,
                    source_uuid=source_uuid,
                )
                direct_source_triplets = await result.single()
                assert direct_source_triplets['direct_source_triplets'] == 0

            await source_triplet_repository.replace_source_triplets(
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
                StatementDraft(
                    uuid='kms2-integration-statement-2',
                    member_block_uuids=[replacement_block.uuid],
                    is_exercise=False,
                )
            ]
            replacement_procedures = [
                ProcedureDraft(
                    uuid='kms2-integration-procedure-2',
                    member_block_uuids=[replacement_block.uuid],
                )
            ]
            await source_graph_repository.replace_source(
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
                          -[:HAS_PAGE]->(:SourcePage)
                          -[:CONTAINS_BLOCK]->(:SourceBlock)
                          -[:MEMBER_OF]->(instruction:Instruction)
                    WITH DISTINCT instruction
                    OPTIONAL MATCH (member:SourceBlock)-[:MEMBER_OF]->(instruction)
                    OPTIONAL MATCH (instruction)-[:GOVERNS]->(statement:SourceStatement)
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
                    MATCH (source:Source {uuid: $source_uuid})-[edge]->
                          (:Instruction)
                    RETURN count(edge) AS direct_instruction_edges
                    """,
                    source_uuid=source_uuid,
                )
                assert (await result.single(strict=True))[
                    'direct_instruction_edges'
                ] == 0

                result = await session.run(
                    """
                    MATCH (block:SourceBlock {uuid: $block_uuid})
                    RETURN labels(block) AS labels, block.content AS content
                    """,
                    block_uuid='kms2-integration-block-3',
                )
                replacement = await result.single()
                assert set(replacement['labels']) == {'SourceBlock'}
                assert replacement['content'] == 'replacement block'
        finally:
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j integration environment is not enabled',
)
def test_kms2_neo4j_clean_schema_materializes_vectors():
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        source_block_repository = SourceBlockRepository(database.session)
        source_graph_repository = SourceGraphRepository(database.session)
        source_triplet_repository = SourceTripletRepository(database.session)
        source_fact_repository = SourceFactRepository(database.session)
        source_entity_repository = SourceEntityRepository(database.session)
        source_event_repository = SourceEventRepository(database.session)
        source_predicate_repository = SourcePredicateRepository(
            database.session
        )
        source_uuid = 'kms2-vector-source'
        block_uuids = ['kms2-vector-block-1', 'kms2-vector-block-2']

        try:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=settings.local_models.embedding.model.dimension,
            )

            await source_graph_repository.replace_source(
                Source(uuid=source_uuid, key='vector-search.pdf'),
                [
                    SourcePage(
                        index=0,
                        blocks=[
                            SourceBlock(
                                uuid=block_uuids[0],
                                block_type='paragraph',
                                content='query block',
                                embedding=_embedding(
                                    1.0,
                                    0.0,
                                    settings.local_models.embedding.model.dimension,
                                ),
                            ),
                            SourceBlock(
                                uuid=block_uuids[1],
                                block_type='paragraph',
                                content='nearest block',
                                embedding=_embedding(
                                    0.9,
                                    0.1,
                                    settings.local_models.embedding.model.dimension,
                                ),
                            ),
                        ],
                    )
                ],
                [],
                [],
                [],
            )

            fact_one = SourceFact(
                uuid='kms2-vector-fact-1',
                target=SourceFactTarget(
                    uuid='kms2-vector-target-1',
                    source_blocks=[
                        SourceBlock(
                            uuid=block_uuids[0],
                            block_type='paragraph',
                            content='query block',
                        )
                    ],
                ),
                context_before=SourceFactContext(uuid='kms2-vector-before-1'),
                context_after=SourceFactContext(uuid='kms2-vector-after-1'),
                text='Alpha supports Appears.',
            )
            fact_two = SourceFact(
                uuid='kms2-vector-fact-2',
                target=SourceFactTarget(
                    uuid='kms2-vector-target-2',
                    source_blocks=[
                        SourceBlock(
                            uuid=block_uuids[1],
                            block_type='paragraph',
                            content='candidate block',
                        )
                    ],
                ),
                context_before=SourceFactContext(uuid='kms2-vector-before-2'),
                context_after=SourceFactContext(uuid='kms2-vector-after-2'),
                text='Beta causes Changes.',
            )
            triplet_occurrences = [
                SourceTripletOccurrence(
                    fact=fact_one,
                    triplet=SourceTriplet(uuid='kms2-vector-triplet-1'),
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
                SourceTripletOccurrence(
                    fact=fact_two,
                    triplet=SourceTriplet(uuid='kms2-vector-triplet-2'),
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
            await source_fact_repository.replace_source_facts(
                source_uuid,
                [fact_one, fact_two],
            )
            await source_triplet_repository.replace_source_triplets(
                source_uuid,
                triplet_occurrences,
            )
            await source_entity_repository.update_source_entity_description(
                source_uuid,
                [
                    SourceEntityDescriptionResult(
                        uuid='kms2-vector-entity-1',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[0],
                        name='Alpha',
                        description='first entity',
                        embedding=_embedding(
                            1.0,
                            0.0,
                            settings.local_models.embedding.model.dimension,
                        ),
                    ),
                    SourceEntityDescriptionResult(
                        uuid='kms2-vector-entity-2',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[1],
                        name='Beta',
                        description='second entity',
                        embedding=_embedding(
                            0.9,
                            0.1,
                            settings.local_models.embedding.model.dimension,
                        ),
                    ),
                ],
            )
            await source_event_repository.update_source_event_description(
                source_uuid,
                [
                    SourceEventDescriptionResult(
                        uuid='kms2-vector-event-1',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[0],
                        name='Appears',
                        description='first event',
                        embedding=_embedding(
                            1.0,
                            0.0,
                            settings.local_models.embedding.model.dimension,
                        ),
                    ),
                    SourceEventDescriptionResult(
                        uuid='kms2-vector-event-2',
                        source_uuid=source_uuid,
                        source_block_uuid=block_uuids[1],
                        name='Changes',
                        description='second event',
                        embedding=_embedding(
                            0.9,
                            0.1,
                            settings.local_models.embedding.model.dimension,
                        ),
                    ),
                ],
            )
            await (
                source_predicate_repository.update_source_predicate_description(
                    source_uuid,
                    [
                        SourcePredicateDescriptionResult(
                            uuid='kms2-vector-predicate-1',
                            source_uuid=source_uuid,
                            source_block_uuid=block_uuids[0],
                            predicate='supports',
                            description='first predicate',
                            embedding=_embedding(
                                1.0,
                                0.0,
                                settings.local_models.embedding.model.dimension,
                            ),
                        ),
                        SourcePredicateDescriptionResult(
                            uuid='kms2-vector-predicate-2',
                            source_uuid=source_uuid,
                            source_block_uuid=block_uuids[1],
                            predicate='causes',
                            description='second predicate',
                            embedding=_embedding(
                                0.9,
                                0.1,
                                settings.local_models.embedding.model.dimension,
                            ),
                        ),
                    ],
                )
            )

            block_matches = await source_block_repository.find_similar_blocks(
                block_uuids[0],
                top_k=1,
            )
            entity_matches = (
                await source_entity_repository.find_similar_source_entities(
                    'kms2-vector-entity-1',
                    top_k=1,
                )
            )
            event_matches = (
                await source_event_repository.find_similar_source_events(
                    'kms2-vector-event-1',
                    top_k=1,
                )
            )
            predicate_matches = await source_predicate_repository.find_similar_source_predicates(
                'kms2-vector-predicate-1',
                top_k=1,
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


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_GDS_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j GDS integration environment is not enabled',
)
def test_kms2_neo4j_source_hub_gds_contracts():
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        source_entity_repository = SourceEntityRepository(database.session)
        source_event_repository = SourceEventRepository(database.session)
        source_predicate_repository = SourcePredicateRepository(
            database.session
        )
        source_uuid = 'kms2-gds-source'
        dimension = settings.local_models.embedding.model.dimension
        first_vector = [1.0] + [0.0] * (dimension - 1)
        second_vector = [0.99, 0.01] + [0.0] * (dimension - 2)
        try:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=dimension,
            )
            async with database.session() as session:
                result = await session.run(
                    """
                    UNWIND $entities AS entity_row
                    CREATE (:SourceEntity {
                        uuid: entity_row.uuid,
                        source_uuid: $source_uuid,
                        source_block_uuid: entity_row.block_uuid,
                        name: entity_row.name,
                        description: entity_row.description,
                        embedding: entity_row.embedding
                    })
                    WITH count(*) AS _
                    UNWIND $events AS event_row
                    CREATE (:SourceEvent {
                        uuid: event_row.uuid,
                        source_uuid: $source_uuid,
                        source_block_uuid: event_row.block_uuid,
                        name: event_row.name,
                        description: event_row.description,
                        embedding: event_row.embedding
                    })
                    WITH count(*) AS _
                    UNWIND $predicates AS predicate_row
                    CREATE (:SourcePredicate {
                        uuid: predicate_row.uuid,
                        source_uuid: $source_uuid,
                        source_block_uuid: predicate_row.block_uuid,
                        predicate: predicate_row.predicate,
                        description: predicate_row.description,
                        embedding: predicate_row.embedding
                    })
                    """,
                    source_uuid=source_uuid,
                    entities=[
                        {
                            'uuid': 'kms2-gds-entity-1',
                            'block_uuid': 'kms2-gds-block-1',
                            'name': 'Alpha',
                            'description': 'same entity',
                            'embedding': first_vector,
                        },
                        {
                            'uuid': 'kms2-gds-entity-2',
                            'block_uuid': 'kms2-gds-block-2',
                            'name': 'Alpha term',
                            'description': 'same entity',
                            'embedding': second_vector,
                        },
                    ],
                    events=[
                        {
                            'uuid': 'kms2-gds-event-1',
                            'block_uuid': 'kms2-gds-block-1',
                            'name': 'Integrates',
                            'description': 'same event',
                            'embedding': first_vector,
                        },
                        {
                            'uuid': 'kms2-gds-event-2',
                            'block_uuid': 'kms2-gds-block-2',
                            'name': 'Integration',
                            'description': 'same event',
                            'embedding': second_vector,
                        },
                    ],
                    predicates=[
                        {
                            'uuid': 'kms2-gds-predicate-1',
                            'block_uuid': 'kms2-gds-block-1',
                            'predicate': 'supports',
                            'description': 'same relation',
                            'embedding': first_vector,
                        },
                        {
                            'uuid': 'kms2-gds-predicate-2',
                            'block_uuid': 'kms2-gds-block-2',
                            'predicate': 'supports strongly',
                            'description': 'same relation',
                            'embedding': second_vector,
                        },
                    ],
                )
                await result.consume()

            await source_entity_repository.replace_source_entity_accepted_edges(
                source_uuid,
                [
                    SourceEntityHubCandidate(
                        left_uuid='kms2-gds-entity-1',
                        left_name='Alpha',
                        left_description='same entity',
                        right_uuid='kms2-gds-entity-2',
                        right_name='Alpha term',
                        right_description='same entity',
                        score=0.95,
                    )
                ],
            )
            await source_event_repository.replace_source_event_accepted_edges(
                source_uuid,
                [
                    SourceEventHubCandidate(
                        left_uuid='kms2-gds-event-1',
                        left_name='Integrates',
                        left_description='same event',
                        right_uuid='kms2-gds-event-2',
                        right_name='Integration',
                        right_description='same event',
                        score=0.95,
                    )
                ],
            )
            await source_predicate_repository.replace_source_predicate_accepted_edges(
                source_uuid,
                [
                    SourcePredicateHubCandidate(
                        left_uuid='kms2-gds-predicate-1',
                        left_predicate='supports',
                        left_description='same relation',
                        left_subject='Alpha',
                        left_object='Alpha term',
                        right_uuid='kms2-gds-predicate-2',
                        right_predicate='supports strongly',
                        right_description='same relation',
                        right_subject='Alpha',
                        right_object='Alpha term',
                        score=0.95,
                    )
                ],
            )
            community_groups = [
                await source_entity_repository.detect_source_entity_communities(
                    source_uuid,
                    max_iterations=10,
                    min_association_strength=0.2,
                    minimum_community_size=2,
                ),
                await source_event_repository.detect_source_event_communities(
                    source_uuid,
                    max_iterations=10,
                    min_association_strength=0.2,
                    minimum_community_size=2,
                ),
                await source_predicate_repository.detect_source_predicate_communities(
                    source_uuid,
                    max_iterations=10,
                    min_association_strength=0.2,
                    minimum_community_size=2,
                ),
            ]
            assert all(
                'embedding' not in member.model_dump()
                for groups in community_groups
                for group in groups
                for member in group
            )

            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (left {source_uuid: $source_uuid})
                          -[similarity:SIMILAR_TO]-
                          (right {source_uuid: $source_uuid})
                    WHERE left.uuid < right.uuid
                    RETURN labels(left) AS left_labels,
                           labels(right) AS right_labels,
                           similarity.score AS score
                    """,
                    source_uuid=source_uuid,
                )
                rows = await result.data()
                for label in ('SourceEntity', 'SourceEvent', 'SourcePredicate'):
                    assert any(
                        label in row['left_labels']
                        and label in row['right_labels']
                        and row['score'] >= 0.8
                        for row in rows
                    )
                for graph_name in (
                    f'kms2-source-entity-hubs-{source_uuid}',
                    f'kms2-source-event-hubs-{source_uuid}',
                    f'kms2-source-predicate-hubs-{source_uuid}',
                ):
                    result = await session.run(
                        """
                        CALL gds.graph.exists($graph_name)
                        YIELD exists
                        RETURN exists
                        """,
                        graph_name=graph_name,
                    )
                    assert not (await result.single())['exists']
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (node)
                    WHERE node.source_uuid = $source_uuid
                    DETACH DELETE node
                    """,
                    source_uuid=source_uuid,
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_GDS_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j GDS integration environment is not enabled',
)
def test_kms2_neo4j_global_predicate_hub_contracts():
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        repository = GlobalPredicateHubRepository(database.session)
        dimension = settings.local_models.embedding.model.dimension
        first_vector = [1.0] + [0.0] * (dimension - 1)
        second_vector = [0.99, 0.01] + [0.0] * (dimension - 2)
        hub_rows = [
            {
                'uuid': 'kms2-global-source-hub-1',
                'source_uuid': 'kms2-global-source-1',
                'predicate': 'supports',
                'description': 'provides support',
                'embedding': first_vector,
            },
            {
                'uuid': 'kms2-global-source-hub-2',
                'source_uuid': 'kms2-global-source-2',
                'predicate': 'supports strongly',
                'description': 'provides support',
                'embedding': second_vector,
            },
            {
                'uuid': 'kms2-global-source-hub-3',
                'source_uuid': 'kms2-global-source-1',
                'predicate': 'supports firmly',
                'description': 'provides support',
                'embedding': second_vector,
            },
        ]
        try:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=dimension,
            )
            async with database.session() as session:
                result = await session.run(
                    """
                    UNWIND $hubs AS row
                    CREATE (:SourcePredicateHub {
                        uuid: row.uuid,
                        source_uuid: row.source_uuid,
                        predicate: row.predicate,
                        description: row.description,
                        aliases: [row.predicate],
                        embedding: row.embedding
                    })
                    """,
                    hubs=hub_rows,
                )
                await result.consume()
                result = await session.run(
                    """
                    UNWIND $hubs AS row
                    CREATE (member:SourcePredicate {
                        uuid: row.member_uuid,
                        source_uuid: row.source_uuid,
                        predicate: row.predicate,
                        description: row.description,
                        embedding: row.embedding
                    })
                    WITH member, row
                    MATCH (hub:SourcePredicateHub {uuid: row.uuid})
                    CREATE (member)-[:IN_SOURCE_HUB]->(hub)
                    """,
                    hubs=[
                        {
                            **row,
                            'member_uuid': f'{row["uuid"]}-member',
                        }
                        for row in hub_rows
                    ],
                )
                await result.consume()

            candidates = await repository.read_global_predicate_hub_candidates(
                candidate_limit=10,
                minimum_similarity=0.8,
            )
            assert candidates
            assert all(
                left['source_uuid'] != right['source_uuid']
                for candidate in candidates
                for left, right in (
                    (
                        next(
                            row
                            for row in hub_rows
                            if row['uuid'] == candidate.left_uuid
                        ),
                        next(
                            row
                            for row in hub_rows
                            if row['uuid'] == candidate.right_uuid
                        ),
                    ),
                )
            )

            first_candidate = next(
                candidate
                for candidate in candidates
                if {candidate.left_uuid, candidate.right_uuid}
                == {
                    'kms2-global-source-hub-1',
                    'kms2-global-source-hub-2',
                }
            )
            await repository.replace_global_predicate_hub_accepted_edges(
                [first_candidate]
            )
            communities = (
                await repository.detect_global_predicate_hub_communities(
                    max_iterations=10,
                    min_association_strength=0.2,
                    minimum_community_size=2,
                )
            )
            assert len(communities) == 1
            global_hub = GlobalPredicateHub(
                predicate='supports',
                aliases=['supports', 'supports strongly'],
                description='provides support',
                embedding=first_vector,
            )
            await repository.replace_global_predicate_hubs(
                [global_hub],
                [[member.uuid for member in communities[0]]],
            )

            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source_hub:SourcePredicateHub)
                    WHERE source_hub.uuid STARTS WITH 'kms2-global-source-hub-'
                    OPTIONAL MATCH (member:SourcePredicate)-[:IN_SOURCE_HUB]->(source_hub)
                    RETURN source_hub.uuid AS uuid,
                           source_hub.source_uuid AS source_uuid,
                           source_hub.predicate AS predicate,
                           source_hub.description AS description,
                           count(member) AS source_members
                    ORDER BY uuid
                    """
                )
                source_snapshot = await result.data()
                result = await session.run(
                    """
                    MATCH (member:SourcePredicateHub)
                          -[:IN_GLOBAL_HUB]->
                          (global_hub:GlobalPredicateHub)
                    RETURN count(DISTINCT global_hub) AS hubs,
                           count(member) AS memberships
                    """
                )
                global_snapshot = await result.single()
            assert global_snapshot == {'hubs': 1, 'memberships': 2}

            await repository.replace_global_predicate_hub_accepted_edges([])
            await repository.replace_global_predicate_hubs([], [])

            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source_hub:SourcePredicateHub)
                    WHERE source_hub.uuid STARTS WITH 'kms2-global-source-hub-'
                    OPTIONAL MATCH (member:SourcePredicate)-[:IN_SOURCE_HUB]->(source_hub)
                    RETURN source_hub.uuid AS uuid,
                           source_hub.source_uuid AS source_uuid,
                           source_hub.predicate AS predicate,
                           source_hub.description AS description,
                           count(member) AS source_members
                    ORDER BY uuid
                    """
                )
                assert await result.data() == source_snapshot
                result = await session.run(
                    'MATCH (hub:GlobalPredicateHub) RETURN count(hub) AS count'
                )
                assert (await result.single())['count'] == 0
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (node)
                    WHERE node.uuid STARTS WITH 'kms2-global-'
                    DETACH DELETE node
                    """
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())


async def _exercise_global_hub_contract(
    *,
    database,
    source_label: str,
    source_member_label: str,
    source_value_property: str,
    global_label: str,
    source_prefix: str,
    global_hub,
    read_candidates,
    replace_edges,
    detect_communities,
    replace_hubs,
) -> None:
    """Exercise one complete cross-source global hub repository contract."""
    settings = Settings()
    dimension = settings.local_models.embedding.model.dimension
    first_vector = _embedding(1.0, 0.0, dimension)
    second_vector = _embedding(0.99, 0.01, dimension)
    hub_rows = [
        {
            'uuid': f'{source_prefix}1',
            'source_uuid': f'{source_prefix}source-1',
            'value': 'Alpha',
            'description': 'same concept',
            'embedding': first_vector,
        },
        {
            'uuid': f'{source_prefix}2',
            'source_uuid': f'{source_prefix}source-2',
            'value': 'Alpha term',
            'description': 'same concept',
            'embedding': second_vector,
        },
        {
            'uuid': f'{source_prefix}3',
            'source_uuid': f'{source_prefix}source-1',
            'value': 'Alpha near match',
            'description': 'same concept',
            'embedding': second_vector,
        },
    ]
    await schema.ensure_schema(
        database.session,
        embedding_dimension=dimension,
    )
    async with database.session() as session:
        result = await session.run(
            f"""
            UNWIND $hubs AS row
            CREATE (hub:{source_label} {{
                uuid: row.uuid,
                source_uuid: row.source_uuid,
                {source_value_property}: row.value,
                description: row.description,
                aliases: [row.value],
                embedding: row.embedding
            }})
            WITH row
            CREATE (member:{source_member_label} {{
                uuid: row.uuid + '-member',
                source_uuid: row.source_uuid,
                description: row.description
            }})
            WITH row, member
            MATCH (hub:{source_label} {{uuid: row.uuid}})
            CREATE (member)-[:IN_SOURCE_HUB]->(hub)
            """,
            hubs=hub_rows,
        )
        await result.consume()

    candidates = await read_candidates(
        candidate_limit=10,
        minimum_similarity=0.8,
    )
    source_uuids = {row['uuid']: row['source_uuid'] for row in hub_rows}
    assert candidates
    assert all(
        source_uuids[candidate.left_uuid] != source_uuids[candidate.right_uuid]
        for candidate in candidates
    )
    first_candidate = next(
        candidate
        for candidate in candidates
        if {candidate.left_uuid, candidate.right_uuid}
        == {f'{source_prefix}1', f'{source_prefix}2'}
    )
    await replace_edges([first_candidate])
    communities = await detect_communities(
        max_iterations=10,
        min_association_strength=0.2,
        minimum_community_size=2,
    )
    assert len(communities) == 1
    assert {member.uuid for member in communities[0]} == {
        f'{source_prefix}1',
        f'{source_prefix}2',
    }
    await replace_hubs(
        [global_hub],
        [[member.uuid for member in communities[0]]],
    )

    source_snapshot_query = f"""
        MATCH (source_hub:{source_label})
        WHERE source_hub.uuid STARTS WITH '{source_prefix}'
        OPTIONAL MATCH (member:{source_member_label})
                       -[:IN_SOURCE_HUB]->(source_hub)
        RETURN source_hub.uuid AS uuid,
               properties(source_hub) AS properties,
               count(member) AS source_members
        ORDER BY uuid
    """
    async with database.session() as session:
        result = await session.run(source_snapshot_query)
        source_snapshot = await result.data()
        result = await session.run(
            f"""
            MATCH (member:{source_label})-[:IN_GLOBAL_HUB]->
                  (global_hub:{global_label})
            RETURN count(DISTINCT global_hub) AS hubs,
                   count(member) AS memberships
            """
        )
        global_snapshot = await result.single()
    assert global_snapshot == {'hubs': 1, 'memberships': 2}

    await replace_edges([])
    await replace_hubs([], [])

    async with database.session() as session:
        result = await session.run(source_snapshot_query)
        assert await result.data() == source_snapshot
        result = await session.run(
            f'MATCH (hub:{global_label}) RETURN count(hub) AS count'
        )
        assert (await result.single())['count'] == 0


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_GDS_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j GDS integration environment is not enabled',
)
def test_kms2_neo4j_global_entity_hub_contracts():
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        repository = GlobalEntityHubRepository(database.session)
        dimension = settings.local_models.embedding.model.dimension
        try:
            await _exercise_global_hub_contract(
                database=database,
                source_label='SourceEntityHub',
                source_member_label='SourceEntity',
                source_value_property='canonical_name',
                global_label='GlobalEntityHub',
                source_prefix='kms2-global-entity-hub-',
                global_hub=GlobalEntityHub(
                    canonical_name='Alpha',
                    aliases=['Alpha', 'Alpha term'],
                    description='same concept',
                    embedding=_embedding(1.0, 0.0, dimension),
                ),
                read_candidates=repository.read_global_entity_hub_candidates,
                replace_edges=repository.replace_global_entity_hub_accepted_edges,
                detect_communities=repository.detect_global_entity_hub_communities,
                replace_hubs=repository.replace_global_entity_hubs,
            )
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (node)
                    WHERE node.uuid STARTS WITH 'kms2-global-entity-hub-'
                       OR node:GlobalEntityHub
                    DETACH DELETE node
                    """
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_GDS_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j GDS integration environment is not enabled',
)
def test_kms2_neo4j_global_event_hub_contracts():
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        repository = GlobalEventHubRepository(database.session)
        dimension = settings.local_models.embedding.model.dimension
        try:
            await _exercise_global_hub_contract(
                database=database,
                source_label='SourceEventHub',
                source_member_label='SourceEvent',
                source_value_property='name',
                global_label='GlobalEventHub',
                source_prefix='kms2-global-event-hub-',
                global_hub=GlobalEventHub(
                    name='Alpha',
                    aliases=['Alpha', 'Alpha term'],
                    description='same concept',
                    embedding=_embedding(1.0, 0.0, dimension),
                ),
                read_candidates=repository.read_global_event_hub_candidates,
                replace_edges=repository.replace_global_event_hub_accepted_edges,
                detect_communities=repository.detect_global_event_hub_communities,
                replace_hubs=repository.replace_global_event_hubs,
            )
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (node)
                    WHERE node.uuid STARTS WITH 'kms2-global-event-hub-'
                       OR node:GlobalEventHub
                    DETACH DELETE node
                    """
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_GDS_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j GDS integration environment is not enabled',
)
def test_kms2_neo4j_global_statement_hub_contracts():
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        repository = GlobalStatementHubRepository(database.session)
        dimension = settings.local_models.embedding.model.dimension
        try:
            await _exercise_global_hub_contract(
                database=database,
                source_label='SourceStatementHub',
                source_member_label='SourceStatement',
                source_value_property='canonical_name',
                global_label='GlobalStatementHub',
                source_prefix='kms2-global-statement-hub-',
                global_hub=GlobalStatementHub(
                    canonical_name='Alpha',
                    aliases=['Alpha', 'Alpha term'],
                    description='same concept',
                    embedding=_embedding(1.0, 0.0, dimension),
                ),
                read_candidates=repository.read_global_statement_hub_candidates,
                replace_edges=repository.replace_global_statement_hub_accepted_edges,
                detect_communities=repository.detect_global_statement_hub_communities,
                replace_hubs=repository.replace_global_statement_hubs,
            )
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (node)
                    WHERE node.uuid STARTS WITH 'kms2-global-statement-hub-'
                       OR node:GlobalStatementHub
                    DETACH DELETE node
                    """
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_GDS_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j GDS integration environment is not enabled',
)
def test_kms2_neo4j_global_procedure_hub_contracts():
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        repository = GlobalProcedureHubRepository(database.session)
        dimension = settings.local_models.embedding.model.dimension
        try:
            await _exercise_global_hub_contract(
                database=database,
                source_label='SourceProcedureHub',
                source_member_label='SourceProcedure',
                source_value_property='canonical_name',
                global_label='GlobalProcedureHub',
                source_prefix='kms2-global-procedure-hub-',
                global_hub=GlobalProcedureHub(
                    canonical_name='Alpha',
                    aliases=['Alpha', 'Alpha term'],
                    description='same concept',
                    embedding=_embedding(1.0, 0.0, dimension),
                ),
                read_candidates=repository.read_global_procedure_hub_candidates,
                replace_edges=repository.replace_global_procedure_hub_accepted_edges,
                detect_communities=repository.detect_global_procedure_hub_communities,
                replace_hubs=repository.replace_global_procedure_hubs,
            )
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (node)
                    WHERE node.uuid STARTS WITH 'kms2-global-procedure-hub-'
                       OR node:GlobalProcedureHub
                    DETACH DELETE node
                    """
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j integration environment is not enabled',
)
def test_kms2_neo4j_global_triplet_projection_contracts():
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        repository = GlobalTripletRepository(database.session)
        dimension = settings.local_models.embedding.model.dimension
        prefix = 'kms2-global-triplet-'
        source_hub_rows = [
            {
                'uuid': f'{prefix}source-{ordinal}',
                'canonical_name': 'supports relation',
                'description': 'A support relation.',
                'subject_uuid': f'{prefix}source-subject',
                'predicate_uuid': f'{prefix}source-predicate',
                'object_uuid': f'{prefix}source-object',
            }
            for ordinal in (1, 2)
        ]
        try:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=dimension,
            )
            async with database.session() as session:
                result = await session.run(
                    """
                    CREATE (:SourceEntityHub {
                        uuid: $source_subject,
                        source_uuid: $source_uuid,
                        canonical_name: 'Alice',
                        description: 'A person.',
                        embedding: $embedding
                    })
                    CREATE (:SourcePredicateHub {
                        uuid: $source_predicate,
                        source_uuid: $source_uuid,
                        predicate: 'supports',
                        aliases: ['supports'],
                        description: 'A support relation.',
                        embedding: $embedding
                    })
                    CREATE (:SourceEntityHub {
                        uuid: $source_object,
                        source_uuid: $source_uuid,
                        canonical_name: 'Acme',
                        description: 'An organization.',
                        embedding: $embedding
                    })
                    CREATE (:GlobalEntityHub {
                        uuid: $global_subject,
                        canonical_name: 'Alice',
                        description: 'A person.',
                        embedding: $embedding
                    })
                    CREATE (:GlobalPredicateHub {
                        uuid: $global_predicate,
                        predicate: 'supports',
                        aliases: ['supports'],
                        description: 'A support relation.',
                        embedding: $embedding
                    })
                    CREATE (:GlobalEntityHub {
                        uuid: $global_object,
                        canonical_name: 'Acme',
                        description: 'An organization.',
                        embedding: $embedding
                    })
                    """,
                    source_subject=source_hub_rows[0]['subject_uuid'],
                    source_predicate=source_hub_rows[0]['predicate_uuid'],
                    source_object=source_hub_rows[0]['object_uuid'],
                    source_uuid=f'{prefix}source',
                    global_subject=f'{prefix}global-subject',
                    global_predicate=f'{prefix}global-predicate',
                    global_object=f'{prefix}global-object',
                    embedding=_embedding(1.0, 0.0, dimension),
                )
                await result.consume()
                result = await session.run(
                    """
                    MATCH (subject:SourceEntityHub {uuid: $source_subject})
                    MATCH (predicate:SourcePredicateHub {uuid: $source_predicate})
                    MATCH (object:SourceEntityHub {uuid: $source_object})
                    MATCH (global_subject:GlobalEntityHub {uuid: $global_subject})
                    MATCH (global_predicate:GlobalPredicateHub {
                        uuid: $global_predicate
                    })
                    MATCH (global_object:GlobalEntityHub {uuid: $global_object})
                    CREATE (subject)-[:IN_GLOBAL_HUB]->(global_subject)
                    CREATE (predicate)-[:IN_GLOBAL_HUB]->(global_predicate)
                    CREATE (object)-[:IN_GLOBAL_HUB]->(global_object)
                    WITH subject, predicate, object
                    UNWIND $source_hubs AS row
                    CREATE (source_hub:SourceTripletHub {
                        uuid: row.uuid,
                        source_uuid: $source_uuid,
                        canonical_name: row.canonical_name,
                        description: row.description,
                        embedding: $embedding,
                        subject_hub_uuid: row.subject_uuid,
                        predicate_hub_uuid: row.predicate_uuid,
                        object_hub_uuid: row.object_uuid
                    })
                    CREATE (source_hub)-[:HAS_SUBJECT_HUB]->(subject)
                    CREATE (source_hub)-[:HAS_PREDICATE_HUB]->(predicate)
                    CREATE (source_hub)-[:HAS_OBJECT_HUB]->(object)
                    """,
                    source_subject=source_hub_rows[0]['subject_uuid'],
                    source_predicate=source_hub_rows[0]['predicate_uuid'],
                    source_object=source_hub_rows[0]['object_uuid'],
                    global_subject=f'{prefix}global-subject',
                    global_predicate=f'{prefix}global-predicate',
                    global_object=f'{prefix}global-object',
                    source_uuid=f'{prefix}source',
                    source_hubs=source_hub_rows,
                    embedding=_embedding(1.0, 0.0, dimension),
                )
                await result.consume()

            assert await repository.replace_global_triplets() == 2
            groups = await repository.read_global_triplet_hub_groups()
            assert len(groups) == 1
            group = groups[0]
            assert len(group.global_triplet_uuids) == 2
            global_hub = GlobalTripletHub(
                canonical_name='supports relation',
                description='A support relation.',
                embedding=_embedding(1.0, 0.0, dimension),
                subject_hub_uuid=f'{prefix}global-subject',
                predicate_hub_uuid=f'{prefix}global-predicate',
                object_hub_uuid=f'{prefix}global-object',
            )
            await repository.replace_global_triplet_hubs(
                [global_hub], [group.global_triplet_uuids]
            )

            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:SourceTripletHub)
                          -[:IN_GLOBAL_TRIPLET]->
                          (triplet:GlobalTriplet)
                          -[:HAS_SUBJECT_HUB]->
                          (subject:GlobalEntityHub)
                    MATCH (triplet)-[:HAS_PREDICATE_HUB]->
                          (predicate:GlobalPredicateHub)
                    MATCH (triplet)-[:HAS_OBJECT_HUB]->
                          (object:GlobalEntityHub)
                    MATCH (triplet)-[:IN_GLOBAL_HUB]->
                          (hub:GlobalTripletHub)
                    WHERE source.uuid STARTS WITH $prefix
                    RETURN count(DISTINCT triplet) AS triplets,
                           count(DISTINCT hub) AS hubs,
                           count(DISTINCT subject) AS subjects,
                           count(DISTINCT predicate) AS predicates,
                           count(DISTINCT object) AS objects
                    """,
                    prefix=prefix,
                )
                snapshot = await result.single()
            assert snapshot == {
                'triplets': 2,
                'hubs': 1,
                'subjects': 1,
                'predicates': 1,
                'objects': 1,
            }
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (node)
                    WHERE node.uuid STARTS WITH $prefix
                       OR node:GlobalTriplet
                       OR node:GlobalTripletHub
                    DETACH DELETE node
                    """,
                    prefix=prefix,
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())
