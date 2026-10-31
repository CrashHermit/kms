"""Opt-in Neo4j integration coverage for current KMS2 persistence contracts."""

import asyncio
import os
from uuid import uuid4

import pytest

from kms2.config.settings import Settings
from kms2.core.model.block import SourceBlock
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_extraction import SourceExtractionTarget
from kms2.core.model.source_pedagogical.instruction import Instruction
from kms2.core.model.source_pedagogical.pedagogical import (
    ProcedureDraft,
    StatementDraft,
)
from kms2.core.model.source_semantic.source_entity import (
    SourceEntity,
    SourceEntityDescriptionResult,
)
from kms2.core.model.source_semantic.source_entity_hub import (
    SourceEntityHub,
)
from kms2.core.model.source_semantic.source_procedure_hub import (
    SourceProcedureHub,
)
from kms2.core.model.source_semantic.source_proposition_extraction import (
    SourceProposition,
    SourcePropositionEmbeddingResult,
)
from kms2.core.model.source_semantic.source_proposition_hub import (
    SourcePropositionHub,
)
from kms2.core.model.source_semantic.source_statement_hub import (
    SourceStatementHub,
)
from kms2.core.model.visual_asset import VisualAsset
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.database.source.source_entity_repository import (
    SourceEntityRepository,
)
from kms2.database.source.source_graph_repository import SourceGraphRepository
from kms2.database.source_semantic.source_procedure_repository import (
    SourceProcedureRepository,
)
from kms2.database.source_semantic.source_proposition_repository import (
    SourcePropositionRepository,
)
from kms2.database.source_semantic.source_statement_repository import (
    SourceStatementRepository,
)

_REQUIRED_ENVIRONMENT = (
    'KMS2_DATABASE__URI',
    'KMS2_DATABASE__USERNAME',
    'KMS2_DATABASE__PASSWORD',
    'KMS2_DATABASE__DATABASE',
)


def _embedding(first: float, second: float, dimension: int) -> list[float]:
    return [first, second, *([0.0] * (dimension - 2))]


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j integration environment is not enabled',
)
def test_kms2_neo4j_source_occurrence_round_trip_and_replacement():
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        source_uuid = f'kms2-proposition-it-{uuid4().hex}'
        other_source_uuid = f'kms2-proposition-other-it-{uuid4().hex}'
        source = Source(
            uuid=source_uuid,
            key=f'{source_uuid}.pdf',
            metadata={'suite': 'kms2'},
        )
        other_source = Source(
            uuid=other_source_uuid,
            key=f'{other_source_uuid}.pdf',
            metadata={'suite': 'kms2'},
        )
        block = SourceBlock(
            uuid=f'{source_uuid}-block',
            block_type='paragraph',
            content='A real square matrix is invertible exactly when its determinant is nonzero.',
            embedding=_embedding(
                1.0, 0.0, settings.local_models.embedding.model.dimension
            ),
            assets=[
                VisualAsset(uuid=f'{source_uuid}-asset-1', path='matrix-a.png'),
                VisualAsset(uuid=f'{source_uuid}-asset-2', path='matrix-b.png'),
            ],
        )
        context_block = SourceBlock(
            uuid=f'{source_uuid}-context-block',
            block_type='paragraph',
            content='The matrix A has real entries.',
        )
        following_context_block = SourceBlock(
            uuid=f'{source_uuid}-following-context-block',
            block_type='paragraph',
            content='The argument continues after the definition.',
        )
        other_block = SourceBlock(
            uuid=f'{other_source_uuid}-block',
            block_type='paragraph',
            content='An unrelated source block.',
            embedding=_embedding(
                0.0, 1.0, settings.local_models.embedding.model.dimension
            ),
        )
        proposition = SourceProposition(
            uuid=f'{source_uuid}-z-proposition',
            source_uuid=source_uuid,
            text='A real square matrix is invertible exactly when its determinant is nonzero.',
            target=SourceExtractionTarget(source_blocks=[block]),
        )
        best_proposition = SourceProposition(
            uuid=f'{source_uuid}-a-proposition-z-best',
            source_uuid=source_uuid,
            text='A distinct proposition member with a query-aligned vector.',
            target=SourceExtractionTarget(source_blocks=[context_block]),
        )
        other_proposition = SourceProposition(
            uuid=f'{other_source_uuid}-proposition',
            source_uuid=other_source_uuid,
            text='An unrelated proposition.',
            target=SourceExtractionTarget(source_blocks=[other_block]),
        )
        entity = SourceEntity(
            uuid=f'{source_uuid}-z-entity',
            source_uuid=source_uuid,
            name='real square matrix',
            target=SourceExtractionTarget(source_blocks=[block]),
        )
        unrelated_entity = SourceEntity(
            uuid=f'{source_uuid}-a-unrelated-entity',
            source_uuid=source_uuid,
            name='determinant',
            target=SourceExtractionTarget(source_blocks=[block]),
        )
        other_entity = SourceEntity(
            uuid=f'{other_source_uuid}-entity',
            source_uuid=other_source_uuid,
            name='unrelated concept',
            target=SourceExtractionTarget(source_blocks=[other_block]),
        )
        proposition_repository = SourcePropositionRepository(database.session)
        entity_repository = SourceEntityRepository(database.session)
        source_graph_repository = SourceGraphRepository(database.session)

        async def clean_fixture(fixture: Source) -> None:
            await source_graph_repository.replace_source(fixture, [])
            async with database.session() as session:
                result = await session.run(
                    'MATCH (source:Source {uuid: $uuid}) DETACH DELETE source',
                    uuid=fixture.uuid,
                )
                await result.consume()

        try:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=settings.local_models.embedding.model.dimension,
            )
            source_pages = [
                SourcePage(index=0, blocks=[block]),
                SourcePage(
                    index=1,
                    blocks=[context_block, following_context_block],
                ),
            ]
            await source_graph_repository.replace_source(source, source_pages)
            await source_graph_repository.replace_pedagogical_material(
                source_uuid,
                source_pages,
                [
                    Instruction(
                        uuid=f'{source_uuid}-instruction',
                        member_block_uuids=[
                            block.uuid,
                            context_block.uuid,
                            following_context_block.uuid,
                        ],
                        governed_statement_uuids=[f'{source_uuid}-statement'],
                    )
                ],
                [
                    StatementDraft(
                        uuid=f'{source_uuid}-statement',
                        content='An exercise about the real matrix A.',
                        embedding=[0.1]
                        * settings.local_models.embedding.model.dimension,
                        member_block_uuids=[
                            context_block.uuid,
                            following_context_block.uuid,
                        ],
                        is_exercise=True,
                    )
                ],
                [
                    ProcedureDraft(
                        uuid=f'{source_uuid}-procedure',
                        content='Read the matrix and determine its dimensions.',
                        embedding=[0.2]
                        * settings.local_models.embedding.model.dimension,
                        member_block_uuids=[block.uuid],
                    )
                ],
            )
            await source_graph_repository.replace_source(
                other_source,
                [SourcePage(index=0, blocks=[other_block])],
            )
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
                    OPTIONAL MATCH (page)-[:CONTAINS_BLOCK]->(block:SourceBlock)
                    OPTIONAL MATCH (block)-[:CONTAINS_VISUAL_ASSET]->(asset:VisualAsset)
                    OPTIONAL MATCH (block)-[:MEMBER_OF]->(statement:SourceStatement)
                    OPTIONAL MATCH (block)-[:MEMBER_OF]->(procedure:SourceProcedure)
                    OPTIONAL MATCH (block)-[:MEMBER_OF]->(instruction:Instruction)
                    RETURN count(DISTINCT page) AS pages,
                           count(DISTINCT block) AS blocks,
                           count(DISTINCT asset) AS assets,
                           count(DISTINCT statement) AS statements,
                           count(DISTINCT procedure) AS procedures,
                           count(DISTINCT instruction) AS instructions,
                           collect(DISTINCT statement.content) AS statement_contents,
                           collect(DISTINCT statement.embedding) AS statement_embeddings,
                           collect(DISTINCT procedure.content) AS procedure_contents,
                           collect(DISTINCT procedure.embedding) AS procedure_embeddings
                    """,
                    source_uuid=source_uuid,
                )
                structure = await result.single(strict=True)
            assert dict(structure) == {
                'pages': 2,
                'blocks': 3,
                'assets': 2,
                'statements': 1,
                'procedures': 1,
                'instructions': 1,
                'statement_contents': ['An exercise about the real matrix A.'],
                'statement_embeddings': [
                    [0.1] * settings.local_models.embedding.model.dimension
                ],
                'procedure_contents': [
                    'Read the matrix and determine its dimensions.'
                ],
                'procedure_embeddings': [
                    [0.2] * settings.local_models.embedding.model.dimension
                ],
            }
            statement_repository = SourceStatementRepository(database.session)
            communities = (
                await statement_repository.detect_source_statement_communities(
                    source_uuid,
                    max_iterations=5,
                    min_association_strength=0.1,
                    minimum_community_size=1,
                )
            )
            assert [
                [member.uuid for member in group] for group in communities
            ] == [[f'{source_uuid}-statement']]
            await proposition_repository.replace_source_propositions(
                source_uuid, [proposition, best_proposition]
            )
            await proposition_repository.replace_source_propositions(
                other_source_uuid, [other_proposition]
            )
            await entity_repository.replace_source_entities(
                source_uuid, [entity, unrelated_entity]
            )
            await entity_repository.replace_source_entities(
                other_source_uuid, [other_entity]
            )

            proposition_vector = _embedding(
                0.0, 1.0, settings.local_models.embedding.model.dimension
            )
            other_proposition_vector = _embedding(
                1.0, 0.0, settings.local_models.embedding.model.dimension
            )
            await proposition_repository.update_source_proposition_embeddings(
                source_uuid,
                [
                    SourcePropositionEmbeddingResult(
                        **proposition.model_dump(),
                        embedding=proposition_vector,
                    ),
                    SourcePropositionEmbeddingResult(
                        **best_proposition.model_dump(),
                        embedding=_embedding(
                            1.0,
                            0.0,
                            settings.local_models.embedding.model.dimension,
                        ),
                    ),
                ],
            )
            await proposition_repository.update_source_proposition_embeddings(
                other_source_uuid,
                [
                    SourcePropositionEmbeddingResult(
                        **other_proposition.model_dump(),
                        embedding=other_proposition_vector,
                    )
                ],
            )
            entity_vector = _embedding(
                0.0, 1.0, settings.local_models.embedding.model.dimension
            )
            await entity_repository.update_source_entity_description(
                source_uuid,
                [
                    SourceEntityDescriptionResult(
                        **entity.model_dump(),
                        description='A matrix with real entries and equal row and column dimensions.',
                        embedding=entity_vector,
                    ),
                    SourceEntityDescriptionResult(
                        **unrelated_entity.model_dump(),
                        description='The scalar determinant of a matrix.',
                        embedding=entity_vector,
                    ),
                ],
            )
            other_entity_vector = _embedding(
                1.0, 0.0, settings.local_models.embedding.model.dimension
            )
            await entity_repository.update_source_entity_description(
                other_source_uuid,
                [
                    SourceEntityDescriptionResult(
                        **other_entity.model_dump(),
                        description='An unrelated source concept.',
                        embedding=other_entity_vector,
                    )
                ],
            )

            proposition_hub = SourcePropositionHub(
                uuid=f'{source_uuid}-proposition-hub',
            )
            other_proposition_hub = SourcePropositionHub(
                uuid=f'{other_source_uuid}-proposition-hub',
            )
            entity_hub = SourceEntityHub(
                uuid=f'{source_uuid}-entity-hub',
            )
            unrelated_entity_hub = SourceEntityHub(
                uuid=f'{source_uuid}-unrelated-entity-hub',
            )
            other_entity_hub = SourceEntityHub(
                uuid=f'{other_source_uuid}-entity-hub',
            )
            await proposition_repository.replace_source_proposition_hubs(
                source_uuid,
                [proposition_hub],
                [[proposition.uuid, best_proposition.uuid]],
            )
            await proposition_repository.replace_source_proposition_hubs(
                other_source_uuid,
                [other_proposition_hub],
                [[other_proposition.uuid]],
            )
            await entity_repository.replace_source_entity_hubs(
                source_uuid,
                [entity_hub, unrelated_entity_hub],
                [[entity.uuid], [unrelated_entity.uuid]],
            )
            await entity_repository.replace_source_entity_hubs(
                other_source_uuid,
                [other_entity_hub],
                [[other_entity.uuid]],
            )
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (hub)
                    WHERE hub.uuid IN $hub_uuids
                    RETURN hub.uuid AS uuid, keys(hub) AS keys
                    ORDER BY uuid
                    """,
                    hub_uuids=[
                        proposition_hub.uuid,
                        other_proposition_hub.uuid,
                        entity_hub.uuid,
                        unrelated_entity_hub.uuid,
                        other_entity_hub.uuid,
                    ],
                )
                persisted_hubs = await result.data()
            assert {row['uuid'] for row in persisted_hubs} == {
                proposition_hub.uuid,
                other_proposition_hub.uuid,
                entity_hub.uuid,
                unrelated_entity_hub.uuid,
                other_entity_hub.uuid,
            }
            assert all(row['keys'] == ['uuid'] for row in persisted_hubs)
            loaded_propositions = (
                await proposition_repository.load_source_propositions(
                    source_uuid
                )
            )

            assert len(loaded_propositions) == 2
            assert [item.uuid for item in loaded_propositions] == [
                proposition.uuid,
                best_proposition.uuid,
            ]
            loaded_proposition = loaded_propositions[0]
            assert loaded_proposition.text == proposition.text
            assert [
                item.uuid for item in loaded_proposition.target.source_blocks
            ] == [block.uuid]
            hub_members = await proposition_repository.read_source_proposition_hub_members(
                source_uuid
            )
            assert [member.uuid for member in hub_members] == [
                proposition.uuid,
                best_proposition.uuid,
            ]
            loaded_entities = await entity_repository.load_source_entities(
                source_uuid
            )
            assert [item.uuid for item in loaded_entities] == [
                unrelated_entity.uuid,
                entity.uuid,
            ]
            loaded_entity = loaded_entities[1]
            assert loaded_entity.name == entity.name
            assert [
                item.uuid for item in loaded_entity.target.source_blocks
            ] == [block.uuid]
            assert all(
                'context_before' not in occurrence.model_dump()
                and 'context_after' not in occurrence.model_dump()
                for occurrence in [*loaded_propositions, *loaded_entities]
            )
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                    MATCH (source)-[:HAS_PAGE]->(:SourcePage)
                        -[:CONTAINS_BLOCK]->(owner:SourceBlock)
                        -[:MEMBER_OF]->(occurrence)
                    WHERE occurrence:SourceEntity
                       OR occurrence:SourceProposition
                    RETURN collect({
                        uuid: occurrence.uuid,
                        owner_uuid: owner.uuid,
                        keys: keys(occurrence),
                        context_edges: size([
                            (occurrence)-[reference]->()
                            WHERE type(reference) CONTAINS 'CONTEXT' | 1
                        ])
                    }) AS occurrence_ownership
                    """,
                    source_uuid=source_uuid,
                )
                evidence_graph = await result.single(strict=True)
            assert len(evidence_graph['occurrence_ownership']) == 4
            assert {
                (item['uuid'], item['owner_uuid'], item['context_edges'])
                for item in evidence_graph['occurrence_ownership']
            } == {
                (proposition.uuid, block.uuid, 0),
                (best_proposition.uuid, context_block.uuid, 0),
                (entity.uuid, block.uuid, 0),
                (unrelated_entity.uuid, block.uuid, 0),
            }
            assert all(
                'ordinal' not in item['keys']
                and 'source_uuid' not in item['keys']
                and 'context_before' not in item['keys']
                and 'context_after' not in item['keys']
                and 'context_start_block_uuid' not in item['keys']
                and 'context_end_block_uuid' not in item['keys']
                for item in evidence_graph['occurrence_ownership']
            )

            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (first:SourceBlock {uuid: $first_uuid})
                        -[:NEXT_BLOCK]->
                        (second:SourceBlock {uuid: $second_uuid})
                    RETURN count(*) AS cross_page_links
                    """,
                    first_uuid=block.uuid,
                    second_uuid=context_block.uuid,
                )
                cross_page_stream = await result.single(strict=True)
            assert cross_page_stream['cross_page_links'] == 1

            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                    OPTIONAL MATCH (source)-[:HAS_PROPOSITION|HAS_ENTITY]->(direct)
                    WITH source, count(DISTINCT direct) AS direct_ownership
                    MATCH (source)-[:HAS_PAGE]->(:SourcePage)
                        -[:CONTAINS_BLOCK]->(block:SourceBlock)-[:MEMBER_OF]->
                        (entity:SourceEntity {uuid: $entity_uuid})
                    RETURN keys(entity) AS entity_keys,
                           direct_ownership,
                           block.uuid AS owning_block_uuid
                    """,
                    source_uuid=source_uuid,
                    entity_uuid=entity.uuid,
                )
                entity_graph = await result.single(strict=True)
            assert 'source_uuid' not in entity_graph['entity_keys']
            assert entity_graph['direct_ownership'] == 0
            assert entity_graph['owning_block_uuid'] == block.uuid
            search_vector = _embedding(
                1.0, 0.0, settings.local_models.embedding.model.dimension
            )
            proposition_matches = (
                await proposition_repository.search_source_proposition_hubs(
                    source_uuid, search_vector, top_k=5
                )
            )
            assert len(proposition_matches) == 1
            proposition_match = proposition_matches[0]
            assert proposition_match.uuid == proposition_hub.uuid
            assert proposition_match.member_uuid == best_proposition.uuid
            assert proposition_match.text == best_proposition.text
            assert proposition_match.target_block_uuids == [context_block.uuid]
            assert proposition_match.source_uuid == source_uuid
            entity_matches = await entity_repository.search_source_entity_hubs(
                source_uuid, search_vector, top_k=2
            )
            assert [match.uuid for match in entity_matches] == [
                entity_hub.uuid,
                unrelated_entity_hub.uuid,
            ]
            entity_match = entity_matches[0]
            assert entity_match.uuid == entity_hub.uuid
            assert entity_match.member_uuid == entity.uuid
            assert entity_match.target_block_uuids == [block.uuid]
            assert entity_match.source_uuid == source_uuid
            other_matches = (
                await proposition_repository.search_source_proposition_hubs(
                    other_source_uuid, search_vector, top_k=1
                )
            )
            assert [match.uuid for match in other_matches] == [
                other_proposition_hub.uuid
            ]

            members = await entity_repository.read_source_entity_hub_members(
                source_uuid
            )
            assert {member.uuid for member in members} == {
                entity.uuid,
                unrelated_entity.uuid,
            }
            owner = next(
                member for member in members if member.uuid == entity.uuid
            )
            assert owner.source_block_uuid == block.uuid
            assert owner.evidence.target_blocks[0].content == block.content
            assert owner.evidence.context_before == []
            assert owner.evidence.context_after == []
            candidates = (
                await entity_repository.read_source_entity_hub_candidates(
                    source_uuid,
                    [[member] for member in members],
                    candidate_limit=1,
                    minimum_similarity=0.0,
                )
            )
            assert {
                frozenset((candidate.left_uuid, candidate.right_uuid))
                for candidate in candidates
            } == {frozenset((entity.uuid, unrelated_entity.uuid))}
            assert (
                await entity_repository.read_source_entity_hub_candidates(
                    source_uuid,
                    [members],
                    candidate_limit=1,
                    minimum_similarity=0.0,
                )
                == []
            )
            other_members = (
                await entity_repository.read_source_entity_hub_members(
                    other_source_uuid
                )
            )
            assert [member.uuid for member in other_members] == [
                other_entity.uuid
            ]
            await proposition_repository.replace_source_proposition_hubs(
                source_uuid, [], []
            )
            await entity_repository.replace_source_entity_hubs(
                source_uuid, [], []
            )
            await proposition_repository.replace_source_propositions(
                source_uuid, []
            )
            assert (
                await proposition_repository.load_source_propositions(
                    source_uuid
                )
                == []
            )
            assert {
                occurrence.uuid
                for occurrence in await entity_repository.load_source_entities(
                    source_uuid
                )
            } == {entity.uuid, unrelated_entity.uuid}
            await entity_repository.replace_source_entities(source_uuid, [])
            assert (
                await entity_repository.load_source_entities(source_uuid) == []
            )
            async with database.session() as session:
                stale = await (
                    await session.run(
                        """
                        MATCH (node)
                        WHERE node.uuid IN $hub_uuids
                        RETURN count(node) AS count
                        """,
                        hub_uuids=[
                            proposition_hub.uuid,
                            entity_hub.uuid,
                            unrelated_entity_hub.uuid,
                        ],
                    )
                ).single(strict=True)
            assert stale['count'] == 0
            assert [
                occurrence.uuid
                for occurrence in await proposition_repository.load_source_propositions(
                    other_source_uuid
                )
            ] == [other_proposition.uuid]
            await proposition_repository.replace_source_propositions(
                source_uuid, [proposition, best_proposition]
            )
            await proposition_repository.replace_source_proposition_hubs(
                source_uuid,
                [proposition_hub],
                [[proposition.uuid, best_proposition.uuid]],
            )
            await entity_repository.replace_source_entities(
                source_uuid, [entity, unrelated_entity]
            )
            await entity_repository.replace_source_entity_hubs(
                source_uuid,
                [entity_hub, unrelated_entity_hub],
                [[entity.uuid], [unrelated_entity.uuid]],
            )
            replacement_block = SourceBlock(
                uuid=f'{source_uuid}-replacement-block',
                block_type='paragraph',
                content='A replacement source block.',
            )
            await source_graph_repository.replace_source(
                source,
                [SourcePage(index=4, blocks=[replacement_block])],
            )

            assert (
                await proposition_repository.load_source_propositions(
                    source_uuid
                )
                == []
            )
            assert (
                await entity_repository.load_source_entities(source_uuid) == []
            )
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $uuid})
                    OPTIONAL MATCH (source)-[:HAS_PROPOSITION|HAS_ENTITY]->(direct)
                    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
                        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(occurrence)
                    WHERE occurrence:SourceProposition OR occurrence:SourceEntity
                    RETURN count(DISTINCT direct) AS direct_ownership,
                           count(DISTINCT occurrence) AS occurrences
                    """,
                    uuid=source_uuid,
                )
                row = await result.single(strict=True)
            assert row['direct_ownership'] == 0
            assert row['occurrences'] == 0
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (hub)
                    WHERE hub.uuid IN $hub_uuids
                    RETURN hub.uuid AS uuid, keys(hub) AS keys
                    ORDER BY uuid
                    """,
                    hub_uuids=[
                        other_proposition_hub.uuid,
                        other_entity_hub.uuid,
                    ],
                )
                remaining_hubs = await result.data()
            assert [hub['uuid'] for hub in remaining_hubs] == [
                other_entity_hub.uuid,
                other_proposition_hub.uuid,
            ]
            assert all(hub['keys'] == ['uuid'] for hub in remaining_hubs)
            assert (
                await entity_repository.read_source_entity_hub_members(
                    source_uuid
                )
                == []
            )
            remaining_propositions = (
                await proposition_repository.load_source_propositions(
                    other_source_uuid
                )
            )
            assert [item.uuid for item in remaining_propositions] == [
                other_proposition.uuid
            ]
            remaining_entities = await entity_repository.load_source_entities(
                other_source_uuid
            )
            assert [item.uuid for item in remaining_entities] == [
                other_entity.uuid
            ]

            async with database.session() as session:
                constraint_result = await session.run(
                    'SHOW CONSTRAINTS YIELD labelsOrTypes RETURN labelsOrTypes'
                )
                constraint_labels = {
                    label
                    for row in await constraint_result.data()
                    for label in row['labelsOrTypes']
                }
                index_result = await session.run(
                    "SHOW INDEXES YIELD name, type WHERE type = 'VECTOR' RETURN name ORDER BY name"
                )
                vector_names = [
                    row['name'] for row in await index_result.data()
                ]
            assert constraint_labels == {
                'Source',
                'SourceBlock',
                'Instruction',
                'SourceStatement',
                'SourceProcedure',
                'VisualAsset',
                'SourceProposition',
                'SourceEntity',
                'SourceEntityHub',
                'SourcePropositionHub',
                'SourceStatementHub',
                'SourceProcedureHub',
                'SourceFlashcard',
                'User',
                'Deck',
                'UserSchedulerSettings',
                'UserCardReview',
                'ReviewEvent',
            }
            assert vector_names == sorted(
                (
                    'source_block_embedding',
                    'source_proposition_embedding',
                    'source_entity_embedding',
                    'source_statement_embedding',
                    'source_procedure_embedding',
                    'source_statement_hub_embedding',
                    'source_procedure_hub_embedding',
                )
            )
        finally:
            await clean_fixture(source)
            await clean_fixture(other_source)
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j integration environment is not enabled',
)
def test_entity_group_retrieval_excludes_own_members_before_limiting():
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        dimension = settings.local_models.embedding.model.dimension
        prefix = f'kms2-entity-groups-{uuid4().hex}'
        source = Source(uuid=f'{prefix}-source', key='groups.pdf')
        foreign_source = Source(uuid=f'{prefix}-foreign', key='foreign.pdf')
        graph_repository = SourceGraphRepository(database.session)
        repository = SourceEntityRepository(database.session)
        blocks = [
            SourceBlock(
                uuid=f'{prefix}-block-{index}',
                block_type='paragraph',
                content=content,
            )
            for index, content in enumerate(
                [
                    'A matrix is a rectangular array of entries.',
                    'The array M is a matrix.',
                    'The determinant is a scalar associated with a square matrix.',
                    'An eigenvalue is a scalar associated with a linear transformation.',
                    'This matrix belongs to a different source.',
                ]
            )
        ]
        entities = [
            SourceEntity(
                uuid=f'{prefix}-entity-{index}',
                source_uuid=source.uuid if index < 4 else foreign_source.uuid,
                name=name,
                target=SourceExtractionTarget(source_blocks=[blocks[index]]),
            )
            for index, name in enumerate(
                [
                    'matrix',
                    'array M',
                    'determinant',
                    'eigenvalue',
                    'foreign matrix',
                ]
            )
        ]
        try:
            await schema.ensure_schema(
                database.session, embedding_dimension=dimension
            )
            for fixture, fixture_blocks, fixture_entities in [
                (source, blocks[:4], entities[:4]),
                (foreign_source, blocks[4:], entities[4:]),
            ]:
                await graph_repository.replace_source(
                    fixture,
                    [SourcePage(index=0, blocks=fixture_blocks)],
                )
                await repository.replace_source_entities(
                    fixture.uuid, fixture_entities
                )
                await repository.update_source_entity_description(
                    fixture.uuid,
                    [
                        SourceEntityDescriptionResult(
                            **entity.model_dump(),
                            description=entity.target.source_blocks[0].content,
                            embedding=_embedding(*coordinates, dimension),
                        )
                        for entity, coordinates in zip(
                            fixture_entities,
                            [(1.0, 0.01), (1.0, 0.0), (0.8, 0.6), (0.0, 1.0)]
                            if fixture is source
                            else [(1.0, 0.0)],
                            strict=True,
                        )
                    ],
                )
            members = {
                member.uuid: member
                for member in await repository.read_source_entity_hub_members(
                    source.uuid
                )
            }
            groups = [
                [members[entities[0].uuid], members[entities[1].uuid]],
                [members[entities[2].uuid]],
                [members[entities[3].uuid]],
            ]
            candidates = await repository.read_source_entity_hub_candidates(
                source.uuid,
                groups,
                candidate_limit=1,
                minimum_similarity=0.0,
            )
            assert {
                (candidate.left_group_uuid, candidate.right_group_uuid)
                for candidate in candidates
            } == {
                (entities[0].uuid, entities[2].uuid),
                (entities[2].uuid, entities[3].uuid),
            }
            candidate = next(
                candidate
                for candidate in candidates
                if candidate.left_group_uuid == entities[0].uuid
            )
            assert candidate.left_uuid in {
                entities[0].uuid,
                entities[1].uuid,
            }
            assert candidate.right_uuid == entities[2].uuid
            assert (
                candidate.right_evidence.target_blocks[0].content
                == blocks[2].content
            )

            hubs = [
                SourceEntityHub(uuid=f'{prefix}-matrix-hub'),
                SourceEntityHub(uuid=f'{prefix}-determinant-hub'),
                SourceEntityHub(uuid=f'{prefix}-eigenvalue-hub'),
            ]
            await repository.replace_source_entity_hubs(
                source.uuid,
                hubs,
                [[member.uuid for member in group] for group in groups],
            )
            matches = await repository.search_source_entity_hubs(
                source.uuid, _embedding(1.0, 0.0, dimension), top_k=1
            )
            assert [
                (match.uuid, match.member_uuid, match.target_block_uuids)
                for match in matches
            ] == [(hubs[0].uuid, entities[1].uuid, [blocks[1].uuid])]
        finally:
            for fixture in [source, foreign_source]:
                await graph_repository.replace_source(fixture, [])
                async with database.session() as session:
                    await (
                        await session.run(
                            'MATCH (source:Source {uuid: $uuid}) DETACH DELETE source',
                            uuid=fixture.uuid,
                        )
                    ).consume()
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS2_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS2 Neo4j integration environment is not enabled',
)
def test_kms2_statement_procedure_source_scope_uses_block_ownership():
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        prefix = f'kms2-semantic-scope-it-{uuid4().hex}'
        source = Source(
            uuid=f'{prefix}-source',
            key=f'{prefix}-source.pdf',
            metadata={'suite': 'kms2'},
        )
        other_source = Source(
            uuid=f'{prefix}-other-source',
            key=f'{prefix}-other-source.pdf',
            metadata={'suite': 'kms2'},
        )
        blocks = [
            SourceBlock(
                uuid=f'{prefix}-block-{index}',
                block_type='paragraph',
                content=f'Source block {index}.',
            )
            for index in range(3)
        ]
        source_pages = [SourcePage(index=0, blocks=blocks[:2])]
        other_pages = [SourcePage(index=0, blocks=blocks[2:])]
        source_graph_repository = SourceGraphRepository(database.session)
        statement_repository = SourceStatementRepository(database.session)
        procedure_repository = SourceProcedureRepository(database.session)
        dimension = settings.local_models.embedding.model.dimension
        source_statements = [
            StatementDraft(
                uuid=f'{prefix}-source-statement-{index}',
                content=f'A real square matrix statement {index}.',
                embedding=_embedding(1.0, index / 100, dimension),
                member_block_uuids=[block.uuid for block in blocks[:index]],
            )
            for index in (1, 2)
        ]
        other_statements = [
            StatementDraft(
                uuid=f'{prefix}-other-statement',
                content='A real square matrix statement from another source.',
                embedding=_embedding(1.0, 0.0, dimension),
                member_block_uuids=[blocks[2].uuid],
            )
        ]
        source_procedures = [
            ProcedureDraft(
                uuid=f'{prefix}-source-procedure-{index}',
                content=f'Procedure step sequence {index}.',
                embedding=_embedding(1.0, index / 100, dimension),
                member_block_uuids=[block.uuid for block in blocks[:index]],
            )
            for index in (1, 2)
        ]
        other_procedures = [
            ProcedureDraft(
                uuid=f'{prefix}-other-procedure',
                content='Procedure step sequence from another source.',
                embedding=_embedding(1.0, 0.0, dimension),
                member_block_uuids=[blocks[2].uuid],
            )
        ]
        source_statement_hub = SourceStatementHub(
            uuid=f'{prefix}-source-statement-hub',
            canonical_name='Source statement',
            description='A source-local statement group.',
            embedding=_embedding(1.0, 0.0, dimension),
        )
        other_statement_hub = SourceStatementHub(
            uuid=f'{prefix}-other-statement-hub',
            canonical_name='Other statement',
            description='A different source-local statement group.',
            embedding=_embedding(1.0, 0.0, dimension),
        )
        source_procedure_hub = SourceProcedureHub(
            uuid=f'{prefix}-source-procedure-hub',
            canonical_name='Source procedure',
            description='A source-local procedure group.',
            embedding=_embedding(1.0, 0.0, dimension),
        )
        other_procedure_hub = SourceProcedureHub(
            uuid=f'{prefix}-other-procedure-hub',
            canonical_name='Other procedure',
            description='A different source-local procedure group.',
            embedding=_embedding(1.0, 0.0, dimension),
        )

        async def clean_fixture(fixture: Source) -> None:
            await source_graph_repository.replace_source(fixture, [])
            async with database.session() as session:
                await (
                    await session.run(
                        'MATCH (source:Source {uuid: $uuid}) DETACH DELETE source',
                        uuid=fixture.uuid,
                    )
                ).consume()

        try:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=dimension,
            )
            for fixture, pages, statements, procedures in (
                (source, source_pages, source_statements, source_procedures),
                (
                    other_source,
                    other_pages,
                    other_statements,
                    other_procedures,
                ),
            ):
                await source_graph_repository.replace_source(fixture, pages)
                await source_graph_repository.replace_pedagogical_material(
                    fixture.uuid,
                    pages,
                    [],
                    statements,
                    procedures,
                )

            loaded_statements = (
                await statement_repository.load_source_statements(source.uuid)
            )
            loaded_procedures = (
                await procedure_repository.load_source_procedures(source.uuid)
            )
            assert [item.uuid for item in loaded_statements] == [
                item.uuid for item in source_statements
            ]
            assert [item.member_block_uuids for item in loaded_statements] == [
                [blocks[0].uuid],
                [blocks[0].uuid, blocks[1].uuid],
            ]
            assert [item.uuid for item in loaded_procedures] == [
                item.uuid for item in source_procedures
            ]
            assert [item.member_block_uuids for item in loaded_procedures] == [
                [blocks[0].uuid],
                [blocks[0].uuid, blocks[1].uuid],
            ]
            assert 'source_uuid' not in loaded_statements[0].model_dump()
            assert 'source_uuid' not in loaded_procedures[0].model_dump()

            statement_candidates = (
                await statement_repository.read_source_statement_hub_candidates(
                    source.uuid,
                    candidate_limit=3,
                    minimum_similarity=0.9,
                )
            )
            procedure_candidates = (
                await procedure_repository.read_source_procedure_hub_candidates(
                    source.uuid,
                    candidate_limit=3,
                    minimum_similarity=0.9,
                )
            )
            assert {
                (item.left_uuid, item.right_uuid)
                for item in statement_candidates
            } == {tuple(sorted(item.uuid for item in source_statements))}
            assert {
                (item.left_uuid, item.right_uuid)
                for item in procedure_candidates
            } == {tuple(sorted(item.uuid for item in source_procedures))}
            assert (
                await statement_repository.read_source_statement_hub_candidates(
                    other_source.uuid,
                    candidate_limit=3,
                    minimum_similarity=0.9,
                )
            ) == []
            assert (
                await procedure_repository.read_source_procedure_hub_candidates(
                    other_source.uuid,
                    candidate_limit=3,
                    minimum_similarity=0.9,
                )
            ) == []
            await statement_repository.replace_source_statement_accepted_edges(
                source.uuid, statement_candidates
            )
            await procedure_repository.replace_source_procedure_accepted_edges(
                source.uuid, procedure_candidates
            )
            for fixture, expected in (
                (source, {item.uuid for item in source_statements}),
                (other_source, {item.uuid for item in other_statements}),
            ):
                communities = await statement_repository.detect_source_statement_communities(
                    fixture.uuid,
                    max_iterations=5,
                    min_association_strength=0.1,
                    minimum_community_size=1,
                )
                assert {
                    member.uuid
                    for community in communities
                    for member in community
                } == expected
            for fixture, expected in (
                (source, {item.uuid for item in source_procedures}),
                (other_source, {item.uuid for item in other_procedures}),
            ):
                communities = await procedure_repository.detect_source_procedure_communities(
                    fixture.uuid,
                    max_iterations=5,
                    min_association_strength=0.1,
                    minimum_community_size=1,
                )
                assert {
                    member.uuid
                    for community in communities
                    for member in community
                } == expected

            await statement_repository.replace_source_statement_hubs(
                source.uuid,
                [source_statement_hub],
                [[item.uuid for item in source_statements]],
            )
            await statement_repository.replace_source_statement_hubs(
                other_source.uuid,
                [other_statement_hub],
                [[item.uuid for item in other_statements]],
            )
            await procedure_repository.replace_source_procedure_hubs(
                source.uuid,
                [source_procedure_hub],
                [[item.uuid for item in source_procedures]],
            )
            await procedure_repository.replace_source_procedure_hubs(
                other_source.uuid,
                [other_procedure_hub],
                [[item.uuid for item in other_procedures]],
            )
            occurrence_uuids = [
                item.uuid
                for item in (
                    *source_statements,
                    *other_statements,
                    *source_procedures,
                    *other_procedures,
                )
            ]
            hub_uuids = [
                source_statement_hub.uuid,
                other_statement_hub.uuid,
                source_procedure_hub.uuid,
                other_procedure_hub.uuid,
            ]
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (node)
                    WHERE node.uuid IN $node_uuids
                    RETURN node.uuid AS uuid, keys(node) AS properties
                    """,
                    node_uuids=[*occurrence_uuids, *hub_uuids],
                )
                persisted_nodes = await result.data()
            assert {row['uuid'] for row in persisted_nodes} == set(
                occurrence_uuids + hub_uuids
            )
            assert all(
                'source_uuid' not in row['properties']
                for row in persisted_nodes
            )

            await source_graph_repository.replace_pedagogical_material(
                source.uuid, source_pages, [], [], []
            )
            assert (
                await statement_repository.load_source_statements(source.uuid)
                == []
            )
            assert (
                await procedure_repository.load_source_procedures(source.uuid)
                == []
            )
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (hub)
                    WHERE hub.uuid IN $hub_uuids
                    RETURN hub.uuid AS uuid
                    ORDER BY uuid
                    """,
                    hub_uuids=hub_uuids,
                )
                remaining_hubs = await result.data()
            assert [row['uuid'] for row in remaining_hubs] == sorted(
                [
                    other_statement_hub.uuid,
                    other_procedure_hub.uuid,
                ]
            )

            await source_graph_repository.replace_pedagogical_material(
                source.uuid,
                source_pages,
                [],
                source_statements,
                source_procedures,
            )
            await statement_repository.replace_source_statement_hubs(
                source.uuid,
                [source_statement_hub],
                [[item.uuid for item in source_statements]],
            )
            await procedure_repository.replace_source_procedure_hubs(
                source.uuid,
                [source_procedure_hub],
                [[item.uuid for item in source_procedures]],
            )
            replacement_block = SourceBlock(
                uuid=f'{prefix}-replacement-block',
                block_type='paragraph',
                content='Replacement source block.',
            )
            await source_graph_repository.replace_source(
                source,
                [SourcePage(index=1, blocks=[replacement_block])],
            )
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (hub)
                    WHERE hub.uuid IN $hub_uuids
                    RETURN hub.uuid AS uuid
                    ORDER BY uuid
                    """,
                    hub_uuids=hub_uuids,
                )
                remaining_hubs = await result.data()
            assert [row['uuid'] for row in remaining_hubs] == sorted(
                [
                    other_statement_hub.uuid,
                    other_procedure_hub.uuid,
                ]
            )
            assert {
                item.uuid
                for item in await statement_repository.load_source_statements(
                    other_source.uuid
                )
            } == {item.uuid for item in other_statements}
            assert {
                item.uuid
                for item in await procedure_repository.load_source_procedures(
                    other_source.uuid
                )
            } == {item.uuid for item in other_procedures}
        finally:
            await clean_fixture(source)
            await clean_fixture(other_source)
            await database.close()

    asyncio.run(exercise())
