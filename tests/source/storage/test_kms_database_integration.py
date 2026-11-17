"""Opt-in Neo4j coverage for v5 source publication and retrieval."""

import asyncio
import hashlib
import os
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PIL import Image

from kms.config.settings import Settings
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.inference.embedding.limits import VOYAGE_EMBEDDING_DIMENSION
from kms.source.ingestion import SourceIngestionService
from kms.source.models.inputs import CapturedPDFSource
from kms.source.models.snapshot import SourceSnapshotConflict
from kms.source.models.source import BlockType, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.processing.state import SourceProcessingState
from kms.source.retrieval.read import SourceReader
from kms.source.storage.artifacts import SourceArtifactStore
from kms.source.storage.retrieval import SourceRetrievalRepository
from kms.source.storage.snapshots import SourceSnapshotRepository

_REQUIRED_ENVIRONMENT = (
    'KMS_DATABASE__URI',
    'KMS_DATABASE__USERNAME',
    'KMS_DATABASE__PASSWORD',
    'KMS_DATABASE__DATABASE',
)


def _embedding(first: float, second: float, dimension: int) -> list[float]:
    return [first, second, *([0.0] * (dimension - 2))]


class _ProcessedGraph:
    def __init__(self, image_path: str, image_digest: str, vector: list[float]):
        self.image_path = image_path
        self.image_digest = image_digest
        self.vector = vector

    async def run(self, state: SourceProcessingState):
        def text(content: str) -> SourceBlock:
            return SourceBlock(
                block_type=BlockType.PARAGRAPH,
                content=content,
                embedding=self.vector,
            )

        def image() -> SourceBlock:
            return SourceBlock(
                block_type=BlockType.IMAGE,
                content=None,
                embedding=self.vector,
                image_path=self.image_path,
                image_sha256=self.image_digest,
            )

        units = [
            SourceUnit(
                kind=SourceUnitKind.EXERCISE,
                embedding=self.vector,
                blocks=[
                    text('Use the shared diagram.'),
                    image(),
                    text('Calculate the determinant.'),
                    text(' from the given matrix.'),
                ],
            ),
            SourceUnit(
                kind=SourceUnitKind.EXERCISE,
                embedding=self.vector,
                blocks=[
                    text('Use the shared diagram.'),
                    image(),
                    text('Find the inverse.'),
                ],
            ),
            SourceUnit(
                kind=SourceUnitKind.PROCEDURE,
                embedding=self.vector,
                blocks=[text('Compute the determinant by row reduction.')],
            ),
        ]
        source = state.source.model_copy(
            update={
                'metadata': {
                    **state.source.metadata,
                    'embedding_profile': {
                        'model': 'integration-test',
                        'dimension': len(self.vector),
                    },
                }
            }
        )
        return SimpleNamespace(
            source=source,
            units=units,
            selected_page_indexes=[0, 1],
        )


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_graph_ingestion_publication_and_unit_retrieval_round_trip(tmp_path):
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        user_uuid = str(uuid4())
        other_user_uuid = str(uuid4())
        dimension = VOYAGE_EMBEDDING_DIMENSION
        vector = _embedding(1.0, 0.0, dimension)
        profile_model = 'integration-test'
        image_source_path = tmp_path / 'shared-diagram.png'
        Image.new('RGB', (8, 8), (0, 0, 255)).save(
            image_source_path, format='PNG'
        )
        image_bytes = image_source_path.read_bytes()
        image_digest = hashlib.sha256(image_bytes).hexdigest()
        snapshots = SourceSnapshotRepository(database.session)
        retrieval = SourceRetrievalRepository(database.session)
        reader = SourceReader(retrieval)
        artifact_store = SourceArtifactStore(tmp_path / 'artifacts')
        source_uuid: str | None = None
        published = None
        ingestion = SourceIngestionService(
            snapshots,
            artifact_store,
            _ProcessedGraph(str(image_source_path), image_digest, vector),
            embedding_model=profile_model,
            embedding_dimension=dimension,
        )
        try:
            await schema.ensure_schema(
                database.session, embedding_dimension=dimension
            )
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:User {uuid: $uuid}), (:User {uuid: $other_uuid})',
                    uuid=user_uuid,
                    other_uuid=other_user_uuid,
                )
                await result.consume()

            capture = CapturedPDFSource(
                url='https://example.test/matrix.pdf',
                title='matrix.pdf',
                retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
                capture_id=f'capture-{uuid4()}',
                data=b'%PDF-1.7\nsynthetic integration capture',
            )
            published = await ingestion.ingest(user_uuid, capture)
            reused = await asyncio.gather(
                snapshots.create_owned_snapshot(user_uuid, published),
                snapshots.create_owned_snapshot(user_uuid, published),
            )
            assert reused == [published, published]
            source_uuid = published.source.uuid
            assert (
                published.source.metadata['identity_schema']
                == 'source-content-v5'
            )
            assert (
                published.units[0].blocks[1].uuid
                != published.units[1].blocks[1].uuid
            )
            assert (
                published.units[0].blocks[1].image_path
                == published.units[1].blocks[1].image_path
            )
            assert published.units[0].blocks[1].image_sha256 == image_digest

            loaded = await snapshots.load_owned_snapshot(user_uuid, source_uuid)
            assert loaded == published
            assert (
                await snapshots.load_owned_snapshot(
                    other_user_uuid, source_uuid
                )
                is None
            )
            assert [unit.kind for unit in loaded.units] == [
                SourceUnitKind.EXERCISE,
                SourceUnitKind.EXERCISE,
                SourceUnitKind.PROCEDURE,
            ]
            assert [block.content for block in loaded.units[0].blocks[:1]] == [
                'Use the shared diagram.'
            ]
            assert (
                ''.join(
                    block.content or '' for block in loaded.units[0].blocks[2:]
                )
                == 'Calculate the determinant. from the given matrix.'
            )
            assert loaded.units[0].blocks[1].image_sha256 == image_digest
            assert (
                loaded.units[0].blocks[1].image_path
                == loaded.units[1].blocks[1].image_path
            )

            block = loaded.units[0].blocks[1]
            image_reads = await reader.read_images(user_uuid, [block.uuid])
            assert [item.block_uuid for item in image_reads] == [block.uuid]
            assert image_reads[0].data == image_bytes
            assert await reader.read_images(other_user_uuid, [block.uuid]) == []

            block_hits = await retrieval.search_owned_blocks(
                user_uuid, vector, candidate_limit=30
            )
            assert any(hit.block.uuid == block.uuid for hit in block_hits)
            assert all(hit.source.uuid == source_uuid for hit in block_hits)
            unit_hits = await retrieval.search_owned_units(
                user_uuid,
                vector,
                candidate_limit=10,
                kind=SourceUnitKind.EXERCISE,
            )
            assert [hit.unit.kind for hit in unit_hits] == [
                SourceUnitKind.EXERCISE,
                SourceUnitKind.EXERCISE,
            ]
            assert {hit.source.uuid for hit in unit_hits} == {source_uuid}

            exercise_uuid = loaded.units[1].uuid
            neighbors = await reader.read_units(
                user_uuid, [exercise_uuid], after_units=1
            )
            assert [item.unit.uuid for item in neighbors] == [
                loaded.units[1].uuid,
                loaded.units[2].uuid,
            ]
            assert [item.unit.kind for item in neighbors] == [
                SourceUnitKind.EXERCISE,
                SourceUnitKind.PROCEDURE,
            ]
            assert (
                await reader.read_units(other_user_uuid, [exercise_uuid]) == []
            )

            # A failed first publication must roll back its source and children.
            bad = published.model_copy(
                update={
                    'source': published.source.model_copy(
                        update={
                            'uuid': str(uuid4()),
                            'metadata': {
                                **published.source.metadata,
                                'processed_digest': '0' * 64,
                            },
                        }
                    )
                }
            )
            with pytest.raises(SourceSnapshotConflict):
                await snapshots.create_owned_snapshot(user_uuid, bad)
            assert (
                await snapshots.load_owned_snapshot(user_uuid, bad.source.uuid)
                is None
            )
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    UNWIND $source_uuids AS source_uuid
                    MATCH (source:Source {uuid: source_uuid})
                    OPTIONAL MATCH (source)-[:HAS_UNIT]->(unit:SourceUnit)
                    OPTIONAL MATCH (unit)-[:HAS_BLOCK]->(block:SourceBlock)
                    DETACH DELETE block, unit, source
                    """,
                    source_uuids=[source_uuid] if source_uuid else [],
                )
                await result.consume()
                result = await session.run(
                    'MATCH (user:User) WHERE user.uuid IN $uuids DETACH DELETE user',
                    uuids=[user_uuid, other_user_uuid],
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())
