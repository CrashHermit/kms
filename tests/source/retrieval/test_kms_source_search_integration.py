"""Opt-in Neo4j coverage for owned block and unit retrieval."""

import asyncio
import hashlib
import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from PIL import Image

from kms.config.inference import ContextWindowSettings
from kms.config.settings import Settings
from kms.core.embedding import EmbeddingRequest
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.inference.embedding.limits import VOYAGE_EMBEDDING_DIMENSION
from kms.media.budget import ImagePayloadBudget
from kms.source.ingestion import SourceIngestionService
from kms.source.models.inputs import CapturedPDFSource
from kms.source.models.snapshot import finalize_snapshot_identities
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.processing.block_embedding import SourceBlockEmbeddingNode
from kms.source.processing.correction.node import ContentCorrectionNode
from kms.source.processing.exercise_finder.node import ExerciseFinderNode
from kms.source.processing.exercise_finder.settings import (
    ExerciseFinderSettings,
)
from kms.source.processing.exercise_splitter.node import ExerciseSplitterNode
from kms.source.processing.formatting.node import FormattingNode
from kms.source.processing.graph import SourceProcessingGraph
from kms.source.processing.image_filter.models import ImageFilterDecision
from kms.source.processing.image_filter.node import ImageFilterNode
from kms.source.processing.image_seam.node import ImageSeamNode
from kms.source.processing.instruction_redistribution.node import (
    InstructionRedistributionNode,
)
from kms.source.processing.instruction_redistribution.settings import (
    InstructionRedistributionSettings,
)
from kms.source.processing.ocr import OCRNode
from kms.source.processing.ocr_models import OCRArtifact, OCRExtraction
from kms.source.processing.procedure_finder.node import ProcedureFinderNode
from kms.source.processing.procedure_finder.settings import (
    ProcedureFinderSettings,
)
from kms.source.processing.procedure_splitter.node import ProcedureSplitterNode
from kms.source.processing.text_seam.node import TextSeamNode
from kms.source.processing.unit_assembly import SourceUnitAssemblyNode
from kms.source.processing.unit_embedding import SourceUnitEmbeddingNode
from kms.source.storage.artifacts import SourceArtifactStore
from kms.source.storage.retrieval import SourceRetrievalRepository
from kms.source.storage.snapshots import SourceSnapshotRepository

_REQUIRED_ENVIRONMENT = (
    'KMS_DATABASE__URI',
    'KMS_DATABASE__USERNAME',
    'KMS_DATABASE__PASSWORD',
    'KMS_DATABASE__DATABASE',
)


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _GraphOCR:
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path

    def extract(self, pdf_path, *, source_uuid, pages=None):
        return OCRExtraction(
            blocks=[
                OCRArtifact(
                    page_index=0,
                    block_index=0,
                    block_type=BlockType.PARAGRAPH,
                    content='Use the shared diagram.',
                    crop_path=self.image_path,
                    crop_bbox=(0, 0, 8, 8),
                ),
                OCRArtifact(
                    page_index=0,
                    block_index=1,
                    block_type=BlockType.IMAGE,
                    crop_path=self.image_path,
                    crop_bbox=(0, 0, 8, 8),
                ),
                OCRArtifact(
                    page_index=0,
                    block_index=2,
                    block_type=BlockType.PARAGRAPH,
                    content='Find the determinant. Proof: row reduction. Extra explanation.',
                    crop_path=self.image_path,
                    crop_bbox=(0, 0, 8, 8),
                ),
                OCRArtifact(
                    page_index=1,
                    block_index=0,
                    block_type=BlockType.PARAGRAPH,
                    content='Compute by row reduction.',
                    crop_path=self.image_path,
                    crop_bbox=(0, 0, 8, 8),
                ),
            ],
            page_indexes=[0, 1],
            page_images={0: self.image_path, 1: self.image_path},
        )


class _GraphDecisions:
    def __init__(self, role):
        self.role = role

    async def acall(self, **kwargs):
        from kms.source.processing.correction.models import (
            ContentCorrectionOutput,
        )
        from kms.source.processing.exercise_finder.models import (
            ExerciseBoundaryDecision,
            ExerciseStartDecision,
        )
        from kms.source.processing.exercise_splitter.models import (
            SplitDecision,
            SplitPiece,
            SplitRouteDecision,
        )
        from kms.source.processing.formatting.models import FormattingOutput
        from kms.source.processing.instruction_redistribution.models import (
            InstructionApplicabilityDecision,
            InstructionBoundaryDecision,
            InstructionStartDecision,
        )
        from kms.source.processing.procedure_finder.models import (
            ProcedureBoundaryDecision,
            ProcedureStartDecision,
        )
        from kms.source.processing.procedure_splitter.models import (
            ProcedureSplitRouteDecision,
        )
        from kms.source.processing.text_seam.models import TextSeamDecision

        role = self.role
        if role == 'correction':
            return [
                ContentCorrectionOutput(
                    position=item.position,
                    keep=True,
                    corrected_content=item.material.text,
                    cleanup_instructions=[],
                )
                for item in kwargs['blocks']
            ]
        if role == 'formatting':
            return [
                FormattingOutput(
                    position=item.position, formatted_content=item.content
                )
                for item in kwargs['blocks']
            ]
        if role == 'text_seam':
            return [
                TextSeamDecision(position=item.position, is_split=False)
                for item in kwargs['items']
            ]
        if role in ('exercise_router', 'procedure_router'):
            return [
                (
                    SplitRouteDecision
                    if role == 'exercise_router'
                    else ProcedureSplitRouteDecision
                )(
                    position=item.position,
                    requires_split=role == 'exercise_router'
                    and (item.source_block.material.text or '').startswith(
                        'Find'
                    ),
                )
                for item in kwargs['targets']
            ]
        if role == 'exercise_split':
            return [
                SplitDecision(
                    position=item.position,
                    pieces=[
                        SplitPiece(content='Find the determinant. '),
                        SplitPiece(content='Proof: row reduction. '),
                        SplitPiece(content='Extra explanation.'),
                    ],
                )
                for item in kwargs['candidates']
            ]
        if role == 'applicability':
            return [
                InstructionApplicabilityDecision(
                    position=item.position, applies=True
                )
                for item in kwargs['items']
            ]
        items = kwargs['items']

        def text(item):
            return item.target_block.material.text or ''

        if role == 'instruction_start':
            return [
                InstructionStartDecision(
                    position=item.position,
                    is_instruction_start=text(item)
                    == 'Use the shared diagram.',
                )
                for item in items
            ]
        if role == 'instruction_boundary':
            return [
                InstructionBoundaryDecision(
                    position=item.position,
                    is_instruction_boundary=text(item).startswith('Find'),
                )
                for item in items
            ]
        if role == 'exercise_start':
            return [
                ExerciseStartDecision(
                    position=item.position,
                    is_exercise_start=text(item).startswith('Find'),
                )
                for item in items
            ]
        if role == 'exercise_boundary':
            return [
                ExerciseBoundaryDecision(
                    position=item.position,
                    is_exercise_boundary=text(item).startswith('Proof'),
                )
                for item in items
            ]
        if role == 'procedure_start':
            return [
                ProcedureStartDecision(
                    position=item.position,
                    is_procedure_start=text(item).startswith(
                        ('Proof', 'Compute')
                    ),
                )
                for item in items
            ]
        return [
            ProcedureBoundaryDecision(
                position=item.position,
                is_procedure_boundary=text(item).startswith('Extra'),
            )
            for item in items
        ]


class _KeepGraphImages:
    async def acall(self, *, items):
        return [
            ImageFilterDecision(position=item.position, keep=True)
            for item in items
        ]


class _SyntheticEmbedder:
    async def embed(self, request: EmbeddingRequest) -> list[list[float]]:
        return [
            [1.0, *([0.0] * (VOYAGE_EMBEDDING_DIMENSION - 1))]
            for _ in request.materials
        ]


def _write_png(path, color: str) -> bytes:
    Image.new('RGB', (3, 2), color).save(path)
    return path.read_bytes()


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_owned_block_reads_unit_neighborhoods_and_native_images(tmp_path):
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        repository = SourceRetrievalRepository(database.session)
        snapshots = SourceSnapshotRepository(database.session)
        first_source_uuid, second_source_uuid = sorted(
            [str(uuid4()), str(uuid4())], reverse=True
        )
        prefix = f'kms-source-search-it-{uuid4().hex}'
        owner = f'{prefix}-user'
        other = f'{prefix}-foreign-user'
        dimension = VOYAGE_EMBEDDING_DIMENSION
        profile = {'model': 'integration-test', 'dimension': dimension}
        image_path = tmp_path / 'shared.png'
        image_bytes = _write_png(image_path, 'blue')

        def block(block_type: BlockType, content: str, **kwargs):
            return SourceBlock(
                block_type=block_type,
                content=content,
                **kwargs,
            )

        first_blocks = [block(BlockType.PARAGRAPH, 'Opening')]
        exercise_blocks = [
            block(BlockType.PARAGRAPH, 'Shared setup'),
            block(
                BlockType.IMAGE,
                'Diagram',
                image_path=str(image_path),
                image_sha256=hashlib.sha256(image_bytes).hexdigest(),
            ),
            block(BlockType.PARAGRAPH, 'Exercise task'),
        ]
        procedure_blocks = [
            block(BlockType.PARAGRAPH, 'Step one'),
            block(BlockType.EQUATION, 'x = 2'),
        ]
        units = [
            SourceUnit(kind=SourceUnitKind.CONTENT, blocks=first_blocks),
            SourceUnit(kind=SourceUnitKind.EXERCISE, blocks=exercise_blocks),
            SourceUnit(kind=SourceUnitKind.PROCEDURE, blocks=procedure_blocks),
        ]
        snapshot = finalize_snapshot_identities(
            Source(
                uuid=first_source_uuid,
                key='mixed-source.pdf',
                metadata={
                    'embedding_profile': profile,
                },
            ),
            units,
        )
        second_snapshot = finalize_snapshot_identities(
            Source(
                uuid=second_source_uuid,
                key='second-owned.pdf',
                metadata={},
            ),
            [
                SourceUnit(
                    kind=SourceUnitKind.CONTENT,
                    blocks=[block(BlockType.PARAGRAPH, 'Second source')],
                )
            ],
        )
        foreign_block = block(BlockType.PARAGRAPH, 'Foreign text')
        foreign_snapshot = finalize_snapshot_identities(
            Source(
                uuid=str(uuid4()),
                key='foreign-source.pdf',
                metadata={},
            ),
            [SourceUnit(kind=SourceUnitKind.CONTENT, blocks=[foreign_block])],
        )
        source_uuids = [
            snapshot.source.uuid,
            second_snapshot.source.uuid,
            foreign_snapshot.source.uuid,
        ]
        unit_ids = [unit.uuid for unit in snapshot.units]
        second_unit_uuid = second_snapshot.units[0].uuid
        block_ids = [
            item.uuid for unit in snapshot.units for item in unit.blocks
        ]
        old_source_uuid = str(uuid4())
        source_uuids.append(old_source_uuid)
        old_unit_uuid = str(uuid4())
        old_block_uuid = str(uuid4())
        try:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=dimension,
            )
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:User {uuid: $owner}), (:User {uuid: $other})',
                    owner=owner,
                    other=other,
                )
                await result.consume()
            await snapshots.create_owned_snapshot(owner, second_snapshot)
            await snapshots.create_owned_snapshot(owner, snapshot)
            await snapshots.create_owned_snapshot(other, foreign_snapshot)
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (owner:User {uuid: $owner})
                    CREATE (owner)-[:OWNS_SOURCE]->(
                        :Source {
                            uuid: $source_uuid,
                            identity_schema: 'source-content-v4',
                            key: 'obsolete.pdf',
                            metadata: '{}',
                            revision: 0
                        }
                    )-[:HAS_UNIT]->(
                        :SourceUnit {
                            uuid: $unit_uuid,
                            kind: 'exercise',
                            embedding: $vector
                        }
                    )-[:HAS_BLOCK]->(
                        :SourceBlock {
                            uuid: $block_uuid,
                            block_type: 'paragraph',
                            content: 'obsolete',
                            embedding: $vector
                        }
                    )
                    """,
                    owner=owner,
                    source_uuid=old_source_uuid,
                    unit_uuid=old_unit_uuid,
                    block_uuid=old_block_uuid,
                    vector=[1.0, *([0.0] * (dimension - 1))],
                )
                await result.consume()

            requested_blocks = await repository.read_blocks(
                owner,
                [block_ids[1], block_ids[0], block_ids[1], 'unknown-block'],
            )
            assert [item.block.uuid for item in requested_blocks] == [
                block_ids[1],
                block_ids[0],
            ]
            assert [item.unit_uuid for item in requested_blocks] == [
                unit_ids[1],
                unit_ids[0],
            ]
            assert requested_blocks[0].source.uuid == snapshot.source.uuid
            assert requested_blocks[0].material.text == 'Shared setup'
            assert await repository.read_blocks(other, [block_ids[0]]) == []
            assert await repository.read_blocks(owner, []) == []
            assert await repository.read_blocks(owner, [old_block_uuid]) == []
            assert await repository.read_units(owner, [old_unit_uuid]) == []

            around_exercise = await repository.read_units(
                owner, [unit_ids[1]], before_units=1, after_units=1
            )
            assert [item.unit.uuid for item in around_exercise] == unit_ids
            exercise_read = around_exercise[1]
            assert exercise_read.unit.kind is SourceUnitKind.EXERCISE
            assert [block.content for block in exercise_read.unit.blocks] == [
                'Shared setup',
                'Diagram',
                'Exercise task',
            ]
            assert [part.type for part in exercise_read.material.parts] == [
                'text',
                'image',
                'text',
            ]
            assert exercise_read.material.images[0].key == (
                'path',
                str(image_path),
            )
            assert await repository.read_units(other, [unit_ids[1]]) == []
            assert await repository.read_units(owner, []) == []

            grouped = await repository.read_units(
                owner,
                [unit_ids[2], unit_ids[0]],
                before_units=0,
                after_units=0,
            )
            assert [item.unit.uuid for item in grouped] == [
                unit_ids[0],
                unit_ids[2],
            ]
            cross_source_order = await repository.read_units(
                owner,
                [unit_ids[1], second_unit_uuid],
                before_units=0,
                after_units=0,
            )
            assert [item.unit.uuid for item in cross_source_order] == [
                unit_ids[1],
                second_unit_uuid,
            ]
            assert [item.source.uuid for item in cross_source_order] == [
                snapshot.source.uuid,
                second_snapshot.source.uuid,
            ]
            endpoint_context = await repository.read_units(
                owner,
                [unit_ids[0]],
                before_units=2,
                after_units=2,
            )
            assert [item.unit.uuid for item in endpoint_context] == unit_ids
            overlapping = await repository.read_units(
                owner,
                [unit_ids[0], unit_ids[1]],
                before_units=1,
                after_units=1,
            )
            assert [item.unit.uuid for item in overlapping] == unit_ids
            source_end = await repository.read_units(
                owner, [unit_ids[-1]], after_units=1
            )
            assert [item.unit.uuid for item in source_end] == [unit_ids[-1]]
            isolated_source = await repository.read_units(
                owner,
                [second_unit_uuid],
                before_units=1,
                after_units=1,
            )
            assert [item.unit.uuid for item in isolated_source] == [
                second_unit_uuid
            ]

            images = await repository.load_images(owner, [block_ids[2]])
            assert len(images) == 1
            assert images[0].block_uuid == block_ids[2]
            assert (
                images[0].image_sha256
                == hashlib.sha256(image_bytes).hexdigest()
            )
            assert await repository.load_images(other, [block_ids[2]]) == []
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    UNWIND $source_uuids AS uuid
                    MATCH (source:Source {uuid: uuid})
                    OPTIONAL MATCH (source)-[:HAS_UNIT]->(unit:SourceUnit)
                    OPTIONAL MATCH (unit)-[:HAS_BLOCK]->(block:SourceBlock)
                    DETACH DELETE block, unit, source
                    """,
                    source_uuids=source_uuids,
                )
                await result.consume()
                result = await session.run(
                    'MATCH (user:User) WHERE user.uuid IN $users '
                    'DETACH DELETE user',
                    users=[owner, other],
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())


@pytest.mark.skipif(
    os.getenv('KMS_NEO4J_IT') != '1'
    or not all(os.getenv(name) for name in _REQUIRED_ENVIRONMENT),
    reason='KMS Neo4j integration environment is not enabled',
)
def test_compiled_graph_ingestion_and_owned_unit_navigation(tmp_path):
    async def exercise() -> None:
        settings = Settings()
        database = DatabaseClient(settings.database)
        dimension = VOYAGE_EMBEDDING_DIMENSION
        vector = [1.0, *([0.0] * (dimension - 1))]
        owner = f'kms-source-graph-it-{uuid4()}'
        source_uuid = None
        image_path = tmp_path / 'shared.png'
        image_bytes = _write_png(image_path, 'blue')
        snapshots = SourceSnapshotRepository(database.session)
        retrieval = SourceRetrievalRepository(database.session)

        counter = MaterialTokenEstimator(_Counter())
        budget = TokenBudget(counter=counter, token_limit=100_000)
        image_budget = ImagePayloadBudget(8, 26_214_400, 40_000_000)
        context = ContextWindowSettings(
            backward_budget=100, forward_budget=100, target_budget=0
        )
        graph = SourceProcessingGraph(
            OCRNode(_GraphOCR(str(image_path))),
            ContentCorrectionNode(
                _GraphDecisions('correction'),
                budget=budget,
                image_budget=image_budget,
                batch_size=8,
            ),
            FormattingNode(
                _GraphDecisions('formatting'), budget=budget, batch_size=8
            ),
            TextSeamNode(
                _GraphDecisions('text_seam'),
                _GraphDecisions('text_seam'),
                judge_budget=budget,
                rewrite_budget=budget,
                judge_batch_size=8,
                rewrite_batch_size=8,
            ),
            ImageSeamNode(
                _GraphDecisions('text_seam'),
                budget=budget,
                image_budget=image_budget,
                batch_size=8,
            ),
            ImageFilterNode(
                _KeepGraphImages(),
                context,
                budget=budget,
                image_budget=image_budget,
                batch_size=8,
                token_counters=(counter,),
            ),
            ExerciseSplitterNode(
                _GraphDecisions('exercise_router'),
                _GraphDecisions('exercise_split'),
                context,
                batch_size=8,
                token_counters=(counter,),
                image_budget=image_budget,
                router_budget=budget,
                splitter_budget=budget,
            ),
            ProcedureSplitterNode(
                _GraphDecisions('procedure_router'),
                _GraphDecisions('procedure_router'),
                context,
                batch_size=8,
                token_counters=(counter,),
                image_budget=image_budget,
                router_budget=budget,
                splitter_budget=budget,
            ),
            InstructionRedistributionNode(
                _GraphDecisions('instruction_start'),
                _GraphDecisions('instruction_boundary'),
                _GraphDecisions('applicability'),
                InstructionRedistributionSettings(),
                start_budget=budget,
                boundary_budget=budget,
                applicability_budget=budget,
                start_token_counters=(counter,),
                boundary_token_counters=(counter,),
                applicability_token_counters=(counter,),
                image_budget=image_budget,
                applicability_image_budget=image_budget,
            ),
            ExerciseFinderNode(
                _GraphDecisions('exercise_start'),
                _GraphDecisions('exercise_boundary'),
                ExerciseFinderSettings(),
                start_budget=budget,
                boundary_budget=budget,
                start_token_counters=(counter,),
                boundary_token_counters=(counter,),
                image_budget=image_budget,
            ),
            ProcedureFinderNode(
                _GraphDecisions('procedure_start'),
                _GraphDecisions('procedure_boundary'),
                ProcedureFinderSettings(),
                start_budget=budget,
                boundary_budget=budget,
                start_token_counters=(counter,),
                boundary_token_counters=(counter,),
                image_budget=image_budget,
            ),
            SourceUnitAssemblyNode(),
            SourceBlockEmbeddingNode(
                _SyntheticEmbedder(),
                model='integration-test',
                dimension=dimension,
            ),
            SourceUnitEmbeddingNode(_SyntheticEmbedder()),
        )
        ingestion = SourceIngestionService(
            snapshots,
            SourceArtifactStore(tmp_path / 'artifacts'),
            graph,
            embedding_model='integration-test',
            embedding_dimension=dimension,
        )
        try:
            await schema.ensure_schema(
                database.session, embedding_dimension=dimension
            )
            async with database.session() as session:
                result = await session.run(
                    'CREATE (:User {uuid: $owner})', owner=owner
                )
                await result.consume()
            capture = CapturedPDFSource(
                url='https://example.test/compiled-graph.pdf',
                title='compiled-graph.pdf',
                retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
                capture_id=f'capture-{uuid4()}',
                data=b'%PDF-1.7\ncompiled graph integration',
            )
            snapshot = await ingestion.ingest(owner, capture)
            source_uuid = snapshot.source.uuid
            assert [unit.kind for unit in snapshot.units] == [
                SourceUnitKind.EXERCISE,
                SourceUnitKind.PROCEDURE,
                SourceUnitKind.CONTENT,
                SourceUnitKind.PROCEDURE,
            ]
            exercise = snapshot.units[0]
            assert [block.content for block in exercise.blocks] == [
                'Use the shared diagram.',
                None,
                'Find the determinant. ',
            ]
            assert (
                exercise.blocks[1].image_sha256
                == hashlib.sha256(image_bytes).hexdigest()
            )
            assert (
                snapshot.units[1].blocks[0].content == 'Proof: row reduction. '
            )
            assert snapshot.units[2].blocks[0].content == 'Extra explanation.'
            assert (
                snapshot.units[3].blocks[0].content
                == 'Compute by row reduction.'
            )

            loaded_exercise = await retrieval.read_units(owner, [exercise.uuid])
            assert len(loaded_exercise) == 1
            assert [
                part.type for part in loaded_exercise[0].material.parts
            ] == [
                'text',
                'image',
                'text',
            ]
            assert loaded_exercise[0].material.images[0].key == (
                'path',
                exercise.blocks[1].image_path,
            )
            block_hits = await retrieval.search_owned_blocks(
                owner, vector, candidate_limit=30
            )
            assert any(hit.unit_uuid == exercise.uuid for hit in block_hits)
            unit_hits = await retrieval.search_owned_units(
                owner, vector, candidate_limit=30, kind=SourceUnitKind.EXERCISE
            )
            assert any(hit.unit.uuid == exercise.uuid for hit in unit_hits)
            following = await retrieval.read_units(
                owner, [exercise.uuid], after_units=1
            )
            assert [item.unit.kind for item in following] == [
                SourceUnitKind.EXERCISE,
                SourceUnitKind.PROCEDURE,
            ]
            assert (
                following[-1].unit.blocks[0].content == 'Proof: row reduction. '
            )
        finally:
            async with database.session() as session:
                result = await session.run(
                    """
                    MATCH (source:Source {uuid: $source_uuid})
                    OPTIONAL MATCH (source)-[:HAS_UNIT]->(unit:SourceUnit)
                    OPTIONAL MATCH (unit)-[:HAS_BLOCK]->(block:SourceBlock)
                    DETACH DELETE block, unit, source
                    """,
                    source_uuid=source_uuid or '',
                )
                await result.consume()
                result = await session.run(
                    'MATCH (user:User {uuid: $owner}) DETACH DELETE user',
                    owner=owner,
                )
                await result.consume()
            await database.close()

    asyncio.run(exercise())
