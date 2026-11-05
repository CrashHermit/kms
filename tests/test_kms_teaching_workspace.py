"""Session workspace persistence and native-image rehydration."""

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kms.core.model.block import SourceBlock
from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.source_evidence import (
    SourceEvidenceBlock,
    SourceImage,
    SourcePassage,
    SourceProcedurePassage,
    SourceStatementPassage,
)
from kms.core.model.source_processing.admission import (
    CapturedPDFSource,
    SourceSnapshotPreparation,
)
from kms.core.model.source_processing.input import PDFSourceInput
from kms.core.model.source_semantic.source_procedure import SourceProcedure
from kms.core.model.source_semantic.source_statement import SourceStatement
from kms.core.model.source_workspace import (
    SourceWorkspaceBlockEntry,
    SourceWorkspaceImageEntry,
)
from kms.core.model.teaching.observation import (
    LearnerObservation,
    LearnerObservationKind,
    LearnerObservationRecord,
)
from kms.core.model.teaching.workspace import (
    ImageInspection,
    TeachingArgumentRegistryEntry,
    TeachingEvidenceWorkspace,
    TeachingLearnerObservationRegistryEntry,
    TeachingProcedureRegistryEntry,
    TeachingStatementRegistryEntry,
)
from kms.core.model.visual_asset import VisualAsset
from kms.service.source_admission import SourceAdmissionService
from kms.service.source_artifacts import SourceArtifactStore
from kms.service.teaching_workspace import (
    restore_workspace,
    serialize_workspace,
)


class _GraphEvidence:
    def __init__(self, source, pages):
        self.source = source
        self.pages = pages

    async def read_passages(
        self, user_uuid, block_uuids, *, expand_memberships, before, after
    ):
        blocks = [
            SourceEvidenceBlock(block=block, page_indexes=[page.index])
            for page in self.pages
            for block in page.blocks
            if block.uuid in block_uuids
        ]
        return [
            SourcePassage(
                source=self.source,
                selected_block_uuids=block_uuids,
                blocks=blocks,
                groups=[],
            )
        ]

    async def read_images_at_revisions(self, user_uuid, references):
        from kms.service.source_artifacts import read_asset_bytes

        assets = {
            asset.uuid: asset
            for page in self.pages
            for block in page.blocks
            for asset in block.assets
        }
        if any(asset_uuid not in assets for _, _, asset_uuid in references):
            raise ValueError(
                'Requested source image revision is no longer available'
            )
        return [
            SourceImage(
                asset=assets[asset_uuid],
                data=read_asset_bytes(assets[asset_uuid]),
            )
            for _, _, asset_uuid in references
        ]

    async def read_statements(self, user_uuid, uuids):
        return []

    async def read_procedures(self, user_uuid, uuids):
        return []


class _GraphLearnerObservations:
    def __init__(self, records=()):
        self.records = {record.observation.uuid: record for record in records}
        self.calls = []

    async def read(self, user_uuid, uuids):
        self.calls.append((user_uuid, list(uuids)))
        return [self.records[uuid] for uuid in uuids]


def test_workspace_saves_references_and_restores_graph_evidence(tmp_path):
    image_bytes = b'original-image-bytes'
    image_path = tmp_path / 'figure.png'
    image_path.write_bytes(image_bytes)
    source = Source(uuid='source', revision=3, key='source')
    asset = VisualAsset(
        path=str(image_path),
        content_sha256=hashlib.sha256(image_bytes).hexdigest(),
    )
    text = SourceBlock(block_type='paragraph', content='private text')
    image = SourceBlock(block_type='image', content='caption', assets=[asset])
    pages = [SourcePage(index=7, blocks=[text, image])]
    workspace = TeachingEvidenceWorkspace(
        block_registry=[
            SourceWorkspaceBlockEntry(
                source=source, page_indexes=[7], block=text
            ),
            SourceWorkspaceBlockEntry(
                source=source, page_indexes=[7], block=image
            ),
        ],
        image_registry=[
            SourceWorkspaceImageEntry(
                asset=asset,
                source=source,
                block_positions=[1],
                data=image_bytes,
            )
        ],
        argument_registry=[
            TeachingArgumentRegistryEntry(
                uuid='argument-1', preview='bounded argument preview'
            )
        ],
        discovered_argument_positions=[0],
        read_image_positions={0},
        inspected_image_positions={0},
        active_image_positions=[],
        receipts=[
            ImageInspection(
                source=source,
                asset=asset,
                block_uuids=[image.uuid],
                data=image_bytes,
            )
        ],
    )
    saved = serialize_workspace(workspace)
    wire = saved.model_dump_json()
    assert 'private text' not in wire
    assert str(image_path) not in wire
    assert asset.content_sha256 not in wire
    assert image_bytes.decode() not in wire
    assert saved.argument_registry[0].preview == 'bounded argument preview'

    restored = asyncio.run(
        restore_workspace(
            saved,
            user_uuid='user',
            source_evidence=_GraphEvidence(source, pages),
            learner_observations=_GraphLearnerObservations(),
        )
    )
    assert [entry.block.uuid for entry in restored.block_registry] == [
        text.uuid,
        image.uuid,
    ]
    assert restored.image_registry[0].data == image_bytes
    assert restored.active_image_positions == []
    assert restored.receipts[0].asset.uuid == asset.uuid
    assert restored.receipts[0].data == image_bytes
    assert restored.argument_registry[0].preview == 'bounded argument preview'


def test_stale_source_discovery_is_pruned_without_losing_other_positions():
    stale_source = Source(uuid='stale', revision=1, key='stale')
    current_stale_source = stale_source.model_copy(update={'revision': 2})
    good_source = Source(uuid='good', revision=4, key='good')
    stale_block = SourceBlock(
        uuid='before-split',
        block_type='paragraph',
        content='Exercise includes two parts.',
    )
    good_block = SourceBlock(
        uuid='good-block',
        block_type='paragraph',
        content='Unchanged evidence.',
    )
    pages = [SourcePage(index=2, blocks=[good_block])]

    class _MixedGraphEvidence(_GraphEvidence):
        async def read_passages(
            self,
            user_uuid,
            block_uuids,
            *,
            expand_memberships,
            before,
            after,
        ):
            if stale_block.uuid in block_uuids:
                return [
                    SourcePassage(
                        source=current_stale_source,
                        selected_block_uuids=block_uuids,
                        blocks=[],
                        groups=[],
                    )
                ]
            return [
                SourcePassage(
                    source=good_source,
                    selected_block_uuids=block_uuids,
                    blocks=[
                        SourceEvidenceBlock(block=good_block, page_indexes=[2])
                    ],
                    groups=[],
                )
            ]

    saved = serialize_workspace(
        TeachingEvidenceWorkspace(
            block_registry=[
                SourceWorkspaceBlockEntry(
                    source=stale_source,
                    page_indexes=[0],
                    block=stale_block,
                ),
                SourceWorkspaceBlockEntry(
                    source=good_source, page_indexes=[2], block=good_block
                ),
            ],
            active_block_positions=[1],
            discovered_block_positions=[0, 1],
            read_block_positions={1},
        )
    )
    restored = asyncio.run(
        restore_workspace(
            saved,
            user_uuid='user',
            source_evidence=_MixedGraphEvidence(good_source, pages),
            learner_observations=_GraphLearnerObservations(),
        )
    )

    assert [entry.block.uuid for entry in restored.block_registry] == [
        good_block.uuid
    ]
    assert restored.active_block_positions == [0]
    assert restored.discovered_block_positions == [0]
    assert restored.read_block_positions == {0}


def test_restore_propagates_missing_graph_image(tmp_path):
    source = Source(uuid='source', revision=1, key='source')
    asset = VisualAsset(
        path=str(tmp_path / 'missing.png'),
        content_sha256=hashlib.sha256(b'bytes').hexdigest(),
    )
    workspace = TeachingEvidenceWorkspace(
        image_registry=[
            SourceWorkspaceImageEntry(asset=asset, source=source, data=b'bytes')
        ],
        inspected_image_positions={0},
        receipts=[
            ImageInspection(
                source=source, asset=asset, block_uuids=[], data=b'bytes'
            )
        ],
    )
    with pytest.raises(
        ValueError,
        match='Requested source image revision is no longer available',
    ):
        asyncio.run(
            restore_workspace(
                serialize_workspace(workspace),
                user_uuid='user',
                source_evidence=_GraphEvidence(source, []),
                learner_observations=_GraphLearnerObservations(),
            )
        )


def test_none_workspace_restores_empty_without_graph_reads():
    workspace = asyncio.run(
        restore_workspace(
            None,
            user_uuid='user',
            source_evidence=_GraphEvidence(None, []),
            learner_observations=_GraphLearnerObservations(),
        )
    )

    assert workspace.block_registry == []
    assert workspace.image_registry == []


def test_restore_rehydrates_only_previously_read_domains():
    source = Source(uuid='source', revision=2, key='source')
    block = SourceBlock(block_type='paragraph', content='passage')
    page_passage = SourcePassage(
        source=source,
        selected_block_uuids=[block.uuid],
        blocks=[SourceEvidenceBlock(block=block, page_indexes=[4])],
        groups=[],
    )
    statement = SourceStatement(
        uuid='statement-read',
        member_block_uuids=[block.uuid],
        content='statement',
        embedding=[0.1],
    )
    unread_statement = SourceStatement(
        uuid='statement-unread',
        member_block_uuids=[block.uuid],
        content='unread statement',
        embedding=[0.2],
    )
    procedure = SourceProcedure(
        uuid='procedure-read',
        member_block_uuids=[block.uuid],
        content='procedure',
        embedding=[0.3],
    )

    class SemanticEvidence(_GraphEvidence):
        statement_requests = []
        procedure_requests = []

        async def read_statements(self, user_uuid, uuids):
            self.statement_requests.append(uuids)
            return [
                SourceStatementPassage(
                    statement=statement, passages=[page_passage]
                )
            ]

        async def read_procedures(self, user_uuid, uuids):
            self.procedure_requests.append(uuids)
            return [
                SourceProcedurePassage(
                    procedure=procedure, passages=[page_passage]
                )
            ]

    evidence = SemanticEvidence(
        source, [SourcePage(index=4, markdown='passage', blocks=[block])]
    )
    workspace = TeachingEvidenceWorkspace(
        statement_registry=[
            TeachingStatementRegistryEntry(
                source=source,
                page_indexes=[4],
                statement=statement,
                passage=SourceStatementPassage(
                    statement=statement, passages=[page_passage]
                ),
            ),
            TeachingStatementRegistryEntry(
                source=source,
                page_indexes=[4],
                statement=unread_statement,
            ),
        ],
        procedure_registry=[
            TeachingProcedureRegistryEntry(
                source=source,
                page_indexes=[4],
                procedure=procedure,
                passage=SourceProcedurePassage(
                    procedure=procedure, passages=[page_passage]
                ),
            )
        ],
    )

    saved = serialize_workspace(workspace)
    assert [entry.was_read for entry in saved.statement_registry] == [
        True,
        False,
    ]
    restored = asyncio.run(
        restore_workspace(
            saved,
            user_uuid='user',
            source_evidence=evidence,
            learner_observations=_GraphLearnerObservations(),
        )
    )

    assert evidence.statement_requests == [['statement-read']]
    assert evidence.procedure_requests == [['procedure-read']]
    assert restored.statement_registry[0].passage is not None
    assert restored.statement_registry[1].passage is None
    assert restored.procedure_registry[0].passage is not None


def test_admission_prepares_pdf_capture_for_standard_processing(tmp_path):
    class Repository:
        async def load_owned_snapshot(self, user_uuid, source_uuid):
            return None

    class ProcessingGraph:
        async def run(self, state):
            raise AssertionError('preparation does not process')

    async def prepare():
        admission = SourceAdmissionService(
            Repository(),
            SourceArtifactStore(tmp_path / 'store'),
            ProcessingGraph(),
            embedding_model='test/model',
            embedding_dimension=4,
        )
        capture = CapturedPDFSource(
            url='https://example.org/doubling.pdf',
            resolved_url='https://cdn.example.org/doubling.pdf',
            title='Doubling',
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            data=b'original-pdf',
        )
        return (
            capture,
            await admission.prepare('user', capture),
            admission,
        )

    capture, preparation, admission = asyncio.run(prepare())
    assert isinstance(preparation, SourceSnapshotPreparation)
    assert isinstance(preparation.input, PDFSourceInput)
    assert Path(preparation.input.pdf_path).read_bytes() == capture.data
    metadata = preparation.source.metadata
    assert metadata['url'] == capture.url
    assert metadata['resolved_url'] == capture.resolved_url
    assert metadata['capture_id'] == capture.capture_id
    assert 'captures' not in metadata
    assert not (tmp_path / 'store').exists()
    admission.cleanup(capture.capture_id)


def test_learner_observation_workspace_persists_references_and_reloads_once():
    now = datetime(2026, 10, 1, tzinfo=UTC)
    read_record = LearnerObservationRecord(
        observation=LearnerObservation(
            uuid='read-observation',
            kind=LearnerObservationKind.GAP,
            topic='vectors',
            description='Needs help with direction.',
            embedding=[0.1],
            created_at=now,
        ),
        evidence=[],
    )
    unread_record = LearnerObservationRecord(
        observation=LearnerObservation(
            uuid='unread-observation',
            kind=LearnerObservationKind.MISCONCEPTION,
            topic='matrices',
            description='Confuses columns and rows.',
            embedding=[0.2],
            created_at=now,
        ),
        evidence=[],
    )
    workspace = TeachingEvidenceWorkspace(
        learner_observation_registry=[
            TeachingLearnerObservationRegistryEntry(
                observation=read_record.observation,
                record=read_record,
            ),
            TeachingLearnerObservationRegistryEntry(
                observation=unread_record.observation,
            ),
        ],
        active_learner_observation_positions=[0],
        discovered_learner_observation_positions=[0, 1],
    )

    saved = serialize_workspace(workspace)
    saved_registry = [
        entry.model_dump() for entry in saved.learner_observation_registry
    ]
    assert saved_registry == [
        {'uuid': 'read-observation', 'was_read': True},
        {'uuid': 'unread-observation', 'was_read': False},
    ]
    assert read_record.observation.description not in saved.model_dump_json()
    reader = _GraphLearnerObservations([read_record, unread_record])
    restored = asyncio.run(
        restore_workspace(
            saved,
            user_uuid='user',
            source_evidence=_GraphEvidence(None, []),
            learner_observations=reader,
        )
    )

    assert reader.calls == [
        ('user', ['read-observation', 'unread-observation'])
    ]
    assert restored.learner_observation_registry[0].record == read_record
    assert restored.learner_observation_registry[1].record is None
    assert (
        restored.learner_observation_registry[0].observation
        == read_record.observation
    )
    assert (
        restored.learner_observation_registry[1].observation
        == unread_record.observation
    )
    assert restored.active_learner_observation_positions == [0]
    assert restored.discovered_learner_observation_positions == [0, 1]
