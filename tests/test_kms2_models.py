import uuid

import pytest
from pydantic import ValidationError

from kms2.core.model.base import (
    Edge,
    Vertex,
)
from kms2.core.model.block import SourceBlock
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.instruction import Instruction
from kms2.core.model.source_processing.ocr import (
    OCRArtifact,
    OCRImageArtifact,
)
from kms2.core.model.source_processing.pedagogical import (
    ProcedureDraft,
    StatementDraft,
)
from kms2.core.model.source_semantic.source_fact_extraction import (
    SourceFact,
    SourceFactContext,
    SourceFactTarget,
)
from kms2.core.model.visual_asset import VisualAsset


def test_vertex_generates_uuid4_and_rejects_extra_fields():
    assert Vertex(uuid='vertex-1').uuid == 'vertex-1'
    generated = [Vertex().uuid for _ in range(2)]
    assert generated[0] != generated[1]
    assert all(uuid.UUID(value).version == 4 for value in generated)

    with pytest.raises(ValidationError):
        Vertex(label='unexpected')


def test_source_fact_owns_node_first_evidence_pointers():
    target = SourceFactTarget(
        uuid='target-1',
        source_blocks=[SourceBlock(uuid='block-1', block_type='paragraph')],
    )
    before = SourceFactContext(uuid='before-1')
    after = SourceFactContext(uuid='after-1')
    fact = SourceFact(
        uuid='fact-1',
        text='Alice works for Acme.',
        target=target,
        context_before=before,
        context_after=after,
    )

    assert fact.model_dump() == {
        'uuid': 'fact-1',
        'text': 'Alice works for Acme.',
        'target': {
            'uuid': 'target-1',
            'source_blocks': [
                {
                    'uuid': 'block-1',
                    'block_type': 'paragraph',
                    'content': None,
                    'embedding': None,
                    'crop_path': None,
                    'crop_bbox': None,
                    'assets': [],
                }
            ],
        },
        'context_before': {'uuid': 'before-1', 'source_blocks': []},
        'context_after': {'uuid': 'after-1', 'source_blocks': []},
    }


def test_edge_contains_directed_vertex_references():
    edge = Edge(
        uuid='edge-1',
        source_uuid='source-1',
        destination_uuid='destination-1',
    )

    assert edge.model_dump() == {
        'uuid': 'edge-1',
        'source_uuid': 'source-1',
        'destination_uuid': 'destination-1',
    }


def test_source_content_is_a_vertex_with_default_assets():
    content = SourceBlock(
        uuid='source-1',
        block_type='paragraph',
    )

    assert isinstance(content, Vertex)
    assert content.content is None
    assert content.embedding is None
    assert content.assets == []


def test_source_content_accepts_text_embedding_and_visual_assets():
    asset = VisualAsset(path='images/block.png')
    content = SourceBlock(
        uuid='source-1',
        block_type='paragraph',
        content='canonical source text',
        embedding=[0.25, -0.5],
        assets=[asset],
    )

    assert isinstance(asset, Vertex)
    assert uuid.UUID(asset.uuid).version == 4
    assert content.content == 'canonical source text'
    assert content.embedding == [0.25, -0.5]
    assert content.assets == [asset]


def test_source_page_contains_ordered_source_blocks():
    block = SourceBlock(
        uuid='block-1',
        block_type='paragraph',
        content='page text',
    )

    page = SourcePage(index=2, blocks=[block])

    assert page.index == 2
    assert page.blocks == [block]


def test_source_is_the_ingestion_root_vertex():
    source = Source(
        uuid='source-1',
        key='calculus-book',
        metadata={'title': 'Calculus'},
    )

    assert isinstance(source, Vertex)
    assert source.model_dump() == {
        'uuid': 'source-1',
        'key': 'calculus-book',
        'metadata': {'title': 'Calculus'},
    }


def test_ocr_artifact_contains_content_and_visual_artifacts():
    artifact = OCRArtifact(
        page_index=2,
        block_index=4,
        block_type='paragraph',
        content='The value is x².',
        images=[
            OCRImageArtifact(
                path='Documents/Document_0002/Images/Image_000.png',
                image_id='image-1',
                bbox=(1.0, 2.0, 30.0, 40.0),
            )
        ],
        crop_path='Documents/Document_0002/Blocks/Block_0004.png',
        crop_bbox=(3, 4, 50, 60),
    )

    assert artifact.model_dump() == {
        'page_index': 2,
        'block_index': 4,
        'block_type': 'paragraph',
        'content': 'The value is x².',
        'images': [
            {
                'path': 'Documents/Document_0002/Images/Image_000.png',
                'image_id': 'image-1',
                'bbox': (1.0, 2.0, 30.0, 40.0),
            }
        ],
        'crop_path': 'Documents/Document_0002/Blocks/Block_0004.png',
        'crop_bbox': (3, 4, 50, 60),
    }


def test_ocr_artifact_supports_image_only_blocks():
    artifact = OCRArtifact(
        page_index=0,
        block_index=0,
        block_type='image',
        images=[OCRImageArtifact(path='Images/Image_000.png')],
    )

    assert artifact.content is None
    assert artifact.images[0].path == 'Images/Image_000.png'


def test_pointer_models_preserve_ordered_uuid_lists_and_defaults():
    instruction = Instruction(
        uuid='instruction-1',
        member_block_uuids=['block-1', 'block-2'],
        governed_statement_uuids=['statement-1'],
    )
    statement = StatementDraft(member_block_uuids=['block-3', 'block-4'])
    procedure = ProcedureDraft(member_block_uuids=['block-5', 'block-6'])

    assert instruction.member_block_uuids == ['block-1', 'block-2']
    assert instruction.governed_statement_uuids == ['statement-1']
    assert statement.is_exercise is False
    assert statement.member_block_uuids == ['block-3', 'block-4']
    assert procedure.member_block_uuids == ['block-5', 'block-6']
