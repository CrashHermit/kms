import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from kms.core.models import Edge, Vertex
from kms.source.models.source import Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.processing.ocr_models import OCRArtifact


def test_vertex_generates_uuid4_and_rejects_extra_fields():
    assert Vertex(uuid='vertex-1').uuid == 'vertex-1'
    generated = [Vertex().uuid for _ in range(2)]
    assert generated[0] != generated[1]
    assert all(uuid.UUID(value).version == 4 for value in generated)

    with pytest.raises(ValidationError):
        Vertex(label='unexpected')


def test_vertex_creation_time_is_utc_and_preserved():
    before = datetime.now(UTC)
    vertex = Vertex()
    after = datetime.now(UTC)
    assert before <= vertex.created_at <= after
    assert vertex.created_at.tzinfo == UTC
    assert vertex.updated_at == vertex.created_at
    assert vertex.invalidated_at is None
    assert Vertex.model_validate_json(vertex.model_dump_json()) == vertex


def test_edge_creation_time_is_utc_and_preserved():
    before = datetime.now(UTC)
    edge = Edge()
    after = datetime.now(UTC)
    assert before <= edge.created_at <= after
    assert edge.created_at.tzinfo == UTC
    assert edge.updated_at == edge.created_at
    assert edge.invalidated_at is None
    assert Edge.model_validate_json(edge.model_dump_json()) == edge


@pytest.mark.parametrize('model', [Vertex, Edge])
def test_lifecycle_defaults_follow_explicit_creation_time(model):
    created_at = datetime(2020, 2, 3, 4, 5, tzinfo=UTC)
    instance = model(created_at=created_at)
    assert instance.updated_at == created_at
    assert instance.invalidated_at is None

    updated_at = datetime(2021, 2, 3, 4, 5, tzinfo=UTC)
    invalidated_at = datetime(2022, 2, 3, 4, 5, tzinfo=UTC)
    explicit = model(
        created_at=created_at,
        updated_at=updated_at,
        invalidated_at=invalidated_at,
    )
    assert model.model_validate_json(explicit.model_dump_json()) == explicit


def test_source_block_defaults_and_image_ownership():
    text = SourceBlock(uuid='text-1', block_type='paragraph')
    image = SourceBlock(
        uuid='image-1',
        block_type='image',
        image_path='images/block.png',
    )

    assert isinstance(text, Vertex)
    assert text.content is None
    assert text.embedding is None
    assert text.crop_path is None
    assert text.image_path is None
    assert 'page_indexes' not in text.model_dump()
    assert 'page_indexes' not in image.model_dump()
    assert image.image_path == 'images/block.png'


def test_source_blocks_accept_content_and_embedding():
    block = SourceBlock(
        uuid='block-1',
        block_type='paragraph',
        content='canonical source text',
        embedding=[0.25, -0.5],
    )

    assert block.content == 'canonical source text'
    assert block.embedding == [0.25, -0.5]


def test_image_blocks_require_image_path_and_non_images_cannot_own_one():
    with pytest.raises(ValidationError):
        SourceBlock(uuid='missing-image', block_type='image')
    with pytest.raises(ValidationError):
        SourceBlock(
            uuid='text-with-image',
            block_type='paragraph',
            image_path='images/block.png',
        )


def test_source_is_the_ingestion_root_vertex():
    source = Source(
        uuid='source-1',
        key='calculus-book',
        metadata={'title': 'Calculus'},
    )

    assert isinstance(source, Vertex)
    assert source.model_dump() == {
        'uuid': 'source-1',
        'created_at': source.created_at,
        'updated_at': source.updated_at,
        'invalidated_at': None,
        'key': 'calculus-book',
        'metadata': {'title': 'Calculus'},
        'revision': 0,
    }


def test_source_unit_has_one_canonical_ordered_block_hierarchy():
    blocks = [
        SourceBlock(uuid='block-1', block_type='paragraph', content='first'),
        SourceBlock(uuid='block-2', block_type='equation', content='second'),
    ]
    unit = SourceUnit(
        uuid='unit-1', kind=SourceUnitKind.EXERCISE, blocks=blocks
    )

    assert isinstance(unit, Vertex)
    assert unit.kind is SourceUnitKind.EXERCISE
    assert unit.blocks == blocks
    assert unit.embedding is None
    assert set(SourceUnitKind) == {
        SourceUnitKind.CONTENT,
        SourceUnitKind.EXERCISE,
        SourceUnitKind.PROCEDURE,
    }
    with pytest.raises(ValidationError):
        SourceUnit(uuid='missing-blocks', kind='content')


def test_ocr_artifact_contains_text_and_crop_provenance():
    artifact = OCRArtifact(
        page_index=2,
        block_index=4,
        block_type='paragraph',
        content='The value is x².',
        crop_path='Documents/Document_0002/Blocks/Block_0004.png',
        crop_bbox=(3, 4, 50, 60),
    )

    assert artifact.model_dump() == {
        'page_index': 2,
        'block_index': 4,
        'block_type': 'paragraph',
        'content': 'The value is x².',
        'crop_path': 'Documents/Document_0002/Blocks/Block_0004.png',
        'crop_bbox': (3, 4, 50, 60),
    }


def test_ocr_image_block_keeps_crop_provenance_without_fragment_assets():
    artifact = OCRArtifact(
        page_index=0,
        block_index=0,
        block_type='image',
        crop_path='Blocks/ImageBlock.png',
        crop_bbox=(1, 2, 30, 40),
    )

    assert artifact.content is None
    assert artifact.crop_path == 'Blocks/ImageBlock.png'
