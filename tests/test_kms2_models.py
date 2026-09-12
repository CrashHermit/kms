import uuid

import pytest
from pydantic import ValidationError

from kms2.core.model import (
    Edge,
    OCRArtifact,
    OCRImageArtifact,
    Source,
    SourceBlock,
    SourcePage,
    Vertex,
    VisualAsset,
)


def test_vertex_generates_uuid4_and_rejects_extra_fields():
    assert Vertex(uuid='vertex-1').uuid == 'vertex-1'
    generated = [Vertex().uuid for _ in range(2)]
    assert generated[0] != generated[1]
    assert all(uuid.UUID(value).version == 4 for value in generated)

    with pytest.raises(ValidationError):
        Vertex(label='unexpected')


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
