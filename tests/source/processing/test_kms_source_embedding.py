import asyncio
from pathlib import Path
from typing import Literal

from PIL import Image

from kms.core.embedding import EmbeddingRequest
from kms.media.models import ImageAsset, ImagePart, Material, TextPart
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import BlockType, Source, SourceBlock, SourcePage
from kms.source.processing.block_embedding import (
    SourceBlockEmbeddingNode,
    embed_source_pages,
)
from kms.source.processing.state import SourceProcessingState


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.calls: list[
            tuple[list[Material], Literal['query', 'document']]
        ] = []

    async def embed(self, request: EmbeddingRequest) -> list[list[float]]:
        self.calls.append((request.materials, request.input_type))
        return [
            [float(index), float(index + 1)]
            for index in range(len(request.materials))
        ]


def test_embedding_preserves_final_block_order_and_provenance():
    first = SourceBlock(
        uuid='child-1',
        block_type=BlockType.IMAGE,
        content='first split child',
        crop_path='child-1.png',
        crop_bbox=(1, 2, 3, 4),
        image_path='figure.png',
        image_page_indexes=[0],
    )
    second = SourceBlock(
        uuid='child-2',
        block_type='list',
        content='second split child',
    )
    pages = [
        SourcePage(index=0, markdown='# Page 0', blocks=[first]),
        SourcePage(index=1, markdown='# Page 1', blocks=[second]),
    ]
    client = _RecordingEmbedder()
    node = SourceBlockEmbeddingNode(client, model='test-model', dimension=2)
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='book.pdf'),
        source=Source(uuid='source-1', key='book.pdf'),
        split_pages=pages,
    )

    result = asyncio.run(node.run(state))
    embedded_pages = result['split_pages']

    assert client.calls[0][0] == [
        Material(
            parts=(
                TextPart(text='first split child'),
                ImagePart(
                    asset=ImageAsset(('path', 'figure.png'), Path('figure.png'))
                ),
            )
        ),
        Material.from_text('second split child'),
    ]
    assert client.calls[0][1] == 'document'
    assert result['source'].metadata['embedding_profile'] == {
        'model': 'test-model',
        'dimension': 2,
    }
    assert [
        block.embedding for page in embedded_pages for block in page.blocks
    ] == [
        [0.0, 1.0],
        [1.0, 2.0],
    ]
    assert embedded_pages[0].markdown == '# Page 0'
    assert embedded_pages[0].blocks[0].model_dump() == first.model_dump(
        exclude={'embedding'}
    ) | {'embedding': [0.0, 1.0]}
    assert embedded_pages[1].blocks[0].uuid == 'child-2'


def test_embedding_preserves_empty_pages_without_provider_call():
    pages = [SourcePage(index=0)]
    client = _RecordingEmbedder()
    node = SourceBlockEmbeddingNode(client, model='test-model', dimension=2)
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='book.pdf'),
        source=Source(uuid='source-1', key='book.pdf'),
        split_pages=pages,
    )

    result = asyncio.run(node.run(state))
    assert result['split_pages'] == pages
    assert result['source'].metadata['embedding_profile'] == {
        'model': 'test-model',
        'dimension': 2,
    }
    assert client.calls == []


def test_shared_embedding_preserves_existing_vectors_and_embeds_missing_in_order():
    first = SourceBlock(
        uuid='child-1',
        block_type='paragraph',
        content='first missing block',
    )
    existing = SourceBlock(
        uuid='existing',
        block_type='paragraph',
        content='already embedded',
        embedding=[9.0, 9.0],
    )
    second = SourceBlock(
        uuid='child-2',
        block_type='paragraph',
        content='second missing block',
    )
    pages = [
        SourcePage(index=0, blocks=[first, existing]),
        SourcePage(index=1, blocks=[second]),
    ]
    client = _RecordingEmbedder()

    embedded_pages = asyncio.run(embed_source_pages(client, pages))

    assert [material.text for material in client.calls[0][0]] == [
        'first missing block',
        'second missing block',
    ]
    assert [
        block.embedding for page in embedded_pages for block in page.blocks
    ] == [
        [0.0, 1.0],
        [9.0, 9.0],
        [1.0, 2.0],
    ]
    assert embedded_pages[0].blocks[1] is existing


def test_shared_embedding_returns_fully_embedded_pages_without_provider_call():
    pages = [
        SourcePage(
            index=0,
            blocks=[
                SourceBlock(
                    uuid='block-1',
                    block_type='paragraph',
                    embedding=[1.0, 2.0],
                )
            ],
        )
    ]
    client = _RecordingEmbedder()

    assert asyncio.run(embed_source_pages(client, pages)) is pages
    assert client.calls == []


def test_shared_embedding_skips_blocks_without_nonempty_text():
    pages = [
        SourcePage(
            index=0,
            blocks=[
                SourceBlock(
                    block_type=BlockType.PARAGRAPH,
                ),
                SourceBlock(block_type=BlockType.PARAGRAPH, content=''),
            ],
        )
    ]
    client = _RecordingEmbedder()

    assert asyncio.run(embed_source_pages(client, pages)) is pages
    assert client.calls == []


def test_shared_embedding_includes_image_only_and_captioned_blocks(tmp_path):
    image_path = tmp_path / 'figure.png'
    Image.new('RGB', (2, 2), color='red').save(image_path)
    pages = [
        SourcePage(
            index=0,
            blocks=[
                SourceBlock(
                    uuid='captioned',
                    block_type=BlockType.IMAGE,
                    content='A caption',
                    image_path=str(image_path),
                    image_page_indexes=[0],
                ),
                SourceBlock(
                    uuid='image-only',
                    block_type=BlockType.IMAGE,
                    image_path=str(image_path),
                    image_page_indexes=[0],
                ),
                SourceBlock(
                    uuid='empty',
                    block_type=BlockType.PARAGRAPH,
                    content='  ',
                ),
            ],
        )
    ]
    embedder = _RecordingEmbedder()

    embedded_pages = asyncio.run(embed_source_pages(embedder, pages))

    materials, input_type = embedder.calls[0]
    assert [material.text for material in materials] == ['A caption', None]
    assert [len(material.images) for material in materials] == [1, 1]
    assert all(
        material.images[0].payload == image_path for material in materials
    )
    assert input_type == 'document'
    assert [block.embedding for block in embedded_pages[0].blocks] == [
        [0.0, 1.0],
        [1.0, 2.0],
        None,
    ]
    assert [block.uuid for block in embedded_pages[0].blocks] == [
        'captioned',
        'image-only',
        'empty',
    ]
    assert embedded_pages[0].blocks[2] is pages[0].blocks[2]
