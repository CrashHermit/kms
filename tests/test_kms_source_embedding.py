import asyncio
from pathlib import Path
from typing import Literal

from PIL import Image

from kms.core.content import ContentImage
from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.content import MultimodalContent
from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.visual_asset import VisualAsset
from kms.langgraph.source_processing.state import SourceProcessingState
from kms.node.source_embedding import embed_source_pages
from kms.node.source_processing.embedding import EmbeddingNode


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.calls: list[
            tuple[
                list[MultimodalContent],
                list[ContentImage],
                Literal['query', 'document'],
            ]
        ] = []

    async def embed(
        self,
        materials: list[MultimodalContent],
        *,
        content_images: list[ContentImage],
        input_type: Literal['query', 'document'],
    ) -> list[list[float]]:
        self.calls.append((materials, content_images, input_type))
        return [
            [float(index), float(index + 1)] for index in range(len(materials))
        ]


def test_embedding_preserves_final_block_order_and_provenance():
    asset = VisualAsset(uuid='asset-1', path='figure.png')
    first = SourceBlock(
        uuid='child-1',
        block_type='list',
        content='first split child',
        crop_path='child-1.png',
        crop_bbox=(1, 2, 3, 4),
        assets=[asset],
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
    node = EmbeddingNode(client, model='test-model', dimension=2)
    state = SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'book.pdf'},
        source=Source(uuid='source-1', key='book.pdf'),
        image_described_pages=pages,
    )

    result = asyncio.run(node.run(state))
    embedded_pages = result['embedded_pages']

    assert client.calls[0][0] == [
        MultimodalContent(text='first split child', image_positions=[0]),
        MultimodalContent(text='second split child'),
    ]
    assert client.calls[0][1] == [
        ContentImage(('path', 'figure.png'), Path('figure.png'))
    ]
    assert client.calls[0][2] == 'document'
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
    node = EmbeddingNode(client, model='test-model', dimension=2)
    state = SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'book.pdf'},
        source=Source(uuid='source-1', key='book.pdf'),
        image_described_pages=pages,
    )

    result = asyncio.run(node.run(state))
    assert result['embedded_pages'] == pages
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
                SourceBlock(block_type=BlockType.IMAGE),
                SourceBlock(block_type=BlockType.PARAGRAPH, content=''),
            ],
        )
    ]
    client = _RecordingEmbedder()

    assert asyncio.run(embed_source_pages(client, pages)) is pages
    assert client.calls == []


def test_shared_embedding_includes_image_only_and_captioned_assets(tmp_path):
    image_path = tmp_path / 'figure.png'
    Image.new('RGB', (2, 2), color='red').save(image_path)
    image_bytes = image_path.read_bytes()
    pages = [
        SourcePage(
            index=0,
            blocks=[
                SourceBlock(
                    uuid='captioned',
                    block_type=BlockType.PARAGRAPH,
                    content='A caption',
                    assets=[
                        VisualAsset(uuid='caption-asset', path=str(image_path))
                    ],
                ),
                SourceBlock(
                    uuid='image-only',
                    block_type=BlockType.IMAGE,
                    assets=[
                        VisualAsset(uuid='image-asset', path=str(image_path))
                    ],
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

    materials, content_images, input_type = embedder.calls[0]
    assert [material.text for material in materials] == ['A caption', None]
    assert [material.image_positions for material in materials] == [[0], [0]]
    assert len(content_images) == 1
    assert Path(content_images[0].payload).read_bytes() == image_bytes
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
