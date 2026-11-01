import asyncio
from collections.abc import Sequence

from kms2.core.embedding import EmbeddingClient
from kms2.core.model.block import SourceBlock
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.visual_asset import VisualAsset
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.node.source_embedding import embed_source_pages
from kms2.node.source_processing.embedding import EmbeddingNode


class _RecordingEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        self.calls.append(values)
        return [
            [float(index), float(index + 1)] for index in range(len(values))
        ]


def test_embedding_client_protocol_is_structural():
    assert isinstance(_RecordingEmbeddingClient(), EmbeddingClient)


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
    client = _RecordingEmbeddingClient()
    node = EmbeddingNode(client)
    state = SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'book.pdf'},
        source=Source(uuid='source-1', key='book.pdf'),
        image_described_pages=pages,
    )

    embedded_pages = asyncio.run(node.run(state))['embedded_pages']

    assert client.calls == [['first split child', 'second split child']]
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
    client = _RecordingEmbeddingClient()
    node = EmbeddingNode(client)
    state = SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'book.pdf'},
        source=Source(uuid='source-1', key='book.pdf'),
        image_described_pages=pages,
    )

    assert asyncio.run(node.run(state)) == {'embedded_pages': pages}
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
    client = _RecordingEmbeddingClient()

    embedded_pages = asyncio.run(embed_source_pages(client, pages))

    assert client.calls == [['first missing block', 'second missing block']]
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
    client = _RecordingEmbeddingClient()

    assert asyncio.run(embed_source_pages(client, pages)) is pages
    assert client.calls == []
