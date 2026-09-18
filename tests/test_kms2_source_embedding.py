import asyncio
from collections.abc import Sequence

from kms2.core.embedding import EmbeddingClient
from kms2.core.model import Source, SourceBlock, SourcePage, VisualAsset
from kms2.langgraph.source.state import SourceState
from kms2.node.source.embedding import EmbeddingNode


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


def test_embedding_dispatch_worker_collect_preserves_final_block_order():
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
    state = SourceState(
        pdf_path='book.pdf',
        source=Source(uuid='source-1', key='book.pdf'),
        split_pages=pages,
    )

    sends = node.dispatch(state)
    assert len(sends) == 1
    result = asyncio.run(node.worker(sends[0].arg))
    embedded_pages = node.collect(
        state.model_copy(
            update={'embedding_results': result['embedding_results']}
        )
    )['embedded_pages']

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


def test_embedding_collect_passes_empty_split_pages_through():
    pages = [SourcePage(index=0)]
    node = EmbeddingNode(_RecordingEmbeddingClient())
    state = SourceState(
        pdf_path='book.pdf',
        source=Source(uuid='source-1', key='book.pdf'),
        split_pages=pages,
    )

    assert node.dispatch(state) == 'embedding_collect'
    assert node.collect(state) == {'embedded_pages': pages}
