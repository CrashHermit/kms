from __future__ import annotations

import asyncio

from kms.core.embedding import EmbeddingRequest
from kms.source.models.source import BlockType, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.processing.state import SourceProcessingState
from kms.source.processing.unit_embedding import SourceUnitEmbeddingNode


class _Embedder:
    def __init__(self) -> None:
        self.requests: list[EmbeddingRequest] = []

    async def embed(self, request: EmbeddingRequest) -> list[list[float]]:
        self.requests.append(request)
        return [
            [float(index), 9.0] for index, _ in enumerate(request.materials)
        ]


def _block(uuid: str, content: str, embedding=None) -> SourceBlock:
    return SourceBlock(
        uuid=uuid,
        block_type=BlockType.PARAGRAPH,
        content=content,
        embedding=embedding,
    )


def test_singleton_reuses_block_vector_and_multiblock_embeds_complete_order():
    singleton_vector = [3.0, 4.0]
    singleton = SourceUnit(
        uuid='singleton',
        kind=SourceUnitKind.CONTENT,
        blocks=[_block('one', 'single', singleton_vector)],
    )
    mixed = SourceUnit(
        uuid='mixed',
        kind=SourceUnitKind.EXERCISE,
        blocks=[_block('two', 'task'), _block('three', 'answer')],
    )
    embedder = _Embedder()
    state = SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'source.pdf'},
        source={'uuid': 'source', 'key': 'source.pdf'},
        units=[singleton, mixed],
    )

    units = asyncio.run(SourceUnitEmbeddingNode(embedder).run(state))['units']

    assert units[0].embedding == singleton_vector
    assert units[1].embedding == [0.0, 9.0]
    assert len(embedder.requests) == 1
    request = embedder.requests[0]
    assert request.input_type == 'document'
    assert [material.text for material in request.materials] == ['taskanswer']


def test_existing_unit_vectors_are_preserved_without_embedding_work():
    unit = SourceUnit(
        uuid='stored',
        kind=SourceUnitKind.PROCEDURE,
        blocks=[_block('block', 'method', [1.0])],
        embedding=[2.0],
    )
    embedder = _Embedder()
    state = SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'source.pdf'},
        source={'uuid': 'source', 'key': 'source.pdf'},
        units=[unit],
    )

    result = asyncio.run(SourceUnitEmbeddingNode(embedder).run(state))

    assert result['units'] == [unit]
    assert embedder.requests == []
