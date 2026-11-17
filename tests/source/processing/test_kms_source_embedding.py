from __future__ import annotations

import asyncio
from pathlib import Path

from PIL import Image

from kms.core.embedding import EmbeddingRequest
from kms.media.models import ImagePart, TextPart
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.processing.block_embedding import SourceBlockEmbeddingNode
from kms.source.processing.state import SourceProcessingState


class _Embedder:
    def __init__(self) -> None:
        self.requests: list[EmbeddingRequest] = []

    async def embed(self, request: EmbeddingRequest) -> list[list[float]]:
        self.requests.append(request)
        return [
            [float(index), float(index + 1)]
            for index in range(len(request.materials))
        ]


def test_block_embeddings_follow_final_unit_encounter_order_and_keep_native_images(
    tmp_path: Path,
):
    image_path = tmp_path / 'diagram.png'
    Image.new('RGB', (8, 8), 'red').save(image_path)
    existing_vector = [8.0, 9.0]
    units = [
        SourceUnit(
            kind=SourceUnitKind.CONTENT,
            blocks=[
                SourceBlock(
                    uuid='text',
                    block_type=BlockType.PARAGRAPH,
                    content='readable text',
                ),
                SourceBlock(
                    uuid='empty',
                    block_type=BlockType.PARAGRAPH,
                    content='   ',
                ),
            ],
        ),
        SourceUnit(
            kind=SourceUnitKind.EXERCISE,
            blocks=[
                SourceBlock(
                    uuid='image',
                    block_type=BlockType.IMAGE,
                    image_path=str(image_path),
                ),
                SourceBlock(
                    uuid='cached',
                    block_type=BlockType.PARAGRAPH,
                    content='already embedded',
                    embedding=existing_vector,
                ),
            ],
        ),
    ]
    embedder = _Embedder()
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source', key='source.pdf'),
        units=units,
    )

    result = asyncio.run(
        SourceBlockEmbeddingNode(embedder, model='embed-v1', dimension=2).run(
            state
        )
    )

    embedded = result['units']
    assert [block.embedding for unit in embedded for block in unit.blocks] == [
        [0.0, 1.0],
        None,
        [1.0, 2.0],
        existing_vector,
    ]
    assert result['source'].metadata['embedding_profile'] == {
        'model': 'embed-v1',
        'dimension': 2,
    }
    request = embedder.requests[0]
    assert request.input_type == 'document'
    assert [material.text for material in request.materials] == [
        'readable text',
        None,
    ]
    assert isinstance(request.materials[1].parts[0], ImagePart)
    assert isinstance(request.materials[0].parts[0], TextPart)


def test_no_readable_blocks_skip_embedding_request():
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='source', key='source.pdf'),
        units=[
            SourceUnit(
                kind=SourceUnitKind.CONTENT,
                blocks=[
                    SourceBlock(
                        uuid='blank',
                        block_type=BlockType.PARAGRAPH,
                        content='',
                    )
                ],
            )
        ],
    )
    embedder = _Embedder()

    result = asyncio.run(
        SourceBlockEmbeddingNode(embedder, model='embed-v1', dimension=2).run(
            state
        )
    )

    assert result['units'] == state.units
    assert embedder.requests == []
