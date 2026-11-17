"""Embed final source blocks and record vector provenance."""

from kms.core.embedding import Embedder, EmbeddingRequest
from kms.source.context import project_source_block
from kms.source.models.unit import SourceUnit
from kms.source.processing.state import SourceProcessingState


async def embed_source_blocks(
    embedder: Embedder, units: list[SourceUnit]
) -> list[SourceUnit]:
    """Embed readable blocks in their unit and block encounter order."""
    positions = [
        (unit_index, block_index)
        for unit_index, unit in enumerate(units)
        for block_index, block in enumerate(unit.blocks)
        if block.embedding is None
        and (
            (block.content is not None and block.content.strip())
            or block.image_path
        )
    ]
    if not positions:
        return units
    vectors = await embedder.embed(
        EmbeddingRequest(
            materials=[
                project_source_block(
                    units[unit_index].blocks[block_index]
                ).material
                for unit_index, block_index in positions
            ],
            input_type='document',
        )
    )
    by_position = dict(zip(positions, vectors, strict=True))
    return [
        unit.model_copy(
            update={
                'blocks': [
                    block.model_copy(
                        update={
                            'embedding': by_position[(unit_index, block_index)]
                        }
                    )
                    if (unit_index, block_index) in by_position
                    else block
                    for block_index, block in enumerate(unit.blocks)
                ]
            }
        )
        for unit_index, unit in enumerate(units)
    ]


class SourceBlockEmbeddingNode:
    """Embed source blocks and record the identity of their vector model."""

    def __init__(
        self, embedder: Embedder, *, model: str, dimension: int
    ) -> None:
        self._embedder = embedder
        self._model = model
        self._dimension = dimension

    async def run(self, state: SourceProcessingState) -> dict[str, object]:
        """Embed missing block vectors and return index provenance."""
        source = state.source.model_copy(
            update={
                'metadata': {
                    **state.source.metadata,
                    'embedding_profile': {
                        'model': self._model,
                        'dimension': self._dimension,
                    },
                }
            }
        )
        return {
            'source': source,
            'units': await embed_source_blocks(self._embedder, state.units),
        }
