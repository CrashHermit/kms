"""Embed complete ordered source units."""

from kms.core.embedding import Embedder, EmbeddingRequest
from kms.source.context import project_source_unit
from kms.source.processing.state import SourceProcessingState


class SourceUnitEmbeddingNode:
    """Embed each multi-block unit and reuse singleton block vectors."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def run(self, state: SourceProcessingState) -> dict[str, object]:
        units = state.units
        positions = [
            index
            for index, unit in enumerate(units)
            if unit.embedding is None and len(unit.blocks) > 1
        ]
        vectors = (
            await self._embedder.embed(
                EmbeddingRequest(
                    materials=[
                        project_source_unit(units[index]) for index in positions
                    ],
                    input_type='document',
                )
            )
            if positions
            else []
        )
        by_position = dict(zip(positions, vectors, strict=True))
        return {
            'units': [
                unit.model_copy(
                    update={
                        'embedding': (
                            unit.blocks[0].embedding
                            if len(unit.blocks) == 1
                            else by_position[index]
                        )
                    }
                )
                if unit.embedding is None
                else unit
                for index, unit in enumerate(units)
            ]
        }
