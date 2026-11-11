"""Final source-block embedding node and page helper."""

from kms.core.embedding import Embedder, EmbeddingRequest
from kms.source.context import project_source_block
from kms.source.models.source import SourcePage
from kms.source.processing.state import SourceProcessingState


async def embed_source_pages(
    embedder: Embedder,
    pages: list[SourcePage],
) -> list[SourcePage]:
    """Embed missing source block text and native images in page order."""
    target_positions = [
        (page_index, block_index)
        for page_index, page in enumerate(pages)
        for block_index, block in enumerate(page.blocks)
        if block.embedding is None
        and (
            (block.content is not None and block.content.strip())
            or block.image_path
        )
    ]
    if not target_positions:
        return pages

    materials = [
        project_source_block(pages[page_index].blocks[block_index]).material
        for page_index, block_index in target_positions
    ]
    embeddings = await embedder.embed(
        EmbeddingRequest(materials=materials, input_type='document')
    )
    vectors_by_position = dict(zip(target_positions, embeddings, strict=True))
    embedded_pages: list[SourcePage] = []
    for page_index, page in enumerate(pages):
        embedded_blocks = []
        for block_index, block in enumerate(page.blocks):
            position = (page_index, block_index)
            embedded_blocks.append(
                block.model_copy(
                    update={'embedding': vectors_by_position[position]}
                )
                if position in vectors_by_position
                else block
            )
        embedded_pages.append(
            page.model_copy(update={'blocks': embedded_blocks})
            if any(
                (page_index, block_index) in vectors_by_position
                for block_index in range(len(page.blocks))
            )
            else page
        )
    return embedded_pages


class SourceBlockEmbeddingNode:
    """Embed source blocks and record the identity of their vector model."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        model: str,
        dimension: int,
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
            'split_pages': await embed_source_pages(
                self._embedder, state.split_pages
            ),
        }
