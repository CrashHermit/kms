"""Embedding helpers shared by source pipeline stages."""

from kms.core.content import MultimodalContentBuilder
from kms.core.embedding import Embedder
from kms.core.model.page import SourcePage
from kms.core.source_context import project_source_block


async def embed_source_pages(
    embedder: Embedder,
    pages: list[SourcePage],
) -> list[SourcePage]:
    """Embed missing source block text and native assets in page order."""
    target_positions = [
        (page_index, block_index)
        for page_index, page in enumerate(pages)
        for block_index, block in enumerate(page.blocks)
        if block.embedding is None
        and (
            (block.content is not None and block.content.strip())
            or block.assets
        )
    ]
    if not target_positions:
        return pages

    builder = MultimodalContentBuilder()
    materials = [
        project_source_block(
            pages[page_index].blocks[block_index], builder
        ).material
        for page_index, block_index in target_positions
    ]
    embeddings = await embedder.embed(
        materials,
        content_images=builder.images,
        input_type='document',
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
