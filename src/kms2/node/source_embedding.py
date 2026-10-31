"""Embedding helpers shared by source pipeline stages."""

from typing import cast

from kms2.core.embedding import EmbeddingClient
from kms2.core.model.page import SourcePage


async def embed_source_pages(
    client: EmbeddingClient,
    pages: list[SourcePage],
) -> list[SourcePage]:
    """Embed only source blocks that do not already have vectors."""
    if not any(
        block.embedding is None for page in pages for block in page.blocks
    ):
        return pages

    embeddings = iter(
        await client.embed(
            [
                cast(str, block.content)
                for page in pages
                for block in page.blocks
                if block.embedding is None
            ]
        )
    )
    embedded_pages: list[SourcePage] = []
    for page in pages:
        embedded_blocks = []
        changed = False
        for block in page.blocks:
            if block.embedding is None:
                embedded_blocks.append(
                    block.model_copy(update={'embedding': next(embeddings)})
                )
                changed = True
            else:
                embedded_blocks.append(block)
        embedded_pages.append(
            page.model_copy(update={'blocks': embedded_blocks})
            if changed
            else page
        )
    return embedded_pages
