"""Embed split source blocks before pedagogical persistence."""

from kms.core.embedding import Embedder
from kms.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms.node.source_embedding import embed_source_pages


class SourcePedagogicalEmbeddingNode:
    """Embed only blocks introduced without vectors by pedagogical splitting."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def run(self, state: SourcePedagogicalState) -> dict[str, object]:
        """Embed missing split-block vectors and retain already embedded blocks."""
        reembedded_block_count = sum(
            block.embedding is None
            for page in state.split_pages
            for block in page.blocks
        )
        embedded_pages = await embed_source_pages(
            self._embedder, state.split_pages
        )
        return {
            'embedded_pages': embedded_pages,
            'reembedded_block_count': reembedded_block_count,
        }
