"""Embed split source blocks before pedagogical persistence."""

from kms2.core.embedding import EmbeddingClient
from kms2.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms2.node.source_embedding import embed_source_pages


class SourcePedagogicalEmbeddingNode:
    """Embed only blocks introduced without vectors by pedagogical splitting."""

    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client

    async def run(self, state: SourcePedagogicalState) -> dict[str, object]:
        """Embed missing split-block vectors and retain already embedded blocks."""
        reembedded_block_count = sum(
            block.embedding is None
            for page in state.split_pages
            for block in page.blocks
        )
        embedded_pages = await embed_source_pages(
            self._client, state.split_pages
        )
        return {
            'embedded_pages': embedded_pages,
            'reembedded_block_count': reembedded_block_count,
        }
