"""LangGraph node for final source-block embeddings."""

from kms2.core.embedding import EmbeddingClient
from kms2.core.model.page import SourcePage
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.node.source_embedding import embed_source_pages


class EmbeddingNode:
    """Embed final source-block content before persistence."""

    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client

    async def run(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourcePage]]:
        """Embed missing block vectors in canonical page and block order."""
        return {
            'embedded_pages': await embed_source_pages(
                self._client, state.image_described_pages
            )
        }
