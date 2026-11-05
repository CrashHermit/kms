"""LangGraph node for final source-block embeddings."""

from kms.core.embedding import Embedder
from kms.langgraph.source_processing.state import SourceProcessingState
from kms.node.source_embedding import embed_source_pages


class EmbeddingNode:
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
            'embedded_pages': await embed_source_pages(
                self._embedder, state.image_described_pages
            ),
        }
