"""Embed source-local entity descriptions in a LangGraph phase."""

from kms.core.embedding import Embedder
from kms.core.model.content import MultimodalContent
from kms.core.model.source_semantic.source_entity import (
    SourceEntityDescriptionResult,
)
from kms.langgraph.source_semantic.state import SourceSemanticState


class SourceEntityEmbeddingNode:
    """Batch entity descriptions and retain occurrence UUID alignment."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def run(
        self, state: SourceSemanticState
    ) -> dict[str, list[SourceEntityDescriptionResult]]:
        """Embed entity term-description pairs in input order."""
        results = state.source_entity_description_results
        if not results:
            return {'source_entity_embedding_results': []}
        vectors = await self._embedder.embed(
            [
                MultimodalContent(text=f'{result.name} : {result.description}')
                for result in results
            ],
            content_images=[],
            input_type='document',
        )
        return {
            'source_entity_embedding_results': [
                result.model_copy(update={'embedding': vector})
                for result, vector in zip(results, vectors, strict=True)
            ]
        }
