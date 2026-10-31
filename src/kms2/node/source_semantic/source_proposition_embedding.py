"""Embed complete source proposition texts."""

from kms2.core.embedding import EmbeddingClient
from kms2.core.model.source_semantic.source_proposition_extraction import (
    SourcePropositionEmbeddingResult,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState


class SourcePropositionEmbeddingNode:
    """Embed propositions without writing to the database."""

    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client

    async def run(
        self, state: SourceSemanticState
    ) -> dict[str, list[SourcePropositionEmbeddingResult]]:
        """Embed complete proposition texts in input order."""
        propositions = state.source_propositions
        if not propositions:
            return {'source_proposition_embedding_results': []}
        vectors = await self._client.embed(
            [proposition.text for proposition in propositions]
        )
        return {
            'source_proposition_embedding_results': [
                SourcePropositionEmbeddingResult(
                    **proposition.model_dump(), embedding=vector
                )
                for proposition, vector in zip(
                    propositions, vectors, strict=True
                )
            ]
        }
