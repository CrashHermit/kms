"""Embed source-local entity descriptions in a LangGraph phase."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.embedding import EmbeddingClient
from kms2.core.model import SourceEntityDescriptionResult
from kms2.langgraph.semantic.state import SemanticState


class SourceEntityEmbeddingWorkerState(TypedDict):
    """State supplied to the entity embedding worker."""

    source_entity_description_results: list[SourceEntityDescriptionResult]


class SourceEntityEmbeddingNode:
    """Batch entity descriptions and retain occurrence UUID alignment."""

    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client

    def dispatch(
        self, state: SemanticState
    ) -> list[Send] | Literal['source_entity_embedding_collect']:
        """Dispatch all entity descriptions as one ordered embedding batch."""
        if not state.source_entity_description_results:
            return 'source_entity_embedding_collect'
        return [
            Send(
                'source_entity_embedding_worker',
                {
                    'source_entity_description_results': state.source_entity_description_results,
                },
            )
        ]

    async def worker(
        self, state: SourceEntityEmbeddingWorkerState
    ) -> dict[str, list[SourceEntityDescriptionResult]]:
        """Embed entity term-description pairs in input order."""
        results = state['source_entity_description_results']
        vectors = await self._client.embed(
            [f'{result.name} : {result.description}' for result in results]
        )
        return {
            'source_entity_embedding_results': [
                result.model_copy(update={'embedding': vector})
                for result, vector in zip(results, vectors, strict=True)
            ]
        }

    def collect(self, state: SemanticState) -> dict[str, object]:
        """Finish embedding fan-in before persistence."""
        return {}
