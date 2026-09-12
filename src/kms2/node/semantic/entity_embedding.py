"""Embed source-local entity descriptions in a LangGraph phase."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model import EntityEnrichmentResult
from kms2.langgraph.semantic.state import SemanticState


class EntityEmbeddingWorkerState(TypedDict):
    """State supplied to the entity embedding worker."""

    entity_description_results: list[EntityEnrichmentResult]


class EntityEmbeddingNode:
    """Batch entity descriptions and retain occurrence UUID alignment."""

    def __init__(self, client) -> None:
        self._client = client

    def dispatch(
        self, state: SemanticState
    ) -> list[Send] | Literal['entity_embedding_collect']:
        """Dispatch all entity descriptions as one ordered embedding batch."""
        if not state.entity_description_results:
            return 'entity_embedding_collect'
        return [
            Send(
                'entity_embedding_worker',
                {
                    'entity_description_results': state.entity_description_results,
                },
            )
        ]

    async def worker(
        self, state: EntityEmbeddingWorkerState
    ) -> dict[str, list[EntityEnrichmentResult]]:
        """Embed entity term-description pairs in input order."""
        results = state['entity_description_results']
        vectors = await self._client.embed(
            [f'{result.name} : {result.description}' for result in results]
        )
        return {
            'entity_embedding_results': [
                result.model_copy(update={'embedding': vector})
                for result, vector in zip(results, vectors, strict=True)
            ]
        }

    def collect(self, state: SemanticState) -> dict[str, object]:
        """Finish embedding fan-in before persistence."""
        return {}
