"""Embed source-local event descriptions in a LangGraph phase."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.embedding import EmbeddingClient
from kms2.core.model import SourceEventDescriptionResult
from kms2.langgraph.semantic.state import SemanticState


class SourceEventEmbeddingWorkerState(TypedDict):
    """State supplied to the event embedding worker."""

    source_event_description_results: list[SourceEventDescriptionResult]


class SourceEventEmbeddingNode:
    """Batch event descriptions and retain occurrence UUID alignment."""

    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client

    def dispatch(
        self, state: SemanticState
    ) -> list[Send] | Literal['source_event_embedding_collect']:
        """Dispatch all event descriptions as one ordered embedding batch."""
        if not state.source_event_description_results:
            return 'source_event_embedding_collect'
        return [
            Send(
                'source_event_embedding_worker',
                {
                    'source_event_description_results': state.source_event_description_results,
                },
            )
        ]

    async def worker(
        self, state: SourceEventEmbeddingWorkerState
    ) -> dict[str, list[SourceEventDescriptionResult]]:
        """Embed event term-description pairs in input order."""
        results = state['source_event_description_results']
        vectors = await self._client.embed(
            [f'{result.name} : {result.description}' for result in results]
        )
        return {
            'source_event_embedding_results': [
                result.model_copy(update={'embedding': vector})
                for result, vector in zip(results, vectors, strict=True)
            ]
        }

    def collect(self, state: SemanticState) -> dict[str, object]:
        """Finish embedding fan-in before persistence."""
        return {}
