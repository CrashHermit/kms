"""Embed source-local event descriptions in a LangGraph phase."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model import EventEnrichmentResult
from kms2.langgraph.semantic.state import SemanticState


class EventEmbeddingWorkerState(TypedDict):
    """State supplied to the event embedding worker."""

    event_description_results: list[EventEnrichmentResult]


class EventEmbeddingNode:
    """Batch event descriptions and retain occurrence UUID alignment."""

    def __init__(self, client) -> None:
        self._client = client

    def dispatch(
        self, state: SemanticState
    ) -> list[Send] | Literal['event_embedding_collect']:
        """Dispatch all event descriptions as one ordered embedding batch."""
        if not state.event_description_results:
            return 'event_embedding_collect'
        return [
            Send(
                'event_embedding_worker',
                {
                    'event_description_results': state.event_description_results,
                },
            )
        ]

    async def worker(
        self, state: EventEmbeddingWorkerState
    ) -> dict[str, list[EventEnrichmentResult]]:
        """Embed event term-description pairs in input order."""
        results = state['event_description_results']
        vectors = await self._client.embed(
            [f'{result.name} : {result.description}' for result in results]
        )
        return {
            'event_embedding_results': [
                result.model_copy(update={'embedding': vector})
                for result, vector in zip(results, vectors, strict=True)
            ]
        }

    def collect(self, state: SemanticState) -> dict[str, object]:
        """Finish embedding fan-in before persistence."""
        return {}
