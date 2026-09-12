"""Embed source-local predicate descriptions in a LangGraph phase."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model import PredicateEnrichmentResult
from kms2.langgraph.semantic.state import SemanticState


class PredicateEmbeddingWorkerState(TypedDict):
    """State supplied to the predicate embedding worker."""

    predicate_description_results: list[PredicateEnrichmentResult]


class PredicateEmbeddingNode:
    """Batch predicate descriptions and retain occurrence UUID alignment."""

    def __init__(self, client) -> None:
        self._client = client

    def dispatch(
        self, state: SemanticState
    ) -> list[Send] | Literal['predicate_embedding_collect']:
        """Dispatch predicate descriptions as one ordered embedding batch."""
        if not state.predicate_description_results:
            return 'predicate_embedding_collect'
        return [
            Send(
                'predicate_embedding_worker',
                {
                    'predicate_description_results': state.predicate_description_results,
                },
            )
        ]

    async def worker(
        self, state: PredicateEmbeddingWorkerState
    ) -> dict[str, list[PredicateEnrichmentResult]]:
        """Embed predicate term-description pairs in input order."""
        results = state['predicate_description_results']
        vectors = await self._client.embed(
            [f'{result.predicate} : {result.description}' for result in results]
        )
        return {
            'predicate_embedding_results': [
                result.model_copy(update={'embedding': vector})
                for result, vector in zip(results, vectors, strict=True)
            ]
        }

    def collect(self, state: SemanticState) -> dict[str, object]:
        """Finish embedding fan-in before persistence."""
        return {}
