"""Embed source-local predicate descriptions in a LangGraph phase."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.embedding import EmbeddingClient
from kms2.core.model import SourcePredicateDescriptionResult
from kms2.langgraph.semantic.state import SemanticState


class SourcePredicateEmbeddingWorkerState(TypedDict):
    """State supplied to the predicate embedding worker."""

    source_predicate_description_results: list[SourcePredicateDescriptionResult]


class SourcePredicateEmbeddingNode:
    """Batch predicate descriptions and retain occurrence UUID alignment."""

    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client

    def dispatch(
        self, state: SemanticState
    ) -> list[Send] | Literal['source_predicate_embedding_collect']:
        """Dispatch predicate descriptions as one ordered embedding batch."""
        if not state.source_predicate_description_results:
            return 'source_predicate_embedding_collect'
        return [
            Send(
                'source_predicate_embedding_worker',
                {
                    'source_predicate_description_results': state.source_predicate_description_results,
                },
            )
        ]

    async def worker(
        self, state: SourcePredicateEmbeddingWorkerState
    ) -> dict[str, list[SourcePredicateDescriptionResult]]:
        """Embed predicate term-description pairs in input order."""
        results = state['source_predicate_description_results']
        vectors = await self._client.embed(
            [f'{result.predicate} : {result.description}' for result in results]
        )
        return {
            'source_predicate_embedding_results': [
                result.model_copy(update={'embedding': vector})
                for result, vector in zip(results, vectors, strict=True)
            ]
        }

    def collect(self, state: SemanticState) -> dict[str, object]:
        """Finish embedding fan-in before persistence."""
        return {}
