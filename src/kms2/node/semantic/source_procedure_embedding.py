"""Embed source procedure descriptions in a LangGraph phase."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.embedding import EmbeddingClient
from kms2.core.model import SourceProcedureDescriptionResult
from kms2.langgraph.semantic.state import SemanticState


class SourceProcedureEmbeddingWorkerState(TypedDict):
    """State supplied to the procedure embedding worker."""

    source_procedure_description_results: list[SourceProcedureDescriptionResult]


class SourceProcedureEmbeddingNode:
    """Batch procedure descriptions and retain occurrence UUID alignment."""

    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client

    def dispatch(
        self, state: SemanticState
    ) -> list[Send] | Literal['source_procedure_embedding_collect']:
        """Dispatch all procedure descriptions as one ordered batch."""
        if not state.source_procedure_description_results:
            return 'source_procedure_embedding_collect'
        return [
            Send(
                'source_procedure_embedding_worker',
                {
                    'source_procedure_description_results': state.source_procedure_description_results,
                },
            )
        ]

    async def worker(
        self, state: SourceProcedureEmbeddingWorkerState
    ) -> dict[str, list[SourceProcedureDescriptionResult]]:
        """Embed procedure descriptions in input order."""
        results = state['source_procedure_description_results']
        vectors = await self._client.embed(
            [result.description for result in results]
        )
        return {
            'source_procedure_embedding_results': [
                result.model_copy(update={'embedding': vector})
                for result, vector in zip(results, vectors, strict=True)
            ]
        }

    def collect(self, state: SemanticState) -> dict[str, object]:
        """Finish procedure embedding fan-in before persistence."""
        return {}
