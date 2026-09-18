"""Embed source statement descriptions in a LangGraph phase."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.embedding import EmbeddingClient
from kms2.core.model import SourceStatementDescriptionResult
from kms2.langgraph.semantic.state import SemanticState


class SourceStatementEmbeddingWorkerState(TypedDict):
    """State supplied to the statement embedding worker."""

    source_statement_description_results: list[SourceStatementDescriptionResult]


class SourceStatementEmbeddingNode:
    """Batch statement descriptions and retain occurrence UUID alignment."""

    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client

    def dispatch(
        self, state: SemanticState
    ) -> list[Send] | Literal['source_statement_embedding_collect']:
        """Dispatch all statement descriptions as one ordered batch."""
        if not state.source_statement_description_results:
            return 'source_statement_embedding_collect'
        return [
            Send(
                'source_statement_embedding_worker',
                {
                    'source_statement_description_results': state.source_statement_description_results,
                },
            )
        ]

    async def worker(
        self, state: SourceStatementEmbeddingWorkerState
    ) -> dict[str, list[SourceStatementDescriptionResult]]:
        """Embed statement descriptions in input order."""
        results = state['source_statement_description_results']
        vectors = await self._client.embed(
            [result.description for result in results]
        )
        return {
            'source_statement_embedding_results': [
                result.model_copy(update={'embedding': vector})
                for result, vector in zip(results, vectors, strict=True)
            ]
        }

    def collect(self, state: SemanticState) -> dict[str, object]:
        """Finish statement embedding fan-in before persistence."""
        return {}
