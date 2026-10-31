"""Embed reconstructed source statements."""

from kms2.core.embedding import EmbeddingClient
from kms2.langgraph.source_pedagogical.state import SourcePedagogicalState


class SourceStatementEmbeddingNode:
    """Embed exact reconstructed statement content in source order."""

    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client

    async def run(self, state: SourcePedagogicalState) -> dict[str, object]:
        """Attach vectors to reconstructed statements without changing identity."""
        if not state.statements:
            return {'statements': []}
        vectors = await self._client.embed(
            [statement.content for statement in state.statements]
        )
        return {
            'statements': [
                statement.model_copy(update={'embedding': vector})
                for statement, vector in zip(
                    state.statements, vectors, strict=True
                )
            ]
        }
