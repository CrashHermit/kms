"""Embed reconstructed source procedures."""

from kms2.core.embedding import EmbeddingClient
from kms2.langgraph.source_pedagogical.state import SourcePedagogicalState


class SourceProcedureEmbeddingNode:
    """Embed exact reconstructed procedure content in source order."""

    def __init__(self, client: EmbeddingClient) -> None:
        self._client = client

    async def run(self, state: SourcePedagogicalState) -> dict[str, object]:
        """Attach vectors to reconstructed procedures without changing identity."""
        if not state.procedures:
            return {'procedures': []}
        vectors = await self._client.embed(
            [procedure.content for procedure in state.procedures]
        )
        return {
            'procedures': [
                procedure.model_copy(update={'embedding': vector})
                for procedure, vector in zip(
                    state.procedures, vectors, strict=True
                )
            ]
        }
