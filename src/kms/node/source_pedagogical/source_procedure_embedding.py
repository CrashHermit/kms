"""Embed reconstructed source procedures."""

from kms.core.embedding import Embedder
from kms.core.model.content import MultimodalContent
from kms.langgraph.source_pedagogical.state import SourcePedagogicalState


class SourceProcedureEmbeddingNode:
    """Embed exact reconstructed procedure content in source order."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def run(self, state: SourcePedagogicalState) -> dict[str, object]:
        """Attach vectors to reconstructed procedures without changing identity."""
        if not state.procedures:
            return {'procedures': []}
        vectors = await self._embedder.embed(
            [
                MultimodalContent(text=procedure.content)
                for procedure in state.procedures
            ],
            content_images=[],
            input_type='document',
        )
        return {
            'procedures': [
                procedure.model_copy(update={'embedding': vector})
                for procedure, vector in zip(
                    state.procedures, vectors, strict=True
                )
            ]
        }
