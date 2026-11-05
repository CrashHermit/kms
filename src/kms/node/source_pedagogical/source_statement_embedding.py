"""Embed reconstructed source statements."""

from kms.core.embedding import Embedder
from kms.core.model.content import MultimodalContent
from kms.langgraph.source_pedagogical.state import SourcePedagogicalState


class SourceStatementEmbeddingNode:
    """Embed exact reconstructed statement content in source order."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def run(self, state: SourcePedagogicalState) -> dict[str, object]:
        """Attach vectors to reconstructed statements without changing identity."""
        if not state.statements:
            return {'statements': []}
        vectors = await self._embedder.embed(
            [
                MultimodalContent(text=statement.content)
                for statement in state.statements
            ],
            content_images=[],
            input_type='document',
        )
        return {
            'statements': [
                statement.model_copy(update={'embedding': vector})
                for statement, vector in zip(
                    state.statements, vectors, strict=True
                )
            ]
        }
