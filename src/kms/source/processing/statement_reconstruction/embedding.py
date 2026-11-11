"""Embed reconstructed source statements with their exact image evidence."""

from kms.core.embedding import Embedder, EmbeddingRequest
from kms.source.context import project_source_material
from kms.source.processing.state import SourceProcessingState


class SourceStatementEmbeddingNode:
    """Embed exact reconstructed statement content in source order."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def run(self, state: SourceProcessingState) -> dict[str, object]:
        """Attach vectors using target and governing instruction evidence."""
        if not state.statements:
            return {'statements': []}
        requests = {
            request.target.uuid: request
            for request in state.source_statement_reconstruction_requests
        }
        materials = []
        for statement in state.statements:
            request = requests[statement.uuid]
            evidence = [*request.target_blocks, *request.instruction_blocks]
            materials.append(
                project_source_material(statement.content, evidence)
            )
        vectors = await self._embedder.embed(
            EmbeddingRequest(materials=materials, input_type='document')
        )
        return {
            'statements': [
                statement.model_copy(update={'embedding': vector})
                for statement, vector in zip(
                    state.statements, vectors, strict=True
                )
            ]
        }
