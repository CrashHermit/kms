"""Embed reconstructed source procedures with their exact image evidence."""

from kms.core.embedding import Embedder, EmbeddingRequest
from kms.source.context import project_source_material
from kms.source.processing.state import SourceProcessingState


class SourceProcedureEmbeddingNode:
    """Embed exact reconstructed procedure content in source order."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    async def run(self, state: SourceProcessingState) -> dict[str, object]:
        """Attach vectors using each procedure's target block evidence."""
        if not state.procedures:
            return {'procedures': []}
        requests = {
            request.target.uuid: request
            for request in state.source_procedure_reconstruction_requests
        }
        materials = [
            project_source_material(
                procedure.content,
                requests[procedure.uuid].target_blocks,
            )
            for procedure in state.procedures
        ]
        vectors = await self._embedder.embed(
            EmbeddingRequest(materials=materials, input_type='document')
        )
        return {
            'procedures': [
                procedure.model_copy(update={'embedding': vector})
                for procedure, vector in zip(
                    state.procedures, vectors, strict=True
                )
            ]
        }
