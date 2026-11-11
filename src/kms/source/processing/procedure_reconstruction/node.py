"""Reconstruct complete procedures from ordered source evidence."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms.core.tokenization.budget import TokenBudget
from kms.inference.batching import pack_content_batches
from kms.media.budget import ImagePayloadBudget
from kms.source.models.material import ProcedureDraft
from kms.source.processing.procedure_reconstruction.inference import (
    SourceProcedureReconstructionModule,
)
from kms.source.processing.procedure_reconstruction.models import (
    SourceProcedureReconstructionItem,
    SourceProcedureReconstructionRequest,
)
from kms.source.processing.state import SourceProcessingState


class SourceProcedureReconstructionWorkerState(TypedDict):
    """State supplied to one procedure reconstruction packet worker."""

    source_procedure_reconstruction_requests: list[
        SourceProcedureReconstructionRequest
    ]


class SourceProcedureReconstructionNode:
    """Prepare, dispatch, and collect procedure reconstruction packets."""

    def __init__(
        self,
        module: SourceProcedureReconstructionModule,
        *,
        budget: TokenBudget,
        batch_size: int,
        image_budget: ImagePayloadBudget,
    ) -> None:
        self._module = module
        self._budget = budget
        self._batch_size = batch_size
        self._image_budget = image_budget

    def run(self, state: SourceProcessingState) -> dict[str, object]:
        """Retain canonical procedure member blocks through dispatch."""
        blocks = [block for page in state.split_pages for block in page.blocks]
        positions = {
            block.uuid: position for position, block in enumerate(blocks)
        }
        requests = []
        for procedure in state.procedures:
            member_uuids = set(procedure.member_block_uuids)
            requests.append(
                SourceProcedureReconstructionRequest(
                    target=procedure,
                    target_blocks=[
                        block for block in blocks if block.uuid in member_uuids
                    ],
                )
            )
        requests.sort(
            key=lambda request: min(
                positions[uuid] for uuid in request.target.member_block_uuids
            )
        )
        return {'source_procedure_reconstruction_requests': requests}

    async def dispatch(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['source_procedure_reconstruction_collect']:
        """Dispatch complete requests within text and image payload limits."""
        requests = state.source_procedure_reconstruction_requests
        if not requests:
            return 'source_procedure_reconstruction_collect'

        def project(
            batch: list[SourceProcedureReconstructionRequest],
        ) -> dict[str, object]:
            return {
                'items': [
                    SourceProcedureReconstructionItem(
                        position=position,
                        request=request.model_input(),
                    )
                    for position, request in enumerate(batch)
                ]
            }

        incoming = [
            requests[position : position + self._batch_size]
            for position in range(0, len(requests), self._batch_size)
        ]
        batches = await pack_content_batches(
            incoming,
            project=project,
            budget=self._budget,
            image_budget=self._image_budget,
        )
        return [
            Send(
                'source_procedure_reconstruction_worker',
                {'source_procedure_reconstruction_requests': batch},
            )
            for batch in batches
        ]

    async def worker(
        self, state: SourceProcedureReconstructionWorkerState
    ) -> dict[str, list[ProcedureDraft]]:
        """Reconstruct packet items and map outputs to their original drafts."""
        requests = state['source_procedure_reconstruction_requests']
        items = [
            SourceProcedureReconstructionItem(
                position=position,
                request=request.model_input(),
            )
            for position, request in enumerate(requests)
        ]
        outputs = await self._module.acall(items=items)
        content_by_position = {
            output.position: output.content for output in outputs
        }
        return {
            'source_procedure_reconstruction_results': [
                request.target.model_copy(
                    update={
                        'content': content_by_position[position],
                        'embedding': None,
                    }
                )
                for position, request in enumerate(requests)
            ]
        }

    def collect(self, state: SourceProcessingState) -> dict[str, object]:
        """Restore the original procedure order after packet fan-in."""
        by_uuid = {
            procedure.uuid: procedure
            for procedure in state.source_procedure_reconstruction_results
        }
        return {
            'procedures': [
                by_uuid[procedure.uuid] for procedure in state.procedures
            ]
        }
