"""Reconstruct complete procedures from ordered source evidence."""

import json
from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model.source_pedagogical.pedagogical import ProcedureDraft
from kms2.core.model.source_pedagogical.source_procedure_reconstruction import (
    SourceProcedureReconstructionItem,
    SourceProcedureReconstructionRequest,
)
from kms2.core.windowing import TokenBudget, pack_items, project_block
from kms2.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms2.module.source_pedagogical.source_procedure_reconstruction import (
    SourceProcedureReconstructionModule,
)


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
    ) -> None:
        self._module = module
        self._budget = budget
        self._batch_size = batch_size

    def run(self, state: SourcePedagogicalState) -> dict[str, object]:
        """Project each procedure's ordered member blocks only."""
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
                    model_input={
                        'target_blocks': [
                            project_block(block)
                            for block in blocks
                            if block.uuid in member_uuids
                        ]
                    },
                )
            )
        requests.sort(
            key=lambda request: min(
                positions[uuid] for uuid in request.target.member_block_uuids
            )
        )
        return {'source_procedure_reconstruction_requests': requests}

    def dispatch(
        self, state: SourcePedagogicalState
    ) -> list[Send] | Literal['source_procedure_reconstruction_collect']:
        """Dispatch complete procedure requests in bounded ordered packets."""
        requests = state.source_procedure_reconstruction_requests
        if not requests:
            return 'source_procedure_reconstruction_collect'
        item_costs = self._budget.counter.count_texts(
            [
                json.dumps(
                    {
                        'position': self._batch_size - 1,
                        'request': request.model_input.model_dump(mode='json'),
                    },
                    ensure_ascii=False,
                    separators=(',', ':'),
                )
                for request in requests
            ]
        )
        list_overhead = self._budget.counter.count_texts(['[,]'])[0]
        envelope_cost = self._budget.counter.count_texts(['{"items":[]}'])[0]
        batches = pack_items(
            requests,
            token_counts=[cost + list_overhead for cost in item_costs],
            token_budget=self._budget.token_limit - envelope_cost,
            max_items=self._batch_size,
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
                request=request.model_input,
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

    def collect(self, state: SourcePedagogicalState) -> dict[str, object]:
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
