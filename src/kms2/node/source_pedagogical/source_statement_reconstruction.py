"""Reconstruct complete statements from ordered source evidence."""

import json
from typing import Literal, TypedDict

from langgraph.types import Send

from kms2.core.model.source_pedagogical.pedagogical import StatementDraft
from kms2.core.model.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionItem,
    SourceStatementReconstructionRequest,
)
from kms2.core.windowing import TokenBudget, pack_items, project_block
from kms2.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms2.module.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionModule,
)


class SourceStatementReconstructionWorkerState(TypedDict):
    """State supplied to one statement reconstruction packet worker."""

    source_statement_reconstruction_requests: list[
        SourceStatementReconstructionRequest
    ]


class SourceStatementReconstructionNode:
    """Prepare, dispatch, and collect statement reconstruction packets."""

    def __init__(
        self,
        module: SourceStatementReconstructionModule,
        *,
        budget: TokenBudget,
        batch_size: int,
    ) -> None:
        self._module = module
        self._budget = budget
        self._batch_size = batch_size

    def run(self, state: SourcePedagogicalState) -> dict[str, object]:
        """Project each statement's own blocks and governing instructions."""
        blocks = [block for page in state.split_pages for block in page.blocks]
        positions = {
            block.uuid: position for position, block in enumerate(blocks)
        }
        requests = []
        for statement in state.statements:
            target_uuids = set(statement.member_block_uuids)
            governed_uuids = {
                uuid
                for instruction in state.instructions
                if statement.uuid in instruction.governed_statement_uuids
                for uuid in instruction.member_block_uuids
            }
            target_blocks = [
                project_block(block)
                for block in blocks
                if block.uuid in target_uuids
            ]
            instruction_blocks = [
                project_block(block)
                for block in blocks
                if block.uuid in governed_uuids
            ]
            requests.append(
                SourceStatementReconstructionRequest(
                    target=statement,
                    model_input={
                        'target_blocks': target_blocks,
                        'instruction_blocks': instruction_blocks,
                    },
                )
            )
        requests.sort(
            key=lambda request: min(
                positions[uuid] for uuid in request.target.member_block_uuids
            )
        )
        return {'source_statement_reconstruction_requests': requests}

    def dispatch(
        self, state: SourcePedagogicalState
    ) -> list[Send] | Literal['source_statement_reconstruction_collect']:
        """Dispatch complete statement requests in bounded ordered packets."""
        requests = state.source_statement_reconstruction_requests
        if not requests:
            return 'source_statement_reconstruction_collect'
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
                'source_statement_reconstruction_worker',
                {'source_statement_reconstruction_requests': batch},
            )
            for batch in batches
        ]

    async def worker(
        self, state: SourceStatementReconstructionWorkerState
    ) -> dict[str, list[StatementDraft]]:
        """Reconstruct packet items and map outputs to their original drafts."""
        requests = state['source_statement_reconstruction_requests']
        items = [
            SourceStatementReconstructionItem(
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
            'source_statement_reconstruction_results': [
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
        """Restore the original statement order after packet fan-in."""
        by_uuid = {
            statement.uuid: statement
            for statement in state.source_statement_reconstruction_results
        }
        return {
            'statements': [
                by_uuid[statement.uuid] for statement in state.statements
            ]
        }
