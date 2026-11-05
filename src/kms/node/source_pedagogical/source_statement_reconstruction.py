"""Reconstruct complete statements from ordered source evidence."""

from typing import Literal, TypedDict

import dspy
from langgraph.types import Send

from kms.core.content import MultimodalContentBuilder
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.source_pedagogical.pedagogical import StatementDraft
from kms.core.model.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionItem,
    SourceStatementReconstructionRequest,
)
from kms.core.tokenization.budget import TokenBudget
from kms.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms.module.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionModule,
)
from kms.service.content import (
    load_content_images,
    pack_content_batches,
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
        image_budget: ImagePayloadBudget,
    ) -> None:
        self._module = module
        self._budget = budget
        self._batch_size = batch_size
        self._image_budget = image_budget

    def run(self, state: SourcePedagogicalState) -> dict[str, object]:
        """Retain canonical statement and governing blocks through dispatch."""
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
            request_blocks = [
                block for block in blocks if block.uuid in target_uuids
            ]
            instruction_blocks = [
                block for block in blocks if block.uuid in governed_uuids
            ]
            requests.append(
                SourceStatementReconstructionRequest(
                    target=statement,
                    target_blocks=request_blocks,
                    instruction_blocks=instruction_blocks,
                )
            )
        requests.sort(
            key=lambda request: min(
                positions[uuid] for uuid in request.target.member_block_uuids
            )
        )
        return {'source_statement_reconstruction_requests': requests}

    async def dispatch(
        self, state: SourcePedagogicalState
    ) -> list[Send] | Literal['source_statement_reconstruction_collect']:
        """Dispatch complete requests within text and image payload limits."""
        requests = state.source_statement_reconstruction_requests
        if not requests:
            return 'source_statement_reconstruction_collect'

        def project(
            batch: list[SourceStatementReconstructionRequest],
            builder: MultimodalContentBuilder,
        ) -> dict[str, object]:
            return {
                'items': [
                    SourceStatementReconstructionItem(
                        position=position,
                        request=request.model_input(builder),
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
        builder = MultimodalContentBuilder()
        items = [
            SourceStatementReconstructionItem(
                position=position,
                request=request.model_input(builder),
            )
            for position, request in enumerate(requests)
        ]
        payloads = await load_content_images(
            builder.images, budget=self._image_budget
        )
        outputs = await self._module.acall(
            items=items,
            content_images=[dspy.Image(payload) for payload in payloads],
        )
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
