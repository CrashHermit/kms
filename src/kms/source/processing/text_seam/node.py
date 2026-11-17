"""LangGraph node for flat source-block text seam processing."""

from typing import Literal, TypedDict

from langgraph.types import Send

from kms.core.tokenization.budget import TokenBudget
from kms.inference.batching import pack_content_batches
from kms.source.context import project_source_block
from kms.source.models.source import BlockType, SourceBlock
from kms.source.processing.state import SourceProcessingState
from kms.source.processing.text_seam.inference import (
    TextSeamJudgeModule,
    TextSeamRewriterModule,
)
from kms.source.processing.text_seam.models import (
    TextSeamInput,
    TextSeamRequest,
    TextSeamResult,
    TextSeamRewriteInput,
    TextSeamRewriteRequest,
    TextSeamRewriteResult,
)


class TextSeamWorkerState(TypedDict):
    """State supplied to one dispatched seam-judge worker."""

    text_seam_requests: list[TextSeamRequest]


class TextSeamRewriteWorkerState(TypedDict):
    """State supplied to one dispatched chain-rewrite worker."""

    text_seam_rewrite_requests: list[TextSeamRewriteRequest]


def _skip(block: SourceBlock) -> bool:
    return block.block_type in {
        BlockType.BIBLIOGRAPHIC,
        BlockType.NOTE,
        BlockType.IMAGE,
    }


def _edge(blocks: list[SourceBlock], *, reverse: bool) -> int | None:
    indexes = range(len(blocks) - 1, -1, -1) if reverse else range(len(blocks))
    for index in indexes:
        block = blocks[index]
        if _skip(block):
            continue
        return index if block.content else None
    return None


def _context(
    blocks: list[SourceBlock], start: int, step: int
) -> SourceBlock | None:
    for index in range(start, len(blocks) if step > 0 else -1, step):
        block = blocks[index]
        if not _skip(block) and block.content:
            return block
    return None


class TextSeamNode:
    """Judge original page edges, then rewrite each maximal accepted chain."""

    def __init__(
        self,
        judge: TextSeamJudgeModule,
        rewriter: TextSeamRewriterModule,
        *,
        judge_budget: TokenBudget,
        rewrite_budget: TokenBudget,
        judge_batch_size: int,
        rewrite_batch_size: int,
    ) -> None:
        self._judge = judge
        self._rewriter = rewriter
        self._judge_budget = judge_budget
        self._rewrite_budget = rewrite_budget
        self._judge_batch_size = judge_batch_size
        self._rewrite_batch_size = rewrite_batch_size

    @staticmethod
    def _page_blocks(
        blocks: list[SourceBlock],
        block_pages: dict[str, int],
        page_index: int,
    ) -> list[SourceBlock]:
        return [
            block for block in blocks if block_pages[block.uuid] == page_index
        ]

    @classmethod
    def _requests(cls, state: SourceProcessingState) -> list[TextSeamRequest]:
        blocks = state.formatted_blocks
        requests = []
        for top_page, bottom_page in zip(
            state.selected_page_indexes,
            state.selected_page_indexes[1:],
            strict=False,
        ):
            if bottom_page != top_page + 1:
                continue
            top = cls._page_blocks(blocks, state.ocr_block_pages, top_page)
            bottom = cls._page_blocks(
                blocks, state.ocr_block_pages, bottom_page
            )
            tail_index = _edge(top, reverse=True)
            head_index = _edge(bottom, reverse=False)
            if tail_index is None or head_index is None:
                continue
            requests.append(
                TextSeamRequest(
                    tail=top[tail_index],
                    head=bottom[head_index],
                    top_context=_context(top, tail_index - 1, -1),
                    bottom_context=_context(bottom, head_index + 1, 1),
                )
            )
        return requests

    @staticmethod
    def _judge_inputs(
        requests: list[TextSeamRequest],
    ) -> list[TextSeamInput]:
        return [
            TextSeamInput(
                position=position,
                tail=project_source_block(request.tail),
                head=project_source_block(request.head),
                top_context=(
                    project_source_block(request.top_context)
                    if request.top_context
                    else None
                ),
                bottom_context=(
                    project_source_block(request.bottom_context)
                    if request.bottom_context
                    else None
                ),
            )
            for position, request in enumerate(requests)
        ]

    async def dispatch(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['text_seam_collect']:
        requests = self._requests(state)
        if not requests:
            return 'text_seam_collect'
        coarse = [
            requests[index : index + self._judge_batch_size]
            for index in range(0, len(requests), self._judge_batch_size)
        ]
        batches = await pack_content_batches(
            coarse,
            project=lambda batch: {'items': self._judge_inputs(batch)},
            budget=self._judge_budget,
            image_budget=None,
        )
        return [
            Send('text_seam_judge_worker', {'text_seam_requests': batch})
            for batch in batches
        ]

    async def worker(
        self, state: TextSeamWorkerState
    ) -> dict[str, list[TextSeamResult]]:
        requests = state['text_seam_requests']
        decisions = await self._judge.acall(items=self._judge_inputs(requests))
        by_position = {item.position: item.is_split for item in decisions}
        return {
            'text_seam_results': [
                TextSeamResult(
                    tail_uuid=request.tail.uuid,
                    head_uuid=request.head.uuid,
                    is_split=by_position[position],
                )
                for position, request in enumerate(requests)
            ]
        }

    def prepare_rewrites(
        self, state: SourceProcessingState
    ) -> dict[str, list[TextSeamRewriteRequest]]:
        """Fold accepted original edges into maximal ordered block chains."""
        requests = self._requests(state)
        accepted = {
            (result.tail_uuid, result.head_uuid)
            for result in state.text_seam_results
            if result.is_split
        }
        chains: list[list[TextSeamRequest]] = []
        for request in requests:
            edge = (request.tail.uuid, request.head.uuid)
            if edge not in accepted:
                continue
            if chains and chains[-1][-1].head.uuid == request.tail.uuid:
                chains[-1].append(request)
            else:
                chains.append([request])
        rewrite_requests = [
            TextSeamRewriteRequest(
                blocks=[chain[0].tail, *(item.head for item in chain)],
                top_context=chain[0].top_context,
                bottom_context=chain[-1].bottom_context,
            )
            for chain in chains
        ]
        return {'text_seam_rewrite_requests': rewrite_requests}

    @staticmethod
    def _rewrite_inputs(
        requests: list[TextSeamRewriteRequest],
    ) -> list[TextSeamRewriteInput]:
        return [
            TextSeamRewriteInput(
                position=position,
                blocks=[
                    project_source_block(block) for block in request.blocks
                ],
                top_context=(
                    project_source_block(request.top_context)
                    if request.top_context
                    else None
                ),
                bottom_context=(
                    project_source_block(request.bottom_context)
                    if request.bottom_context
                    else None
                ),
            )
            for position, request in enumerate(requests)
        ]

    async def dispatch_rewrites(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['text_seam_collect']:
        requests = state.text_seam_rewrite_requests
        if not requests:
            return 'text_seam_collect'
        coarse = [
            requests[index : index + self._rewrite_batch_size]
            for index in range(0, len(requests), self._rewrite_batch_size)
        ]
        batches = await pack_content_batches(
            coarse,
            project=lambda batch: {'items': self._rewrite_inputs(batch)},
            budget=self._rewrite_budget,
            image_budget=None,
        )
        return [
            Send(
                'text_seam_rewrite_worker',
                {'text_seam_rewrite_requests': batch},
            )
            for batch in batches
        ]

    async def rewrite_worker(
        self, state: TextSeamRewriteWorkerState
    ) -> dict[str, list[TextSeamRewriteResult]]:
        requests = state['text_seam_rewrite_requests']
        outputs = await self._rewriter.acall(
            items=self._rewrite_inputs(requests)
        )
        by_position = {item.position: item.merged for item in outputs}
        return {
            'text_seam_rewrite_results': [
                TextSeamRewriteResult(
                    block_uuids=[block.uuid for block in request.blocks],
                    merged=by_position[position],
                )
                for position, request in enumerate(requests)
            ]
        }

    def collect(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourceBlock]]:
        replacements = {
            result.block_uuids[0]: result
            for result in state.text_seam_rewrite_results
        }
        removed = {
            block_uuid
            for result in state.text_seam_rewrite_results
            for block_uuid in result.block_uuids[1:]
        }
        output = []
        for block in state.formatted_blocks:
            if block.uuid in removed:
                continue
            result = replacements.get(block.uuid)
            if result is None:
                output.append(block)
                continue
            output.append(
                block.model_copy(
                    update={'content': result.merged, 'embedding': None}
                )
            )
        return {'text_seam_blocks': output}
