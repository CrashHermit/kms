"""LangGraph node for judging and physically fusing image seams."""

from typing import Literal, TypedDict, cast

from langgraph.types import Send

from kms.core.batching import pack_items
from kms.core.tokenization.budget import TokenBudget
from kms.inference.batching import pack_content_batches
from kms.media.budget import ImagePayloadBudget
from kms.media.models import ImagePart, Material
from kms.source.context import project_source_block
from kms.source.images import fuse_image_blocks
from kms.source.models.source import BlockType, SourceBlock
from kms.source.processing.image_seam.inference import ImageSeamJudgeModule
from kms.source.processing.image_seam.models import (
    ImageSeamInput,
    ImageSeamRequest,
    ImageSeamResult,
)
from kms.source.processing.state import SourceProcessingState


class ImageSeamWorkerState(TypedDict):
    """State supplied to one dispatched image seam worker."""

    image_seam_requests: list[ImageSeamRequest]


_APPARATUS = {BlockType.BIBLIOGRAPHIC, BlockType.NOTE}


def _image_mergeable(block: SourceBlock) -> bool:
    return bool(
        block.block_type is BlockType.IMAGE
        and block.image_path
        and not (block.content or '').strip()
    )


def _edge(blocks: list[SourceBlock], *, reverse: bool) -> SourceBlock | None:
    indices = range(len(blocks) - 1, -1, -1) if reverse else range(len(blocks))
    for index in indices:
        block = blocks[index]
        if block.block_type in _APPARATUS:
            continue
        return block if _image_mergeable(block) else None
    return None


class ImageSeamNode:
    """Judge original physical edges, then fuse accepted image chains."""

    def __init__(
        self,
        judge: ImageSeamJudgeModule,
        *,
        budget: TokenBudget,
        image_budget: ImagePayloadBudget,
        batch_size: int,
    ) -> None:
        self._judge = judge
        self._budget = budget
        self._image_budget = image_budget
        self._batch_size = batch_size

    @staticmethod
    def _anchored(
        blocks: list[SourceBlock],
        block_pages: dict[str, int],
        page_index: int,
    ) -> list[SourceBlock]:
        return [
            block for block in blocks if block_pages[block.uuid] == page_index
        ]

    @classmethod
    def _pairs(cls, state: SourceProcessingState) -> list[ImageSeamRequest]:
        blocks = state.text_seam_blocks
        requests = []
        for top_page, bottom_page in zip(
            state.selected_page_indexes,
            state.selected_page_indexes[1:],
            strict=False,
        ):
            if bottom_page != top_page + 1:
                continue
            top = _edge(
                cls._anchored(blocks, state.ocr_block_pages, top_page),
                reverse=True,
            )
            bottom = _edge(
                cls._anchored(blocks, state.ocr_block_pages, bottom_page),
                reverse=False,
            )
            if top is not None and bottom is not None:
                requests.append(
                    ImageSeamRequest(top_block=top, bottom_block=bottom)
                )
        return requests

    def _project(self, requests: list[ImageSeamRequest]) -> dict[str, object]:
        items = []
        for position, request in enumerate(requests):
            top = project_source_block(request.top_block).material
            bottom = project_source_block(request.bottom_block).material
            items.append(
                ImageSeamInput(
                    position=position,
                    top=Material(
                        parts=tuple(ImagePart(asset=a) for a in top.images)
                    ),
                    bottom=Material(
                        parts=tuple(ImagePart(asset=a) for a in bottom.images)
                    ),
                )
            )
        return {'items': items}

    async def dispatch(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['image_seam_collect']:
        requests = self._pairs(state)
        if not requests:
            return 'image_seam_collect'
        coarse = pack_items(
            requests,
            token_counts=[1] * len(requests),
            token_budget=self._batch_size,
            max_items=self._batch_size,
        )
        batches = await pack_content_batches(
            coarse,
            project=self._project,
            budget=self._budget,
            image_budget=self._image_budget,
        )
        return [
            Send('image_seam_worker', {'image_seam_requests': batch})
            for batch in batches
        ]

    async def worker(
        self, state: ImageSeamWorkerState
    ) -> dict[str, list[ImageSeamResult]]:
        requests = state['image_seam_requests']
        items = cast(list[ImageSeamInput], self._project(requests)['items'])
        decisions = await self._judge.acall(items=items)
        by_position = {d.position: d.is_continuation for d in decisions}
        return {
            'image_seam_results': [
                ImageSeamResult(
                    top_uuid=request.top_block.uuid,
                    bottom_uuid=request.bottom_block.uuid,
                    is_continuation=by_position[position],
                )
                for position, request in enumerate(requests)
            ]
        }

    async def collect(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourceBlock]]:
        requests = self._pairs(state)
        accepted = {
            (result.top_uuid, result.bottom_uuid)
            for result in state.image_seam_results
            if result.is_continuation
        }
        by_uuid = {block.uuid: block for block in state.text_seam_blocks}
        chains: list[list[SourceBlock]] = []
        for request in requests:
            pair = (request.top_block.uuid, request.bottom_block.uuid)
            if pair not in accepted:
                continue
            if chains and chains[-1][-1].uuid == pair[0]:
                chains[-1].append(request.bottom_block)
            else:
                chains.append([request.top_block, request.bottom_block])

        replacements: dict[str, SourceBlock] = {}
        removed: set[str] = set()
        for chain in chains:
            parts = [
                (state.ocr_block_pages[block.uuid], block) for block in chain
            ]
            replacements[chain[0].uuid] = await fuse_image_blocks(
                parts,
                page_images=state.ocr_page_images,
                budget=self._image_budget,
            )
            removed.update(block.uuid for block in chain[1:])

        output = []
        for block in state.text_seam_blocks:
            if block.uuid in removed:
                continue
            output.append(replacements.get(block.uuid, by_uuid[block.uuid]))
        return {'image_seam_blocks': output}
