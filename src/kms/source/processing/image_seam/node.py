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
from kms.source.models.source import BlockType, SourceBlock, SourcePage
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
    """Return whether a blank IMAGE block owns one canonical image."""
    return bool(
        block.block_type is BlockType.IMAGE
        and block.image_path
        and not (block.content or '').strip()
    )


def _edge_index(blocks: list[SourceBlock], *, reverse: bool) -> int | None:
    """Return the nearest eligible image edge, skipping apparatus blocks."""
    indices = range(len(blocks) - 1, -1, -1) if reverse else range(len(blocks))
    for index in indices:
        block = blocks[index]
        if block.block_type in _APPARATUS:
            continue
        return index if _image_mergeable(block) else None
    return None


class ImageSeamNode:
    """Judge original adjacent-page seams, then fuse accepted image chains."""

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
    def _pairs(pages: list[SourcePage]) -> list[ImageSeamRequest]:
        requests = []
        for index in range(len(pages) - 1):
            top, bottom = pages[index], pages[index + 1]
            if (
                bottom.index == top.index + 1
                and _edge_index(top.blocks, reverse=True) is not None
                and _edge_index(bottom.blocks, reverse=False) is not None
            ):
                requests.append(
                    ImageSeamRequest(
                        position=len(requests),
                        top_page=top,
                        bottom_page=bottom,
                    )
                )
        return requests

    @staticmethod
    def _edge_positions(request: ImageSeamRequest) -> tuple[int, int]:
        return (
            cast(int, _edge_index(request.top_page.blocks, reverse=True)),
            cast(int, _edge_index(request.bottom_page.blocks, reverse=False)),
        )

    def _project(self, requests: list[ImageSeamRequest]) -> dict[str, object]:
        items = []
        for local_position, request in enumerate(requests):
            top_position, bottom_position = self._edge_positions(request)
            top = project_source_block(
                request.top_page.blocks[top_position]
            ).material
            bottom = project_source_block(
                request.bottom_page.blocks[bottom_position]
            ).material
            items.append(
                ImageSeamInput(
                    position=local_position,
                    top=Material(
                        parts=tuple(
                            ImagePart(asset=asset) for asset in top.images
                        )
                    ),
                    bottom=Material(
                        parts=tuple(
                            ImagePart(asset=asset) for asset in bottom.images
                        )
                    ),
                )
            )
        return {'items': items}

    async def dispatch(
        self, state: SourceProcessingState
    ) -> list[Send] | Literal['image_seam_collect']:
        """Dispatch all eligible original adjacent pairs without mutation."""
        requests = self._pairs(state.text_seam_pages)
        if not requests:
            return 'image_seam_collect'
        coarse_batches = pack_items(
            requests,
            token_counts=[1] * len(requests),
            token_budget=self._batch_size,
            max_items=self._batch_size,
        )
        batches = await pack_content_batches(
            coarse_batches,
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
        """Judge one packet and map packet-local decisions to candidate positions."""
        requests = state['image_seam_requests']
        items = cast(list[ImageSeamInput], self._project(requests)['items'])
        decisions = await self._judge.acall(items=items)
        decision_by_position = {
            decision.position: decision.is_continuation
            for decision in decisions
        }
        return {
            'image_seam_results': [
                ImageSeamResult(
                    position=request.position,
                    is_continuation=decision_by_position[local_position],
                )
                for local_position, request in enumerate(requests)
            ]
        }

    async def collect(
        self, state: SourceProcessingState
    ) -> dict[str, list[SourcePage]]:
        """Fold accepted original edges into maximal chains and fuse each once."""
        pages = state.text_seam_pages
        requests = self._pairs(pages)
        decisions = {
            result.position: result.is_continuation
            for result in state.image_seam_results
        }
        chains: list[list[tuple[int, SourceBlock]]] = []
        chain_locations: list[list[tuple[int, int]]] = []
        for request in requests:
            if not decisions[request.position]:
                continue
            top_position, bottom_position = self._edge_positions(request)
            top_block = request.top_page.blocks[top_position]
            bottom_block = request.bottom_page.blocks[bottom_position]
            first = (request.top_page.index, top_position)
            second = (request.bottom_page.index, bottom_position)
            part_pair = [
                (request.top_page.index, top_block),
                (request.bottom_page.index, bottom_block),
            ]
            if chain_locations and chain_locations[-1][-1] == first:
                chain_locations[-1].append(second)
                chains[-1].append(part_pair[-1])
            else:
                chain_locations.append([first, second])
                chains.append(part_pair)

        replacements: dict[tuple[int, int], SourceBlock] = {}
        removed: set[tuple[int, int]] = set()
        for locations, parts in zip(chain_locations, chains, strict=True):
            if len(parts) < 2:
                continue
            fused = await fuse_image_blocks(
                parts,
                page_images=state.ocr_page_images,
                budget=self._image_budget,
            )
            replacements[locations[0]] = fused
            removed.update(locations[1:])

        output_pages = []
        for page in pages:
            blocks = []
            for block_position, block in enumerate(page.blocks):
                location = (page.index, block_position)
                if location in removed:
                    continue
                blocks.append(replacements.get(location, block))
            output_pages.append(page.model_copy(update={'blocks': blocks}))
        return {'image_seam_pages': output_pages}
